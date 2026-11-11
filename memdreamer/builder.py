"""Hierarchical Graph Memory construction (paper Sec 3.1): streaming adaptive segmentation ->
downward subgraph extraction -> upward hierarchical aggregation. Every stage is cached on disk."""
import json, logging, math, os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from .embedding import NodeIndex
from .memory import Memory, REL_TYPE
from .prompts import SEGMENT_PROMPT, SUBGRAPH_PROMPT, AGGREGATE_PROMPT, MACRO_SUMMARY_PROMPT, SYSTEM_VISION
from .video import Video, fmt_ts, parse_ts, vision_messages

log = logging.getLogger("memdreamer.build")


def _cached(path, fn):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    out = fn()
    with open(path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    return out


# ------------------------------------------------------------------ 1. streaming adaptive segmentation
def _clean_episodes(raw, t0, t1):
    if isinstance(raw, dict):
        raw = next((v for v in raw.values() if isinstance(v, list)), [])
    eps = []
    for r in raw if isinstance(raw, list) else []:
        if not isinstance(r, dict):
            continue
        s, e = parse_ts(r.get("start_time")), parse_ts(r.get("end_time"))
        if s is None or e is None:
            continue
        s, e = t0 + max(s, 0), t0 + max(e, 0)  # model answers are window-relative
        eps.append({"start": min(s, t1), "end": min(e, t1), "title": str(r.get("episode_title", "Segment"))})
    eps.sort(key=lambda x: x["start"])
    out = []
    for ep in eps:  # force contiguous, non-overlapping
        start = out[-1]["end"] if out else t0
        if ep["end"] - start >= 2:
            out.append({"start": start, "end": ep["end"], "title": ep["title"]})
    return out


def segment_video(video, llm, cfg):
    p = cfg["perception"]
    tau, t = float(p["window_sec"]), 0.0
    episodes = []
    while t < video.duration - 1:
        t1 = min(t + tau, video.duration)
        final = t1 >= video.duration - 1
        mos = video.mosaics(t, t1, tuple(p["grid"]), p["max_images"], p["tile_width"], quality=p["jpeg_quality"])
        prompt = SEGMENT_PROMPT.replace("<<DUR>>", fmt_ts(t1 - t))
        header = f"Video window {fmt_ts(t)}-{fmt_ts(t1)} of the full video. Timestamps are relative to this window."
        try:
            eps = _clean_episodes(llm.chat_json(vision_messages(mos, header, prompt)), t, t1)
        except Exception as e:
            log.warning("segmentation failed in window %.0f: %s", t, e)
            eps = []
        if not eps:
            eps = [{"start": t, "end": t1, "title": "Segment"}]
        if not final and p.get("carry_last_episode") and len(eps) > 1:
            eps = eps[:-1]
        if final:
            eps[-1]["end"] = t1
        episodes += eps
        nxt = eps[-1]["end"]  # concluding boundary of e_N = start of the next window
        t = nxt if nxt - t >= p["min_advance_sec"] else t1
        log.info("window -> %d episodes, next start %.0fs", len(eps), t)
    return episodes


# ------------------------------------------------------------------ 2. downward subgraph extraction
def _norm_subgraph(raw, mid, t0, t1, counters, name2id):
    """Normalise one chunk's JSON. Node ids are renumbered macro-wide (E#, M#, T#); named entities are
    de-duplicated across chunks of the same macro event."""
    raw = raw if isinstance(raw, dict) else {}
    idmap = {}

    def new(prefix):
        counters[prefix] += 1
        return f"{mid}:{prefix}{counters[prefix]}"

    def trange(x):
        x = x if isinstance(x, (list, tuple)) and len(x) >= 2 else [0, 0]
        a, b = parse_ts(x[0], 0), parse_ts(x[1], parse_ts(x[0], 0))
        a, b = min(t0 + a, t1), min(t0 + b, t1)
        return [a, max(a, b)]

    ents, evs, txs, edges = [], [], [], []
    for i, e in enumerate(raw.get("entities") or []):
        if not isinstance(e, dict):
            continue
        key = str(e.get("entity_id") or f"E{i + 1}").strip()
        name = e.get("name")
        nk = (str(name).strip().lower(), str(e.get("entity_type", "")).upper()) if name else None
        if nk and nk in name2id:
            idmap.setdefault(key, name2id[nk])
            continue
        g = e.get("visual_grounding")
        gt = parse_ts(g.get("primary_time_sec"), 0) if isinstance(g, dict) else 0
        nid = new("E")
        idmap.setdefault(key, nid)
        if nk:
            name2id[nk] = nid
        ents.append({"id": nid, "name": name, "entity_type": e.get("entity_type", "OBJECT"), "attributes": e.get("attributes", {}),
                     "description": e.get("description", ""), "time_sec": min(t0 + gt, t1)})
    for i, e in enumerate(raw.get("micro_events") or []):
        if isinstance(e, dict):
            nid = new("M")
            idmap.setdefault(str(e.get("event_id") or f"M{i + 1}").strip(), nid)
            evs.append({"id": nid, "event_type": e.get("event_type", ""), "time_range": trange(e.get("time_range")), "subject": e.get("subject", ""),
                        "object": e.get("object", ""), "action": e.get("action", ""), "description": e.get("description", "")})
    for i, e in enumerate(raw.get("on_screen_text") or []):
        if isinstance(e, dict):
            nid = new("T")
            idmap.setdefault(str(e.get("text_id") or f"T{i + 1}").strip(), nid)
            txs.append({"id": nid, "text": e.get("text", ""), "time_range": trange(e.get("time_range")), "description": e.get("description", "")})
    for e in raw.get("edges") or []:
        if not isinstance(e, dict):
            continue
        s_, t_, lab = idmap.get(str(e.get("source_id"))), idmap.get(str(e.get("target_id"))), str(e.get("relation_label", "")).upper()
        if s_ and t_ and s_ != t_ and lab in REL_TYPE:
            edges.append({"source": s_, "target": t_, "label": lab, "type": REL_TYPE[lab], "description": e.get("description", "")})
    return {"entities": ents, "events": evs, "texts": txs, "edges": edges}


def extract_subgraph(video, ep, mid, llm, cfg):
    """Macro Event -> chunks of `chunk_sec` (<=45 frames each, since the API takes <=5 images) -> merged subgraph."""
    p = cfg["perception"]
    t0, t1 = ep["start"], ep["end"]
    n = max(1, math.ceil((t1 - t0) / p["chunk_sec"]))
    bounds = [t0 + (t1 - t0) * i / n for i in range(n + 1)]
    counters, name2id = Counter(), {}
    g = {"entities": [], "events": [], "texts": [], "edges": []}
    labels, summaries = [], []
    for i in range(n):
        c0, c1 = bounds[i], bounds[i + 1]
        mos = video.mosaics(c0, c1, tuple(p["grid"]), p["max_images"], p["tile_width"], quality=p["jpeg_quality"])
        header = f"Video segment of length {fmt_ts(c1 - c0)} (chunk {i + 1}/{n} of an episode). Timestamps are relative to the segment start (00:00:00)."
        try:
            raw = llm.chat_json(vision_messages(mos, header, SUBGRAPH_PROMPT))
        except Exception as e:
            log.warning("subgraph extraction failed for %s chunk %d: %s", mid, i, e)
            continue
        part = _norm_subgraph(raw, mid, c0, c1, counters, name2id)
        for k in g:
            g[k] += part[k]
        if isinstance(raw, dict):
            labels.append(str(raw.get("label") or ""))
            summaries.append(f"[{fmt_ts(c0)}-{fmt_ts(c1)}] {raw.get('summary', '')}")
    if len(summaries) > 1:
        try:
            r = llm.chat_json([{"role": "system", "content": SYSTEM_VISION},
                               {"role": "user", "content": MACRO_SUMMARY_PROMPT.replace("<<TITLE>>", ep["title"]).replace("<<CHUNKS>>", "\n".join(summaries))}])
            label, summary = str(r.get("label") or labels[0]), str(r.get("summary") or " ".join(summaries))
        except Exception:
            label, summary = labels[0] or ep["title"], " ".join(summaries)[:1500]
    elif summaries:
        label, summary = labels[0] or ep["title"], summaries[0].split("] ", 1)[-1]
    else:
        label, summary = ep["title"], ep["title"]
    deg = Counter()
    for e in g["edges"]:
        deg[e["source"]] += 1
        deg[e["target"]] += 1
    named = sorted([x for x in g["entities"] if x["name"]], key=lambda x: -deg[x["id"]])
    macro = {"id": mid, "label": label, "summary": summary, "time_range": [t0, t1], "key_entities": [x["name"] for x in named[:5]],
             "event_types": sorted({e["event_type"] for e in g["events"] if e["event_type"]})[:8],
             "ocr_texts": [t["text"] for t in g["texts"]][:8], "segment_title": ep["title"]}
    return macro, g


# ------------------------------------------------------------------ 3. upward hierarchical aggregation
def aggregate(macros, llm):
    brief = [{"macro_id": m["id"], "label": m["label"], "time_range": [fmt_ts(m["time_range"][0]), fmt_ts(m["time_range"][1])],
              "summary": m["summary"][:350], "key_entities": m["key_entities"], "event_types": m["event_types"], "ocr_texts": m["ocr_texts"][:5]} for m in macros]
    msgs = [{"role": "user", "content": AGGREGATE_PROMPT + "\n# Input Macro events\n" + json.dumps(brief, ensure_ascii=False)}]
    try:
        raw = llm.chat_json(msgs, max_tokens=8000)
    except Exception as e:
        log.warning("aggregation failed (%s); using fixed-size grouping", e)
        raw = None
    return _validate_hierarchy(raw if isinstance(raw, dict) else {}, macros)


def _validate_hierarchy(raw, macros):
    ids = [m["id"] for m in macros]
    order = {i: k for k, i in enumerate(ids)}
    by_id = {m["id"]: m for m in macros}
    groups, seen = [], set()
    for s in raw.get("super_events") or []:
        if not isinstance(s, dict):
            continue
        mids = [x for x in (s.get("sub_macro_ids") or []) if x in order and x not in seen]
        seen.update(mids)
        if mids:
            groups.append({"mids": sorted(mids, key=order.get), "label": s.get("label"), "description": s.get("description"), "old": s.get("super_id")})
    if not groups:  # fallback: chunks of 5
        for k in range(0, len(ids), 5):
            groups.append({"mids": ids[k:k + 5], "label": None, "description": None, "old": None})
    else:
        for mid in [i for i in ids if i not in seen]:
            prev = ids[order[mid] - 1] if order[mid] > 0 else None
            g = next((g for g in groups if prev in g["mids"]), None)
            if g:
                g["mids"] = sorted(g["mids"] + [mid], key=order.get)
            else:
                groups.append({"mids": [mid], "label": None, "description": None, "old": None})
    groups.sort(key=lambda g: order[g["mids"][0]])
    supers, oldmap, macro2super = [], {}, {}
    for k, g in enumerate(groups):
        sid = f"super_{k + 1:02d}"
        ms = [by_id[i] for i in g["mids"]]
        cnt = Counter(e for m in ms for e in m["key_entities"])
        supers.append({"id": sid, "label": str(g["label"] or ms[0]["label"]),
                       "description": str(g["description"] or " ".join(m["summary"] for m in ms))[:1500],
                       "time_range": [min(m["time_range"][0] for m in ms), max(m["time_range"][1] for m in ms)],
                       "macro_ids": g["mids"], "key_entities": [e for e, _ in cnt.most_common(8)]})
        if g["old"]:
            oldmap[str(g["old"])] = sid
        for i in g["mids"]:
            macro2super[i] = sid

    def rel(e, kind, allowed, ok):
        if not isinstance(e, dict):
            return None
        s, t, ty = e.get("source"), e.get("target"), str(e.get("type", "")).upper()
        if kind == "super":
            s, t = oldmap.get(str(s), s), oldmap.get(str(t), t)
        if ty in allowed and ok(s, t):
            return {"source": s, "target": t, "label": ty, "type": "CAUSAL" if kind == "macro" else "PROGRESSION", "description": e.get("reason", "")}

    sids = {s["id"] for s in supers}
    macro_rel = [r for r in (rel(e, "macro", {"CAUSES", "PREVENTS", "ENABLES"},
                                 lambda s, t: s in macro2super and t in macro2super and s != t and macro2super[s] == macro2super[t]) for e in raw.get("macro_relations") or []) if r]
    super_rel = [r for r in (rel(e, "super", {"LEADS_TO", "RESOLVES", "CONTRASTS_WITH"},
                                 lambda s, t: s in sids and t in sids and s != t) for e in raw.get("super_relations") or []) if r]
    for a, b in zip(supers, supers[1:]):  # BEFORE edges between consecutive supers are auto-generated
        super_rel.append({"source": a["id"], "target": b["id"], "label": "BEFORE", "type": "TEMPORAL", "description": ""})
    r = raw.get("root") if isinstance(raw.get("root"), dict) else {}
    root = {"title": str(r.get("title") or "Video"), "description": str(r.get("description") or " ".join(s["label"] for s in supers)),
            "themes": r.get("themes") or [], "key_entities": r.get("key_entities") or [], "emotional_tone": r.get("emotional_tone") or ""}
    return {"supers": supers, "macro_super": macro2super, "macro_relations": macro_rel, "super_relations": super_rel, "root": root}


# ------------------------------------------------------------------ driver
def build_memory(video_path, out_dir, cfg, llm, embedder, force=False):
    os.makedirs(os.path.join(out_dir, "macros"), exist_ok=True)
    mem_path, idx_path = os.path.join(out_dir, "memory.json"), os.path.join(out_dir, "index.npz")
    if os.path.exists(mem_path) and not force:
        return load_bundle(out_dir, embedder)
    video = Video(video_path)
    segs = _cached(os.path.join(out_dir, "segments.json"), lambda: segment_video(video, llm, cfg))
    log.info("%d macro events", len(segs))

    def work(args):
        i, ep = args
        mid = f"macro_{i:04d}"
        return _cached(os.path.join(out_dir, "macros", f"{mid}.json"), lambda: dict(zip(("macro", "graph"), extract_subgraph(video, ep, mid, llm, cfg))))

    with ThreadPoolExecutor(cfg["perception"]["workers"]) as ex:
        res = list(tqdm(ex.map(work, enumerate(segs)), total=len(segs), desc="subgraphs"))
    macros = [r["macro"] for r in res]
    hier = _cached(os.path.join(out_dir, "hierarchy.json"), lambda: aggregate(macros, llm))
    for m in macros:
        m["super_id"] = hier["macro_super"][m["id"]]
    root = dict(hier["root"], time_range=[0, video.duration])
    data = {"video": {"path": video_path, "duration": video.duration}, "root": root, "supers": hier["supers"], "macros": macros,
            "subgraphs": {r["macro"]["id"]: r["graph"] for r in res},
            "macro_relations": hier["macro_relations"], "super_relations": hier["super_relations"]}
    mem = Memory(data)
    mem.save(mem_path)
    index = NodeIndex.build(mem, embedder)
    index.save(idx_path)
    with open(os.path.join(out_dir, "build_usage.json"), "w") as f:
        json.dump(llm.usage, f)
    return mem, index


def load_bundle(out_dir, embedder):
    mem = Memory.load(os.path.join(out_dir, "memory.json"))
    p = os.path.join(out_dir, "index.npz")
    if os.path.exists(p):
        return mem, NodeIndex.load(p)
    idx = NodeIndex.build(mem, embedder)
    idx.save(p)
    return mem, idx
