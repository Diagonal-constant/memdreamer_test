"""Hierarchical Graph Memory construction (paper Sec 3.1): streaming adaptive segmentation ->
downward subgraph extraction -> upward hierarchical aggregation. Every stage is cached on disk."""
import json, logging, math, os, re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from .embedding import NodeIndex
from .memory import Memory, REL_TYPE
from .prompts import SEGMENT_PROMPT, SUBGRAPH_PROMPT, AGGREGATE_PROMPT, MACRO_SUMMARY_PROMPT, SYSTEM_VISION, PEOPLE_PROMPT
from .video import Video, fmt_ts, parse_ts, vision_messages

log = logging.getLogger("memdreamer.build")


def _cached(path, fn):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    out = fn()
    with open(path + ".tmp", "w") as f:  # atomic: a crash never leaves a half-written cache that later loads as truth
        json.dump(out, f, ensure_ascii=False, indent=1)
    os.replace(path + ".tmp", path)
    return out


def _offset(times, c0):
    """Tiles carry ABSOLUTE time. If every time the model gave is before the chunk start, it answered
    chunk-relative: shift by c0. Otherwise 0 (values are only clamped)."""
    ts = [t for t in times if t is not None]
    return c0 if c0 > 1 and ts and max(ts) < c0 - 1 else 0.0  # ponytail: a mixed relative/absolute answer is clamped, not repaired


# ------------------------------------------------------------------ 1. streaming adaptive segmentation
def _clean_episodes(raw, t0, t1):
    cast = []
    if isinstance(raw, dict):
        cast = [c for c in raw.get("cast") or [] if isinstance(c, dict)]
        raw = raw.get("episodes") or next((v for v in raw.values() if isinstance(v, list) and v is not raw.get("cast")), [])
    eps = []
    for r in raw if isinstance(raw, list) else []:
        if not isinstance(r, dict):
            continue
        s, e = parse_ts(r.get("start_time")), parse_ts(r.get("end_time"))
        if s is None or e is None:
            continue
        eps.append({"start": s, "end": e, "title": str(r.get("episode_title", "Segment"))})
    off = _offset([x for ep in eps for x in (ep["start"], ep["end"])], t0)
    for ep in eps:
        ep["start"], ep["end"] = (min(max(ep[k] + off, t0), t1) for k in ("start", "end"))
    eps.sort(key=lambda x: x["start"])
    out = []
    for ep in eps:  # force contiguous, non-overlapping
        start = out[-1]["end"] if out else t0
        if ep["end"] - start >= 2:
            out.append({"start": start, "end": ep["end"], "title": ep["title"], "cast": cast})
    return out


def segment_video(video, llm, cfg):
    p = cfg["perception"]
    tau, t = float(p["window_sec"]), 0.0
    episodes = []
    while t < video.duration - 1:
        t1 = min(t + tau, video.duration)
        final = t1 >= video.duration - 1
        mos = video.mosaics(t, t1, tuple(p.get("segment_grid", p["grid"])), p["max_images"], p["tile_width"], quality=p["jpeg_quality"])
        prompt = SEGMENT_PROMPT.replace("<<DUR>>", fmt_ts(t1 - t)).replace("<<START>>", fmt_ts(t)).replace("<<END>>", fmt_ts(t1))
        header = f"Video window {fmt_ts(t)}-{fmt_ts(t1)} of the full video. Tile timestamps are absolute video time."
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
def _norm_subgraph(raw, mid, c0, c1, counters, name2ent, known=None):
    """Normalise one chunk's JSON into absolute-time nodes with macro-wide ids (E#, M#, T#).
    An entity whose id is in `known` (short id -> node from an earlier chunk) or whose name matches an earlier
    entity extends that node's time_range instead of becoming a duplicate; a known event id continues that event."""
    raw = raw if isinstance(raw, dict) else {}
    known = known or {}
    idmap = {}
    off = _offset([parse_ts(v) for k in ("micro_events", "on_screen_text", "entities") for e in raw.get(k) or []
                   if isinstance(e, dict) and isinstance(e.get("time_range"), (list, tuple)) for v in e["time_range"][:2]], c0)
    clamp = lambda v: min(max(v + off, c0), c1)

    def new(prefix):
        counters[prefix] += 1
        return f"{mid}:{prefix}{counters[prefix]}"

    def trange(x, default=None):
        if isinstance(x, (list, tuple)) and len(x) >= 2 and parse_ts(x[0]) is not None:
            a = clamp(parse_ts(x[0]))
            return [a, max(a, clamp(parse_ts(x[1], parse_ts(x[0]))))]
        return default or [c0, c0]

    def extend(node, tr):
        node["time_range"] = [min(node["time_range"][0], tr[0]), max(node["time_range"][1], tr[1])]
        if "time_sec" in node:
            node["time_sec"] = node["time_range"][0]

    ents, evs, txs, edges = [], [], [], []
    for i, e in enumerate(raw.get("entities") or []):
        if not isinstance(e, dict):
            continue
        key = str(e.get("entity_id") or f"E{i + 1}").strip()
        name, etype = e.get("name"), str(e.get("entity_type") or "OBJECT").upper()
        nk = (str(name).strip().lower(), etype) if name else None
        g = e.get("visual_grounding")
        gt = parse_ts(g.get("primary_time_sec")) if isinstance(g, dict) else None
        pt = clamp(gt) if gt is not None else c0
        tr = trange(e.get("time_range"), [pt, pt])
        old = known.get(key)
        # ponytail: trusts the model to reuse a listed id only for the same thing; the type check catches only cross-type collisions
        old = old if old is not None and str(old.get("entity_type", "")).upper() == etype else None
        old = old or (name2ent.get(nk) if nk else None)
        if old is not None:
            idmap.setdefault(key, old["id"])
            extend(old, tr)
            continue
        ent = {"id": new("E"), "name": name, "entity_type": etype, "attributes": e.get("attributes", {}),
               "description": e.get("description", ""), "time_sec": tr[0], "time_range": tr}
        idmap.setdefault(key, ent["id"])
        if nk:
            name2ent[nk] = ent
        ents.append(ent)
    full2ent = {x["id"]: x for x in ents + [v for v in known.values() if "entity_type" in v]}
    readable = {k: (full2ent[v].get("name") or str(full2ent[v].get("description", ""))[:50]) for k, v in idmap.items() if v in full2ent}
    readable.update({k: (v.get("name") or str(v.get("description", ""))[:50]) for k, v in known.items() if "entity_type" in v and k not in readable})
    unid = lambda x: re.sub(r"\s*\(seen [^)]*\)", "", re.sub(r"\b(E\d+)\b", lambda m: readable.get(m[1], m[1]), str(x or "")))  # models echo registry ids; text must stay searchable
    for i, e in enumerate(raw.get("micro_events") or []):
        if not isinstance(e, dict):
            continue
        e = dict(e, **{k: unid(e.get(k)) for k in ("subject", "object", "action", "description")})
        key, tr = str(e.get("event_id") or f"M{i + 1}").strip(), trange(e.get("time_range"))
        old = known.get(key)
        if old is not None and key.startswith("M") and tr[0] <= old["time_range"][1] + 2:  # a continuation must touch the old event's end
            idmap.setdefault(key, old["id"])
            extend(old, tr)
            continue
        nid = new("M")
        idmap.setdefault(key, nid)
        evs.append({"id": nid, "event_type": e.get("event_type", ""), "time_range": tr, "subject": e.get("subject", ""),
                    "object": e.get("object", ""), "action": e.get("action", ""), "description": e.get("description", "")})
    for i, e in enumerate(raw.get("on_screen_text") or []):
        if isinstance(e, dict):
            nid = new("T")
            idmap.setdefault(str(e.get("text_id") or f"T{i + 1}").strip(), nid)
            txs.append({"id": nid, "text": e.get("text", ""), "time_range": trange(e.get("time_range")), "description": e.get("description", "")})
    look = lambda k: idmap.get(k) or (known[k]["id"] if k in known else None)
    for e in raw.get("edges") or []:
        if not isinstance(e, dict):
            continue
        s_, t_, lab = look(str(e.get("source_id"))), look(str(e.get("target_id"))), str(e.get("relation_label", "")).upper()
        if s_ and t_ and s_ != t_ and lab in REL_TYPE:
            edges.append({"source": s_, "target": t_, "label": lab, "type": REL_TYPE[lab], "description": e.get("description", "")})
    return {"entities": ents, "events": evs, "texts": txs, "edges": edges}


def _context(g, counters):
    """Registry handed to the next chunk so ids and in-progress events carry across chunk borders."""
    if not g["entities"] and not g["events"]:
        return "This is the first part of the episode: no earlier entities or events."
    short = lambda i: i.split(":", 1)[1]
    # ponytail: lists the 40 most recently seen entities; long macros with more actors would need a search instead
    ents = sorted(g["entities"], key=lambda e: e["time_range"][1])[-40:]
    el = "\n".join(f"- {short(e['id'])} [{e['entity_type']}] {e['name'] or ''} {str(e['description'])[:120]} "
                   f"(seen {fmt_ts(e['time_range'][0])}-{fmt_ts(e['time_range'][1])})" for e in ents)
    vl = "\n".join(f"- {short(e['id'])} [{fmt_ts(e['time_range'][0])}-{fmt_ts(e['time_range'][1])}] {str(e['description'])[:160]}"
                   for e in sorted(g["events"], key=lambda e: e["time_range"][0])[-5:])
    return (f"Entities already identified earlier in this episode. When the same person/object appears, REUSE its entity_id (and update its time_range for this segment); never create a duplicate:\n{el or '(none)'}\n"
            f"Most recent micro-events. If one is still in progress at the start of this segment, reuse its event_id with its time_range in this segment; do NOT re-emit a finished event:\n{vl or '(none)'}\n"
            f"New entities must use ids starting at E{counters['E'] + 1}; new events at M{counters['M'] + 1}.")


def extract_subgraph(video, ep, mid, llm, cfg):
    """Macro Event -> chunks of `chunk_sec` (frames at max_fps in grid mosaics, <=5 images per call) -> merged subgraph.
    Chunks run in order and each sees the entity/event registry so far (stands in for the paper's single-pass episode)."""
    p = cfg["perception"]
    t0, t1 = ep["start"], ep["end"]
    n = max(1, math.ceil((t1 - t0) / p["chunk_sec"]))
    bounds = [t0 + (t1 - t0) * i / n for i in range(n + 1)]
    counters, name2ent = Counter(), {}
    cast = "\n".join(f'- {c.get("pid")} {c.get("description", "")} (seen {c.get("first_seen", "?")}-{c.get("last_seen", "?")})'
                     for c in ep.get("cast") or []) or "(no cast list)"
    g = {"entities": [], "events": [], "texts": [], "edges": []}
    labels, summaries = [], []
    for i in range(n):
        c0, c1 = bounds[i], bounds[i + 1]
        mos = video.mosaics(c0, c1, tuple(p["grid"]), p["max_images"], p["tile_width"], max_fps=p.get("max_fps", 1.0), quality=p["jpeg_quality"])
        header = f"Video segment {fmt_ts(c0)}-{fmt_ts(c1)} (chunk {i + 1}/{n} of an episode). Tile timestamps are absolute video time."
        known = {e["id"].split(":", 1)[1]: e for e in g["entities"] + sorted(g["events"], key=lambda e: e["time_range"][0])[-5:]}
        try:
            raw = llm.chat_json(vision_messages(mos, header, SUBGRAPH_PROMPT.replace("<<CONTEXT>>", _context(g, counters)).replace("<<CAST>>", cast)))
        except Exception as e:
            log.warning("subgraph extraction failed for %s chunk %d: %s", mid, i, e)
            continue
        part = _norm_subgraph(raw, mid, c0, c1, counters, name2ent, known)
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


def link_people(graphs, llm):
    """Macros are built in parallel, so the same person gets a different id in every macro.
    One text-only call groups PERSON ids across macros -> {entity_id: canonical label}."""
    ents = [e for g in graphs.values() for e in g["entities"] if e["entity_type"] == "PERSON"]
    if len(ents) < 2:
        return {}
    lines = "\n".join(f'{e["id"]} [{fmt_ts(e["time_range"][0])}-{fmt_ts(e["time_range"][1])}] {e.get("name") or ""} '
                      f'{json.dumps(e.get("attributes") or {}, ensure_ascii=False)[:150]} {str(e.get("description", ""))[:160]}' for e in ents)
    try:
        r = llm.chat_json([{"role": "system", "content": SYSTEM_VISION}, {"role": "user", "content": PEOPLE_PROMPT.replace("<<ENTS>>", lines)}], max_tokens=4000)
    except Exception as e:
        log.warning("people linking failed: %s", e)
        return {}
    out = {}
    for k, grp in enumerate(r.get("people") or [] if isinstance(r, dict) else []):
        if isinstance(grp, dict):
            for i in grp.get("ids") or []:
                out.setdefault(str(i), str(grp.get("label") or f"person {k + 1}"))
    return out


def _apply_people(res, people):
    """Unnamed person entities take their canonical label as name (so search, key_entities and the
    aggregator all see one identity); root gets the list of distinct people with their time spans."""
    spans = {}
    for r in res:
        g, m = r["graph"], r["macro"]
        for e in g["entities"]:
            lab = people.get(e["id"])
            if lab:
                e["person"] = lab
                e["name"] = e.get("name") or lab
                s = spans.setdefault(lab, [e["time_range"][0], e["time_range"][1], set()])
                s[0], s[1] = min(s[0], e["time_range"][0]), max(s[1], e["time_range"][1])
                s[2].add(m["id"])
        deg = Counter(x for e in g["edges"] for x in (e["source"], e["target"]))
        m["key_entities"] = list(dict.fromkeys(x["name"] for x in sorted([x for x in g["entities"] if x["name"]], key=lambda x: -deg[x["id"]])))[:5]
    return [{"label": k, "time_range": v[:2], "macros": sorted(v[2])} for k, v in sorted(spans.items(), key=lambda kv: kv[1][0])]


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
    if force:  # rebuild every stage, not only the final memory
        for f in ["segments.json", "people.json", "hierarchy.json", "memory.json", "index.npz"] + [os.path.join("macros", x) for x in os.listdir(os.path.join(out_dir, "macros"))]:
            if f.endswith((".json", ".npz", ".tmp")) and os.path.exists(os.path.join(out_dir, f)):
                os.remove(os.path.join(out_dir, f))
    video = Video(video_path)
    segs = _cached(os.path.join(out_dir, "segments.json"), lambda: segment_video(video, llm, cfg))
    mx = cfg["perception"].get("max_macro_sec")
    if mx:  # long episodes run their chunks serially (registry); capping their length lets macros run in parallel
        segs = [dict(ep, start=ep["start"] + (ep["end"] - ep["start"]) * k / n, end=ep["start"] + (ep["end"] - ep["start"]) * (k + 1) / n)
                for ep in segs for n in [max(1, math.ceil((ep["end"] - ep["start"]) / mx - 1e-6))] for k in range(n)]
    log.info("%d macro events", len(segs))

    def work(args):
        i, ep = args
        mid = f"macro_{i:04d}"
        return _cached(os.path.join(out_dir, "macros", f"{mid}.json"), lambda: dict(zip(("macro", "graph"), extract_subgraph(video, ep, mid, llm, cfg))))

    with ThreadPoolExecutor(cfg["perception"]["workers"]) as ex:
        res = list(tqdm(ex.map(work, enumerate(segs)), total=len(segs), desc="subgraphs"))
    casts = {json.dumps(ep.get("cast") or [], sort_keys=True) for ep in segs}
    if len(casts) == 1 and segs[0].get("cast"):  # one segmentation window saw the whole video: its cast is the identity list
        # ponytail: pids are per window; videos longer than window_sec fall back to the text linker below
        people = [{"label": f'{c.get("pid")} {c.get("description", "")}'.strip(), "macros": [],
                   "time_range": [parse_ts(c.get("first_seen"), 0.0), parse_ts(c.get("last_seen"), video.duration)]} for c in segs[0]["cast"]]
    else:
        people = _apply_people(res, _cached(os.path.join(out_dir, "people.json"), lambda: link_people({r["macro"]["id"]: r["graph"] for r in res}, llm)))
    macros = [r["macro"] for r in res]
    hier = _cached(os.path.join(out_dir, "hierarchy.json"), lambda: aggregate(macros, llm))
    for m in macros:
        m["super_id"] = hier["macro_super"][m["id"]]
    root = dict(hier["root"], time_range=[0, video.duration], people=people)
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
