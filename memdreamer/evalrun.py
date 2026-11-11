import hashlib, json, logging, os, threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from .agent import Agent, _letter
from .builder import build_memory, load_bundle
from .prompts import BASELINE_PROMPT
from .tools import MemoryTools
from .video import Video, vision_messages

log = logging.getLogger("memdreamer.eval")


def make_agent(mem, idx, embedder, llm_r, cfg):
    return Agent(mem, MemoryTools(mem, embedder, idx, cfg), llm_r, cfg)


def _find_video(name, video_dir):
    if os.path.exists(name):
        return name
    for ext in (".mp4", ".mkv", ".webm", ".avi", ".mov"):
        p = os.path.join(video_dir or ".", name + ext)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"video not found: {name}")


def run_eval(cfg, data_path, video_dir, mem_dir, out_path, llm_p, llm_r, embedder, mode="memdreamer", limit=None, workers=4):
    rows = [json.loads(l) for l in open(data_path) if l.strip()]
    if limit:
        rows = rows[:limit]
    key = lambda r: hashlib.md5((str(r["video"]) + r["question"]).encode()).hexdigest()
    done = {}
    if os.path.exists(out_path):
        done = {json.loads(l)["key"]: json.loads(l) for l in open(out_path) if l.strip()}
    by_video = defaultdict(list)
    for r in rows:
        if key(r) not in done:
            by_video[r["video"]].append(r)
    lock, fout = threading.Lock(), open(out_path, "a")

    def record(r, res):
        rec = {"key": key(r), "video": r["video"], "question": r["question"], "gt": str(r["answer"]).strip().upper()[:1],
               "pred": res["answer"], "type": r.get("type") or r.get("question_type"), "rounds": res.get("rounds"),
               "prompt_tokens": res.get("prompt_tokens")}
        rec["correct"] = rec["pred"] == rec["gt"]
        with lock:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            done[rec["key"]] = rec

    for vid, qs in by_video.items():
        vpath = _find_video(vid, video_dir)
        if mode == "memdreamer":
            mem, idx = build_memory(vpath, os.path.join(mem_dir, os.path.basename(str(vid)).rsplit(".", 1)[0]), cfg, llm_p, embedder)

            def one(r):
                try:
                    record(r, make_agent(mem, idx, embedder, llm_r, cfg).run(r["question"]))
                except Exception as e:
                    log.error("question failed: %s", e)
        else:
            video = Video(vpath)
            pc = cfg["perception"]   # service allows only 5 images -> the baseline sees <=45 uniformly sampled frames
            mos = video.mosaics(0, video.duration, tuple(pc["grid"]), pc["max_images"], pc["tile_width"], quality=pc["jpeg_quality"])

            def one(r):
                try:
                    msgs = vision_messages(mos, f"Full video, {video.duration:.0f} seconds, frames sampled uniformly. Timestamps are from the video start.", BASELINE_PROMPT.replace("<<Q>>", r["question"]))
                    out = llm_r.chat_json(msgs)
                    record(r, {"answer": _letter(out.get("answer")) or "A", "prompt_tokens": [llm_r.last_prompt_tokens()], "rounds": 1})
                except Exception as e:
                    log.error("baseline question failed: %s", e)

        with ThreadPoolExecutor(workers) as ex:
            list(tqdm(ex.map(one, qs), total=len(qs), desc=str(vid)))
    fout.close()
    return summarize(list(done.values()))


def summarize(recs):
    if not recs:
        return {}
    tot = sum(r["correct"] for r in recs) / len(recs)
    per = defaultdict(list)
    for r in recs:
        for t in (r["type"] if isinstance(r["type"], list) else [r["type"]]):
            if t:
                per[t].append(r["correct"])
    toks = [sum(r["prompt_tokens"]) / max(len(r["prompt_tokens"]), 1) for r in recs if r.get("prompt_tokens")]
    return {"n": len(recs), "accuracy": round(100 * tot, 2), "by_type": {k: round(100 * sum(v) / len(v), 2) for k, v in per.items()},
            "avg_rounds": round(sum(r.get("rounds") or 0 for r in recs) / len(recs), 2),
            "avg_prompt_tokens_per_call": round(sum(toks) / len(toks)) if toks else None}
