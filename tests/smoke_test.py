"""End-to-end smoke test with a scripted fake LLM + fake embedder + synthetic video.
Checks plumbing only (no model quality). Run: python tests/smoke_test.py"""
import os, re, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
try:
    import openai  # noqa
except ImportError:  # allow running without the SDK installed
    import types; m = types.ModuleType("openai"); m.OpenAI = object
    m.APIStatusError = type("APIStatusError", (Exception,), {}); m.APIConnectionError = type("APIConnectionError", (Exception,), {})
    sys.modules["openai"] = m
try:
    import tqdm  # noqa
except ImportError:
    import types; m = types.ModuleType("tqdm"); m.tqdm = lambda it, **k: it; sys.modules["tqdm"] = m
import cv2, numpy as np

from memdreamer.config import load_config
from memdreamer.builder import build_memory
from memdreamer.evalrun import make_agent, run_eval, summarize


class FakeLLM:
    usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
    def last_prompt_tokens(self): return 123
    def chat_json(self, messages, **kw):
        text = "".join(c["text"] for c in messages[-1]["content"] if c["type"] == "text") if isinstance(messages[-1]["content"], list) else messages[-1]["content"]
        sysm = messages[0]["content"] if messages[0]["role"] == "system" else ""
        if "Expert Visual Story Segmenter" in text:
            d = re.search(r"window is (\d\d):(\d\d):(\d\d) long", text); dur = int(d[2]) * 60 + int(d[3])
            h = dur // 2
            f = lambda s: f"00:{s//60:02d}:{s%60:02d}"
            return [{"start_time": "00:00:00", "end_time": f(h), "episode_title": "red phase"}, {"start_time": f(h), "end_time": f(dur), "episode_title": "blue phase"}]
        if "knowledge-graph annotator" in text:
            return {"label": "Test scene", "summary": "A man pours fish into a vat to cook them.", "event_types": ["action"],
                    "micro_events": [{"event_id": "M1", "event_type": "action", "time_range": [0, 3], "subject": "worker", "object": "fish", "action": "pours", "description": "worker pours fish into vat"},
                                     {"event_id": "M2", "event_type": "action", "time_range": [3, 6], "subject": "water", "object": "fish", "action": "boils", "description": "fish boil in vat"}],
                    "entities": [{"entity_id": "E1", "name": "Worker", "entity_type": "PERSON", "description": "man in blue shirt", "visual_grounding": {"primary_time_sec": 1}}],
                    "on_screen_text": [{"text_id": "T1", "text": "SHIRASU", "time_range": [0, 5], "description": "caption"}],
                    "edges": [{"source_id": "E1", "target_id": "M1", "relation_label": "PERFORMS", "description": ""},
                              {"source_id": "M1", "target_id": "M2", "relation_label": "CAUSES", "description": "pouring leads to boiling"},
                              {"source_id": "X", "target_id": "M2", "relation_label": "CAUSES"}]}
        if "Merge these chronological chunk summaries" in text:
            return {"label": "Merged scene", "summary": "Merged: worker pours fish and boils them."}
        if "video narrative analyst" in text:
            ids = sorted(set(re.findall(r"macro_\d{4}", text)))
            return {"super_events": [{"super_id": "S1", "label": "Everything", "description": "all", "sub_macro_ids": ids[:1]}],  # second macro deliberately missing
                    "macro_relations": [], "super_relations": [], "root": {"title": "Test video", "description": "d", "themes": ["t"]}}
        if "reasoning model R" in sysm:
            obs = text.split("# Latest observation")[1]
            if "(none yet" in obs:
                return {"reasoning": "", "can_answer": False, "useful_info": "start", "information_need": {"type": "search_nodes", "args": {"query": "worker pours fish"}}}
            if "Found" in obs and "=== Subgraph" not in obs:
                mid = re.search(r"macro_\d{4}", obs)[0]
                return {"reasoning": "", "can_answer": False, "useful_info": f"look at {mid}", "information_need": {"type": "get_macro_details", "args": {"macro_id": mid}}}
            assert "Subgraph" in obs, obs
            return {"reasoning": "boils -> cooking", "can_answer": True, "answer": "C", "useful_info": "", "information_need": None}
        if "multiple choice" in text:
            return {"reasoning": "", "answer": "C"}
        raise AssertionError("unexpected prompt")


class FakeEmb:
    def encode(self, texts, is_query=False):
        v = np.zeros((len(texts), 64), np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z]+", t.lower()): v[i, hash(w) % 64] += 1
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)


def main():
    tmp = tempfile.mkdtemp()
    vp = os.path.join(tmp, "toy.mp4")
    w = cv2.VideoWriter(vp, cv2.VideoWriter_fourcc(*"mp4v"), 5, (160, 90))
    for i in range(5 * 90):  # 90 s
        w.write(np.full((90, 160, 3), (0, 0, 255) if i < 225 else (255, 0, 0), np.uint8))
    w.release()
    cfg = load_config(os.path.join(os.path.dirname(__file__), "..", "config.yaml"))
    cfg["perception"].update(window_sec=45, chunk_sec=12, workers=2)
    llm, emb = FakeLLM(), FakeEmb()
    mem, idx = build_memory(vp, os.path.join(tmp, "mem"), cfg, llm, emb)
    print("supers", len(mem.supers), "macros", len(mem.macros), "nodes", len(mem.nodes), "indexed", len(idx.ids))
    assert len(mem.macros) >= 2 and all(m["super_id"] in mem.supers for m in mem.macros.values())
    res = make_agent(mem, idx, emb, llm, cfg).run("Why does the worker pour the fish? (A) Drying (B) Cleaning (C) Cooking (D) Mincing")
    print("agent:", res["answer"], "rounds", res["rounds"]); assert res["answer"] == "C" and res["rounds"] == 3
    import json
    data = os.path.join(tmp, "qa.jsonl")
    open(data, "w").write(json.dumps({"video": vp, "question": "Why? (A) a (B) b (C) c (D) d", "answer": "C", "type": ["KIR"]}) + "\n")
    for mode in ("memdreamer", "baseline"):
        print(mode, run_eval(cfg, data, tmp, os.path.join(tmp, "mems"), os.path.join(tmp, f"r_{mode}.jsonl"), llm, llm, emb, mode, None, 1))
    print("SMOKE TEST PASSED")

if __name__ == "__main__":
    main()
