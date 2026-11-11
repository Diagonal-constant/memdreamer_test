import argparse, json, logging
from .config import load_config, llm_cfg
from .embedding import Embedder
from .llm import LLM


def main():
    ap = argparse.ArgumentParser(prog="memdreamer")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build hierarchical graph memory for one video")
    b.add_argument("--video", required=True); b.add_argument("--out", required=True); b.add_argument("--force", action="store_true")
    a = sub.add_parser("ask", help="answer a question (multiple-choice if it lists (A)...(E) options, otherwise free text) from a built memory")
    a.add_argument("--memory", required=True); a.add_argument("--question", required=True, help="question text INCLUDING options, e.g. '... (A) x (B) y'")
    a.add_argument("--trace", action="store_true")
    e = sub.add_parser("eval", help="evaluate on a QA jsonl (video, question, answer[, type])")
    e.add_argument("--data", required=True); e.add_argument("--video-dir", default="."); e.add_argument("--mem-dir", default="memories")
    e.add_argument("--out", default="results.jsonl"); e.add_argument("--mode", choices=["memdreamer", "baseline"], default="memdreamer")
    e.add_argument("--limit", type=int); e.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(asctime)s %(name)s %(message)s")
    cfg = load_config(args.config)
    llm_p, llm_r = LLM(llm_cfg(cfg, "perception")), LLM(llm_cfg(cfg, "reasoning"))
    emb = Embedder(cfg) if (args.cmd != "eval" or args.mode == "memdreamer") else None

    if args.cmd == "build":
        from .builder import build_memory
        mem, _ = build_memory(args.video, args.out, cfg, llm_p, emb, args.force)
        print(f"{len(mem.supers)} super events, {len(mem.macros)} macro events, {len(mem.nodes)} nodes -> {args.out}")
        print("perception usage:", llm_p.usage)
    elif args.cmd == "ask":
        from .builder import load_bundle
        from .evalrun import make_agent
        mem, idx = load_bundle(args.memory, emb)
        res = make_agent(mem, idx, emb, llm_r, cfg).run(args.question, verbose=args.verbose)
        print("ANSWER:", res["answer"], "| rounds:", res["rounds"], "| prompt tokens/round:", res["prompt_tokens"])
        if args.trace:
            print(json.dumps(res["trace"], indent=1, ensure_ascii=False))
    else:
        from .evalrun import run_eval
        s = run_eval(cfg, args.data, args.video_dir, args.mem_dir, args.out, llm_p, llm_r, emb, args.mode, args.limit, args.workers)
        print(json.dumps(s, indent=1))
