# MemDreamer re-implementation on the ACI-LLM API (Qwen, OpenAI-compatible)

Re-implementation of *MemDreamer* (arXiv 2606.07512) from the paper text. All perception and reasoning goes
through your ACI-LLM endpoint; only text embeddings run locally (the service has no `/v1/embeddings`).

## Adapted to the service limits
| Service limit (from the developer guide) | How the code handles it |
|---|---|
| **max 5 images / request** | Frames are packed into 3x3 **mosaic** images (<=1 MP) with timestamps burned in and also listed in text: up to 45 frames per call |
| **32,768-token context** (prompt + output) | Macro Events are analysed in `chunk_sec` (90 s) chunks, then merged; agent rounds are ~6-8K tokens; `max_tokens` kept <=12K |
| Default assistant persona injected without a system message | Every call sends its own system message |
| Retry only 429/500/503/504, backoff + jitter, honour `Retry-After`; never retry 400/401/404 | `llm.py` implements exactly this; timeout 600 s |
| ~8 parallel requests per app; 600 req/min prod, 120 staging | `perception.workers: 6`, eval `--workers 4` |
| Thinking via `reasoning_effort` (content may be null if the budget runs out) | Agent uses `medium`; perception uses none |
| Base64 `data:` images only | That is what is sent |

## Setup
```bash
pip install -r requirements.txt
python tests/smoke_test.py                       # offline plumbing test (fake LLM)
```
Edit `config.yaml`: `llm.base_url` (`http://<HOST>:18766/v1` = staging), `llm.api_key` (`qwk_test_...`), `llm.model: ACI-LLM`.
Check the connection and image support first:
```bash
curl $BASE/v1/models -H "Authorization: Bearer $KEY"
```
Then:
```bash
python -m memdreamer -v build --video my_video.mp4 --out memories/my_video
python -m memdreamer -v ask --memory memories/my_video --trace \
   --question "Why does the worker pour the fish into the container? (A) Drying (B) Cleaning (C) Cooking (D) Mincing"
python -m memdreamer eval --data qa.jsonl --video-dir videos --mem-dir memories --out results.jsonl --workers 4
python -m memdreamer eval --data qa.jsonl --video-dir videos --mode baseline --out baseline.jsonl
```
Switch to production only after staging works (change base_url and use the `qwk_live_` key).
Builds are resumable (`segments.json`, `macros/*.json`, `hierarchy.json`, `memory.json`, `index.npz`).

**Eval data** (JSONL): `{"video": "<id or path>", "question": "<text with (A)..(D)>", "answer": "A", "type": ["KIR"]}`.

## Paper -> code
| Paper | File |
|---|---|
| Streaming adaptive segmentation | `builder.segment_video` |
| Downward subgraph extraction (+ chunk merge) | `builder.extract_subgraph` |
| Upward aggregation, Root | `builder.aggregate` |
| Memory schema (Table 1) | `memory.py` |
| 7 tools (Table 2) | `tools.py` |
| ORA loop (Eqs 1-3) | `agent.py` |
| Eval, baseline, token stats | `evalrun.py` |

## Caveats
* The **baseline** mode only sees <=45 uniformly sampled frames (service limit), so it is not the paper's full-video baseline.
* Frame spacing is coarse: ~13 s in segmentation windows, ~2 s inside chunks. Lower `chunk_sec` for finer detail (more calls).
* Same named entity in different chunks of one macro event is merged by name; unnamed entities may duplicate.
* Cost: about duration/90 s multimodal calls (~50 s each at ~56 tok/s) plus segmentation and aggregation; 6 workers keep this reasonable.
* Prompts are reconstructed from the appendix (Fig. 8 is truncated in the PDF); expect numbers below the paper's.
