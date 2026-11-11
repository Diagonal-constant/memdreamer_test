import json, logging, random, re, threading, time
from openai import OpenAI, APIStatusError, APIConnectionError

log = logging.getLogger("memdreamer.llm")
RETRY_STATUS = {429, 500, 503, 504}   # per the service guide; never retry 400/401/404/413


def strip_think(text):
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def extract_json(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    dec = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "[{":
            try:
                obj, _ = dec.raw_decode(text[i:])
                return obj
            except json.JSONDecodeError:
                continue
    raise ValueError("no JSON found in model output")


class LLM:
    """OpenAI-compatible chat client with the ACI-LLM retry policy (exponential backoff + jitter, honours Retry-After)."""

    def __init__(self, cfg):
        self.client = OpenAI(base_url=cfg["base_url"], api_key=cfg.get("api_key", "EMPTY"),
                             timeout=cfg.get("timeout", 600), max_retries=0)
        self.model = cfg["model"]
        self.temperature = cfg.get("temperature", 0.7)
        self.max_tokens = cfg.get("max_tokens", 6000)
        self.max_retries = cfg.get("max_retries", 5)
        self.json_mode = cfg.get("json_mode", False)
        self.effort = cfg.get("reasoning_effort")
        self._lock = threading.Lock()
        self._local = threading.local()
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def last_prompt_tokens(self):
        return getattr(self._local, "prompt_tokens", 0)

    def chat(self, messages, max_tokens=None, temperature=None):
        kw = dict(model=self.model, messages=messages,
                  temperature=self.temperature if temperature is None else temperature,
                  max_tokens=max_tokens or self.max_tokens)
        if self.json_mode:
            kw["response_format"] = {"type": "json_object"}
        if self.effort and self.effort != "none":
            kw["extra_body"] = {"reasoning_effort": self.effort}
        err = None
        for attempt in range(self.max_retries):
            delay = min(2 ** attempt, 30) * (0.5 + random.random() / 2)
            try:
                r = self.client.chat.completions.create(**kw)
                u = r.usage
                pt, ct = (u.prompt_tokens, u.completion_tokens) if u else (0, 0)
                self._local.prompt_tokens = pt
                with self._lock:
                    self.usage["calls"] += 1
                    self.usage["prompt_tokens"] += pt
                    self.usage["completion_tokens"] += ct
                ch = r.choices[0]
                if ch.finish_reason == "length":
                    log.warning("finish_reason=length (output truncated; thinking may have used the budget)")
                return strip_think(ch.message.content or "")   # content can be null when thinking runs out
            except APIStatusError as e:
                if e.status_code not in RETRY_STATUS:
                    raise RuntimeError(f"HTTP {e.status_code}: {e.message}") from e
                ra = e.response.headers.get("retry-after") if getattr(e, "response", None) is not None else None
                if ra and re.fullmatch(r"\d+(\.\d+)?", ra):
                    delay = float(ra)
                err = e
            except APIConnectionError as e:   # includes timeouts
                err = e
            log.warning("retryable error (%s); sleeping %.1fs", err, delay)
            time.sleep(delay)
        raise RuntimeError(f"LLM call failed after {self.max_retries} tries: {err}")

    def chat_json(self, messages, max_tokens=None, temperature=None, tries=3):
        msgs = list(messages)
        for _ in range(tries):
            text = self.chat(msgs, max_tokens, temperature)
            try:
                return extract_json(text)
            except ValueError:
                msgs = msgs + [{"role": "assistant", "content": text[:3000]},
                               {"role": "user", "content": "Your previous reply was not valid JSON. Reply again with ONLY the JSON, no commentary."}]
        raise ValueError("model did not return valid JSON")
