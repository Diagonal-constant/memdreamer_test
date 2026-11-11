import numpy as np


class Embedder:
    """Qwen3-Embedding via sentence-transformers (local) or any OpenAI-compatible /v1/embeddings server."""

    def __init__(self, cfg):
        self.cfg = cfg["embedding"]
        self.backend = self.cfg.get("backend", "sentence-transformers")
        self.instruction = self.cfg.get("query_instruction", "")
        if self.backend == "openai":
            from openai import OpenAI
            self.client = OpenAI(base_url=self.cfg["base_url"], api_key=self.cfg.get("api_key", "EMPTY"))
        else:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.cfg["model"])

    def encode(self, texts, is_query=False, batch=64):
        if is_query and self.instruction:  # Qwen3-Embedding query format
            texts = [f"Instruct: {self.instruction}\nQuery:{t}" for t in texts]
        outs = []
        for i in range(0, len(texts), batch):
            chunk = texts[i:i + batch]
            if self.backend == "openai":
                r = self.client.embeddings.create(model=self.cfg["model"], input=chunk)
                v = np.array([d.embedding for d in r.data], dtype=np.float32)
            else:
                v = self.model.encode(chunk, normalize_embeddings=False, show_progress_bar=False)
            outs.append(np.asarray(v, dtype=np.float32))
        v = np.concatenate(outs) if outs else np.zeros((0, 1), np.float32)
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)


class NodeIndex:
    def __init__(self, ids, types, vecs):
        self.ids, self.types, self.vecs = list(ids), list(types), vecs

    @classmethod
    def build(cls, memory, embedder):
        rows = memory.searchable()
        vecs = embedder.encode([r[2] for r in rows]) if rows else np.zeros((0, 1), np.float32)
        return cls([r[0] for r in rows], [r[1] for r in rows], vecs)

    def save(self, path):
        np.savez(path, ids=np.array(self.ids), types=np.array(self.types), vecs=self.vecs)

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        return cls(z["ids"].tolist(), z["types"].tolist(), z["vecs"])

    def search(self, qvec, k=10, node_types=None):
        if not len(self.ids):
            return []
        scores = self.vecs @ qvec
        if node_types:
            mask = np.array([t in node_types for t in self.types])
            scores = np.where(mask, scores, -1e9)
        order = np.argsort(-scores)[:k]
        return [(self.ids[i], float(scores[i])) for i in order if scores[i] > -1e8]
