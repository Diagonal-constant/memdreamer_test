import json
from collections import defaultdict
from .video import fmt_ts

# relation label -> relation category (Table 1)
REL_TYPE = {
    "PERFORMS": "SEMANTIC", "RECEIVES": "SEMANTIC", "USES_TOOL": "SEMANTIC", "HAPPENS_IN": "SEMANTIC",
    "CONTEXT_FOR": "SEMANTIC", "ANNOTATES": "SPATIAL",
    "PART_OF": "SPATIAL", "LOCATED_IN": "SPATIAL", "NEXT_TO": "SPATIAL", "ATTACHED_TO": "SPATIAL",
    "IS_SAME_AS": "IDENTITY",
    "BEFORE": "TEMPORAL", "OVERLAP": "TEMPORAL", "CAUSES": "CAUSAL", "PREVENTS": "CAUSAL",
    "SUBEVENT_OF": "HIERARCHICAL",
}


def _trim(s, n):
    s = " ".join(str(s or "").split())
    return s if len(s) <= n else s[: n - 1] + "…"


class Memory:
    """Hierarchical Graph Memory G: Root -> Super -> Macro -> {entities, micro-events, text} + edges."""

    def __init__(self, data):
        self.d = data
        self.supers = {s["id"]: s for s in data["supers"]}
        self.macros = {m["id"]: m for m in data["macros"]}
        self.sub = data["subgraphs"]
        self.nodes = {}
        self.out, self.inc = defaultdict(list), defaultdict(list)
        self._build()

    def _edge(self, e):
        self.out[e["source"]].append(e)
        self.inc[e["target"]].append(e)

    def _build(self):
        r = self.d["root"]
        self.nodes["root"] = {"type": "Root", "macro_id": None, "text": f'{r.get("title","")}: {r.get("description","")}'}
        for sid, s in self.supers.items():
            self.nodes[sid] = {"type": "SuperEvent", "macro_id": None, "text": f'{s["label"]}: {s.get("description","")}', "time": s["time_range"]}
            self._edge({"source": sid, "target": "root", "label": "SUBEVENT_OF", "type": "HIERARCHICAL", "description": ""})
        for mid, m in self.macros.items():
            self.nodes[mid] = {"type": "MacroEvent", "macro_id": mid, "text": f'{m["label"]}: {m.get("summary","")}', "time": m["time_range"]}
            self._edge({"source": mid, "target": m["super_id"], "label": "SUBEVENT_OF", "type": "HIERARCHICAL", "description": ""})
        for e in self.d["macro_relations"] + self.d["super_relations"]:
            self._edge(e)
        for mid, g in self.sub.items():
            for e in g["entities"]:
                self.nodes[e["id"]] = {"type": "Entity", "macro_id": mid, "time": e.get("time_range") or [e.get("time_sec", 0)] * 2,
                                       "text": f'{e.get("name") or "unnamed"} ({e.get("entity_type","")}): {e.get("description","")}'}
            for e in g["events"]:
                self.nodes[e["id"]] = {"type": "Event", "macro_id": mid, "time": e["time_range"],
                                       "text": f'{e.get("subject","")} {e.get("action","")} {e.get("object","")}. {e.get("description","")}'}
            for e in g["texts"]:
                self.nodes[e["id"]] = {"type": "Text", "macro_id": mid, "time": e["time_range"],
                                       "text": f'On-screen text "{e.get("text","")}". {e.get("description","")}'}
            for e in g["edges"]:
                self._edge(e)

    def searchable(self):
        return [(i, n["type"], n["text"]) for i, n in self.nodes.items() if n["type"] in ("Entity", "Event", "Text")]

    def label(self, nid, n=160):
        nd = self.nodes.get(nid)
        if not nd:
            return nid
        t, tm = nd["type"], nd.get("time")
        span = f" [{fmt_ts(tm[0])}-{fmt_ts(tm[1])}]" if tm and tm[0] != tm[1] else (f" [{fmt_ts(tm[0])}]" if tm else "")
        return f"({t}) {nid}{span} {_trim(nd['text'], n)}"

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.d, f, ensure_ascii=False, indent=1)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(json.load(f))
