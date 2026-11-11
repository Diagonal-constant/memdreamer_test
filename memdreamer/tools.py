import re
from .video import fmt_ts, parse_ts
from .memory import _trim

_TYPE_ALIAS = {"entity": "Entity", "event": "Event", "text": "Text", "ocr": "Text", "ontext": "Text", "onscreentext": "Text"}


class MemoryTools:
    """The seven tools of Table 2. Every tool returns a plain-text observation."""

    def __init__(self, memory, embedder, index, cfg):
        self.m, self.emb, self.idx = memory, embedder, index
        self.top_k = cfg["reasoning"]["top_k"]
        self.max_chars = cfg["reasoning"]["max_obs_chars"]

    def call(self, name, args):
        fn = getattr(self, name, None)
        if fn is None or name.startswith("_") or name == "call":
            return f"[tool error] unknown tool {name}"
        try:
            out = fn(**(args or {}))
        except Exception as e:
            return f"[tool error] {type(e).__name__}: {e}"
        if len(out) > self.max_chars:
            out = out[: self.max_chars] + "\n…[truncated; use search_nodes / get_relations for specifics]"
        return out

    # ---------- hierarchical navigation ----------
    def get_summary(self, **_):
        r = self.m.d["root"]
        return (f'## Video Summary\n{r.get("title","")}\n{r.get("description","")}\n'
                f'Themes: {", ".join(r.get("themes", []))}\nKey entities: {", ".join(map(str, r.get("key_entities", [])))}')

    def get_super_events(self, **_):
        lines = [f"## Found {len(self.m.supers)} super events"]
        for sid, s in self.m.supers.items():
            a, b = s["time_range"]
            lines.append(f'[{sid}] {s["label"]} [{fmt_ts(a)}-{fmt_ts(b)}] (macros: {len(s["macro_ids"])}) {_trim(s.get("description",""), 260)}')
        return "\n".join(lines)

    def browse_macros(self, super_id=None, macro_ids=None, **_):
        ids = macro_ids or (self.m.supers[super_id]["macro_ids"] if super_id in self.m.supers else None)
        if ids is None:
            return f"[tool error] unknown super_id {super_id}"
        lines = []
        for mid in ids:
            m = self.m.macros.get(mid)
            if m:
                a, b = m["time_range"]
                lines.append(f'[{mid}] {m["label"]} [{fmt_ts(a)}-{fmt_ts(b)}] key_entities: {", ".join(m["key_entities"])}\n   {_trim(m.get("summary",""), 220)}')
        return "\n".join(lines) or "(no macros)"

    get_macro_events = browse_macros

    def get_macro_details(self, macro_id=None, **_):
        if macro_id not in self.m.macros:
            return f"[tool error] unknown macro_id {macro_id}"
        mc, g = self.m.macros[macro_id], self.m.sub[macro_id]
        a, b = mc["time_range"]
        L = [f'=== Subgraph: {mc["label"]} ({macro_id}, in {mc["super_id"]}) ===', f"Time: {fmt_ts(a)}-{fmt_ts(b)}",
             f'Summary: {mc.get("summary","")}', f'--- Events ({len(g["events"])}) ---']
        for e in sorted(g["events"], key=lambda x: x["time_range"][0]):
            L.append(f'{e["id"]} [{fmt_ts(e["time_range"][0])}-{fmt_ts(e["time_range"][1])}] {_trim(e.get("subject",""),50)} | {_trim(e.get("action",""),60)} | {_trim(e.get("object",""),50)}: {_trim(e.get("description",""),240)}')
        if g["texts"]:
            L.append(f'--- On-screen text ({len(g["texts"])}) ---')
            for t in g["texts"]:
                L.append(f'{t["id"]} [{fmt_ts(t["time_range"][0])}] "{_trim(t.get("text",""),120)}": {_trim(t.get("description",""),140)}')
        L.append(f'--- Key relations ({len(g["edges"])}) ---')
        for e in g["edges"]:
            L.append(f'{self._short(e["source"])} -[{e["label"]}]-> {self._short(e["target"])}' + (f': {_trim(e["description"],100)}' if e.get("description") else ""))
        L.append(f'--- Entities ({len(g["entities"])}) ---')
        for e in g["entities"]:
            L.append(f'{e["id"]} {e.get("name") or "unnamed"} ({e.get("entity_type","")}): {_trim(e.get("description",""),200)}')
        return "\n".join(L)

    get_subgraph = get_macro_details

    # ---------- precise search ----------
    def search_nodes(self, query="", top_k=None, node_types=None, **_):
        types = None
        if node_types:
            if isinstance(node_types, str):
                node_types = [node_types]
            types = {_TYPE_ALIAS.get(re.sub(r"[^a-z]", "", t.lower()), t) for t in node_types}
        k = int(top_k or self.top_k)
        qv = self.emb.encode([query], is_query=True)[0]
        hits = self.idx.search(qv, k, types)
        if not hits:
            return "Found 0 matching nodes."
        L = [f"Found {len(hits)} matching nodes:"]
        for rank, (nid, sc) in enumerate(hits):
            nd = self.m.nodes[nid]
            mc = self.m.macros.get(nd["macro_id"], {})
            L.append(f'In <Macro Event> {nd["macro_id"]} "{mc.get("label","")}" ({mc.get("super_id","")}) <Match Node {rank}> score={sc:.2f}\n  {self.m.label(nid, 240)}')
            nbrs = [e for e in (self.m.out[nid] + self.m.inc[nid]) if e["label"] != "SUBEVENT_OF" or nd["type"] not in ("Event",)][:4]
            for e in nbrs:
                other, arrow = (e["target"], "->") if e["source"] == nid else (e["source"], "<-")
                L.append(f'    {arrow}[{e["label"]}] {self.m.label(other, 120)}')
        return "\n".join(L)

    def search_by_time(self, start_sec=0, end_sec=0, **_):
        s, e = parse_ts(start_sec, 0), parse_ts(end_sec, 0)
        if e < s:
            s, e = e, s
        L = []
        for mid, m in self.m.macros.items():
            a, b = m["time_range"]
            if a <= e and b >= s:
                L.append(f'[{mid}] {m["label"]} [{fmt_ts(a)}-{fmt_ts(b)}] (in {m["super_id"]}: {self.m.supers[m["super_id"]]["label"]})')
        return f"Macro events overlapping {fmt_ts(s)}-{fmt_ts(e)}:\n" + ("\n".join(L) or "(none)")

    # ---------- graph traversal ----------
    def get_relations(self, node_id=None, direction="both", **_):
        if node_id not in self.m.nodes:
            return f"[tool error] unknown node_id {node_id}. Ids look like macro_0020:M5."
        L = [f"Relations of {self.m.label(node_id, 200)}"]
        if direction in ("out", "outgoing", "both"):
            L.append("Outgoing:")
            L += [f'  -[{e["label"]}]-> {self.m.label(e["target"], 150)}' + (f' ({_trim(e["description"],100)})' if e.get("description") else "") for e in self.m.out[node_id][:30]] or ["  (none)"]
        if direction in ("in", "incoming", "both"):
            L.append("Incoming:")
            L += [f'  <-[{e["label"]}]- {self.m.label(e["source"], 150)}' + (f' ({_trim(e["description"],100)})' if e.get("description") else "") for e in self.m.inc[node_id][:30]] or ["  (none)"]
        return "\n".join(L)

    def _short(self, nid):
        nd = self.m.nodes.get(nid)
        return f'{nid}"{_trim(nd["text"], 40)}"' if nd else nid
