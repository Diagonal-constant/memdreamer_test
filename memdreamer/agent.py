"""Agentic Observation-Reason-Action loop (paper Sec 3.2, Eqs. 1-3, Fig. 10)."""
import json, re
from .prompts import AGENT_PROMPT

NEED2TOOL = {"get_summary": "get_summary", "get_super_events": "get_super_events", "browse_macros": "browse_macros",
             "get_macro_details": "get_macro_details", "search_nodes": "search_nodes", "search_time": "search_by_time",
             "get_relations": "get_relations"}


def _letter(x):
    m = re.match(r"\s*\(?([A-Ea-e])\b", str(x or ""))
    return m.group(1).upper() if m else None


class Agent:
    def __init__(self, memory, tools, llm, cfg):
        self.m, self.tools, self.llm, self.cfg = memory, tools, llm, cfg["reasoning"]

    def _initial(self):
        return self.tools.get_summary() + "\n" + self.tools.get_super_events()

    def run(self, question, verbose=False):
        T = self.cfg["max_rounds"]
        system = AGENT_PROMPT.replace("<<TMAX>>", str(T))
        useful, history, seen, inspected = "", [], set(), set()
        obs, trace, ptoks = "(none yet: only the initial context is available)", [], []
        ans = None
        for r in range(1, T + 1):
            unexplored = sorted(seen - inspected)
            user = (f"# Question\n{question}\n\n# Initial context\n{self._initial()}\n\n"
                    f"# Exploration history\n{chr(10).join(history) or '(empty)'}\n"
                    f"Macros seen in observations but not yet inspected with get_macro_details: {', '.join(unexplored) or '(none)'}\n\n"
                    f"# Useful info carried from previous rounds\n{useful or '(empty)'}\n\n"
                    f"# Latest observation (round {r}/{T})\n{obs}\n"
                    + ("\nThis is the FINAL round: you MUST answer now with your best guess (can_answer=true)." if r == T else ""))
            out = self.llm.chat_json([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                     temperature=self.cfg.get("temperature"), max_tokens=self.cfg.get("max_tokens"))
            out = out if isinstance(out, dict) else {}
            ptoks.append(self.llm.last_prompt_tokens())
            useful = str(out.get("useful_info") or useful)
            ans = _letter(out.get("answer"))
            step = {"round": r, "reasoning": out.get("reasoning"), "can_answer": out.get("can_answer"), "answer": ans, "need": out.get("information_need")}
            trace.append(step)
            if verbose:
                print(f"--- round {r}: can_answer={out.get('can_answer')} need={out.get('information_need')}")
            if ans and (out.get("can_answer") or r == T):
                break
            need = out.get("information_need")
            name = NEED2TOOL.get(str(need.get("type"))) if isinstance(need, dict) else None
            if name is None:
                obs = "[error] you neither answered nor gave a valid information_need.type; call one tool or answer."
                history.append(f"R{r}: (invalid request)")
                continue
            args = need.get("args") or {}
            obs = self.tools.call(name, args)
            step["observation_preview"] = obs[:300]
            history.append(f"R{r}: {name}({json.dumps(args, ensure_ascii=False)})")
            seen |= set(re.findall(r"macro_\d{4}", obs))
            if name == "get_macro_details":
                inspected.add(str(args.get("macro_id")))
        return {"answer": ans or "A", "answered": bool(ans), "rounds": len(trace), "prompt_tokens": ptoks, "trace": trace}
