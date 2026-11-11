import copy, os
import yaml


def _merge(a, b):
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path="config.yaml", overrides=None):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg = _merge(cfg, overrides or {})
    env = {"MEMDREAMER_BASE_URL": "base_url", "MEMDREAMER_MODEL": "model", "MEMDREAMER_API_KEY": "api_key"}
    for e, k in env.items():
        if os.environ.get(e):
            cfg["llm"][k] = os.environ[e]
    return cfg


def llm_cfg(cfg, role):
    """role in {'perception','reasoning'}: base [llm] merged with optional [<role>_llm]."""
    return _merge(cfg["llm"], cfg.get(f"{role}_llm"))
