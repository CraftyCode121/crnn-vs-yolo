"""synth_pages generator package: shared helpers."""
from pathlib import Path

import yaml


def load_config(path="config.yaml"):
    """Load config.yaml. Relative paths inside it resolve against its folder."""
    path = Path(path).resolve()
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = path.parent
    return cfg


def charset_list(cfg):
    """Ordered class list: lowercase, uppercase, digits, punctuation (space excluded)."""
    cs = cfg["charset"]
    return list(cs["lowercase"]) + list(cs["uppercase"]) + list(cs["digits"]) + list(cs["punctuation"])
