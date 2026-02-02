from dataclasses import dataclass
import yaml

@dataclass
class Config:
    paths: dict
    params: dict

def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return Config(paths=cfg["paths"], params=cfg["params"])