import numpy as np
import re

def _is_nan(x) -> bool:
    return isinstance(x, float) and np.isnan(x)

def clean_text(x) -> str:
    if x is None or _is_nan(x):
        return ""
    s = str(x)
    s = s.replace("\u00ae", " ").replace("\u2122", " ")
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > 1200:
        s = s[:1200]
    return s

def listish_to_text(v) -> str:
    if v is None or _is_nan(v):
        return ""
    if isinstance(v, list):
        out = []
        for item in v:
            if isinstance(item, dict):
                for k in ("Value", "Text", "value", "text"):
                    if k in item and item[k]:
                        out.append(clean_text(item[k]))
                        break
            else:
                t = clean_text(item)
                if t:
                    out.append(t)
        return " | ".join([x for x in out if x])
    if isinstance(v, dict):
        parts = []
        for k in v:
            t = clean_text(v.get(k))
            if t:
                parts.append(t)
        return " | ".join(parts)
    return clean_text(v)

def extract_spec_tokens(text: str) -> str:
    if not text:
        return ""
    t = text.lower()
    out = []

    out += [f"gb:{m}" for m in re.findall(r"\b(\d{2,4})\s*gb\b", t)]
    out += [f"mah:{m}" for m in re.findall(r"\b(\d{3,5})\s*mah\b", t)]
    out += [f"hz:{m}" for m in re.findall(r"\b(\d{2,4})\s*hz\b", t)]
    out += [f"w:{m}" for m in re.findall(r"\b(\d{2,4})\s*w\b", t)]
    out += [f"in:{m}" for m in re.findall(r"\b(\d{1,2}\.?\d?)\s*(?:inch|inches|in|\" )\b", t)]
    out += [f"res:{a}x{b}" for a, b in re.findall(r"\b(\d{3,4})\s*x\s*(\d{3,4})\b", t)]

    if len(out) > 50:
        out = out[:50]
    return " ".join(out)

def row_to_doc(row: dict) -> str:
    parts = []
    for k, v in row.items():
        if k in ("BulletPoints.Values", "BulletPoints"):
            v = listish_to_text(v)
        else:
            v = clean_text(v)
        if v:
            parts.append(f"{k.lower()}: {v}")
        doc = "\n".join(parts)
        spec = extract_spec_tokens(doc)
        if spec:
         doc = doc + "\n" + "spec_tokens: " + spec
        return doc