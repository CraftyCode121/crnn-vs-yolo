"""Draws text; returns an ink layer and a per-character label map.

Every character is rendered individually, so the ink image and the label map
come from exactly the same glyph pixels. Boxes are derived later from the label
map (after any warping), which keeps labels correct under every degradation.
"""
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .text_layout import sample_page_blocks, wrap_words


# Fonts

@lru_cache(maxsize=1024)
def get_font(path, size):
    return ImageFont.truetype(path, size)


def classify_style(name):
    n = name.lower()
    if "italic" in n or "oblique" in n:
        return "italic"
    if any(t in n for t in ("bold", "black", "heavy")):
        return "bold"
    return "regular"


def font_supports(path, chars):
    """True if the font draws a visible glyph for every character (and not the 'missing glyph' box)."""
    def draw(font, ch):
        im = Image.new("L", (80, 80), 0)
        ImageDraw.Draw(im).text((10, 60), ch, font=font, fill=255, anchor="ls")
        return im

    try:
        font = ImageFont.truetype(path, 40)
        missing = draw(font, "\U0010ffff").tobytes()
        for ch in chars:
            im = draw(font, ch)
            if im.getbbox() is None or im.tobytes() == missing:
                return False
        return True
    except Exception:
        return False


def scan_fonts(folder, extensions, chars):
    """Return {'regular': [...], 'bold': [...], 'italic': [...]} of usable font files."""
    inv = {"regular": [], "bold": [], "italic": []}
    skipped = []
    for p in sorted(Path(folder).rglob("*")):
        if p.suffix.lower() not in extensions:
            continue
        if font_supports(str(p), chars):
            inv[classify_style(p.name)].append(str(p))
        else:
            skipped.append(p.name)
    return inv, skipped


def pick_font(rng, inv, weights, want=None):
    styles = [s for s in ("regular", "bold", "italic") if inv.get(s)]
    if not styles:
        raise RuntimeError("No usable fonts available for this split.")
    if want and inv.get(want):
        style = want
    else:
        w = np.array([weights.get(s, 0.0) for s in styles], dtype=float)
        w = w / w.sum() if w.sum() > 0 else np.ones(len(styles)) / len(styles)
        style = styles[rng.choice(len(styles), p=w)]
    path = inv[style][rng.integers(0, len(inv[style]))]
    return path, style


# Page planning

@dataclass
class PagePlan:
    width: int
    height: int
    chars: list      # dicts: ch, x, y (baseline), line, path, size
    lines: list      # dicts: text, kind, n_chars
    meta: dict


def plan_page(rng, cfg, sampler, inv):
    L = cfg["layout"]
    cv = L["canvas"]
    W, dpi = cv["width_px"], cv["dpi"]
    blocks = sample_page_blocks(rng, sampler, L)

    mg = int(rng.integers(L["margins_px"]["min"], L["margins_px"]["max"] + 1))
    frac = rng.uniform(*L["column_fraction"])
    col_w = int((W - 2 * mg) * frac)
    col_left = mg + int(rng.integers(0, W - 2 * mg - col_w + 1))

    pt = rng.uniform(L["font_size_pt"]["min"], L["font_size_pt"]["max"])
    size = max(10, int(round(pt * dpi / 72)))
    body_path, body_style = pick_font(rng, inv, cfg["fonts"]["weights"])
    spacing = rng.uniform(L["line_spacing"]["min"], L["line_spacing"]["max"])
    ap = L["alignment_probs"]
    align = "justified" if rng.random() < ap["justified"] / (ap["left"] + ap["justified"]) else "left"
    indent_on = rng.random() < L["paragraph_indent_prob"]
    hcfg = L["heading"]

    max_h = cv["max_height_px"]
    max_lines = L["max_lines_per_page"]
    y = float(mg)
    lines, chars = [], []
    last_base, last_desc = None, 0
    stop = False

    for bi, blk in enumerate(blocks):
        if stop:
            break
        heading = blk["type"] == "heading"
        if heading:
            bold_path = None
            if rng.random() < hcfg["bold_prob"] and inv.get("bold"):
                bold_path, _ = pick_font(rng, inv, cfg["fonts"]["weights"], want="bold")
            path = bold_path or body_path
            fsize = int(size * rng.uniform(*hcfg["size_multiplier"]))
        else:
            path, fsize = body_path, size
        font = get_font(path, fsize)
        asc, desc = font.getmetrics()
        natural = asc + desc
        line_h = natural * (1.1 if heading else spacing)
        if lines:
            y += rng.uniform(L["paragraph_gap_lines"]["min"], L["paragraph_gap_lines"]["max"]) * natural
        space_w = font.getlength(" ")
        indent = fsize * rng.uniform(1.0, 2.5) if (not heading and indent_on) else 0.0

        cache = {}

        def measure(w, _font=font, _cache=cache):
            if w not in _cache:
                _cache[w] = _font.getlength(w)
            return _cache[w]

        wrapped = wrap_words(blk["words"], measure, col_w, space_w, first_indent=indent)
        for li, lw in enumerate(wrapped):
            if len(lines) >= max_lines or y + natural > max_h - mg:
                stop = True
                break
            base_y = y + asc
            first = li == 0
            avail = col_w - (indent if first else 0.0)
            x = col_left + (indent if first else 0.0)
            widths = [measure(w) for w in lw]
            nat_w = sum(widths) + space_w * (len(lw) - 1)
            justify = align == "justified" and not heading and li < len(wrapped) - 1 and len(lw) > 1
            gap = space_w + ((avail - nat_w) / (len(lw) - 1) if justify else 0.0)
            line_idx = len(lines)
            for w, ww in zip(lw, widths):
                for i, ch in enumerate(w):
                    cx = x + font.getlength(w[: i + 1]) - font.getlength(ch)
                    chars.append({"ch": ch, "x": cx, "y": base_y, "line": line_idx, "path": path, "size": fsize})
                x += ww + gap
            lines.append({"text": " ".join(lw), "kind": blk["type"], "n_chars": sum(len(w) for w in lw)})
            last_base, last_desc = base_y, desc
            y += line_h

    H = int(np.clip(math.ceil(last_base + last_desc + mg), cv["min_height_px"], max_h))
    meta = {
        "font_body": Path(body_path).name, "font_style": body_style, "size_pt": round(pt, 2),
        "size_px": size, "line_spacing": round(spacing, 3), "alignment": align,
        "margin_px": mg, "column_px": col_w, "indent": bool(indent_on),
        "n_lines": len(lines), "n_chars": len(chars),
    }
    return PagePlan(W, H, chars, lines, meta)


# Rendering

def render_plan(plan):
    """Return (ink uint8 HxW, labels int32 HxW). Label id = index in plan.chars + 1."""
    H, W = plan.height, plan.width
    ink = np.zeros((H, W), dtype=np.uint8)
    labels = np.zeros((H, W), dtype=np.int32)
    for cid, c in enumerate(plan.chars, start=1):
        font = get_font(c["path"], c["size"])
        l, t, r, b = font.getbbox(c["ch"], anchor="ls")
        gw, gh = int(r - l) + 2, int(b - t) + 2
        if gw <= 0 or gh <= 0:
            continue
        glyph = Image.new("L", (gw, gh), 0)
        ImageDraw.Draw(glyph).text((1 - l, 1 - t), c["ch"], font=font, fill=255, anchor="ls")
        g = np.asarray(glyph)
        x0 = int(round(c["x"])) + int(l) - 1
        y0 = int(round(c["y"])) + int(t) - 1
        x1, y1 = x0 + gw, y0 + gh
        cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
        if cx0 >= cx1 or cy0 >= cy1:
            continue
        gc = g[cy0 - y0: cy1 - y0, cx0 - x0: cx1 - x0]
        region = ink[cy0:cy1, cx0:cx1]
        np.maximum(region, gc, out=region)
        mask = gc > 64
        if not mask.any():
            mask = gc > 0
        labels[cy0:cy1, cx0:cx1][mask] = cid
    return ink, labels