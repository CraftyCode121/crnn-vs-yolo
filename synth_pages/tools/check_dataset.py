"""Sanity checks on a generated dataset (run after build.py).

    python tools/check_dataset.py
Checks: label format, box counts, class balance, box sizes, tiers share pages,
font split has no leakage, CRNN labels match metadata, and exact reproducibility.
"""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build  
from generator import charset_list, load_config  
from generator.render import scan_fonts  

OK, BAD = "PASS", "FAIL"
results = []


def check(name, cond, detail=""):
    results.append(cond)
    print(f"[{OK if cond else BAD}] {name}" + (f"  ({detail})" if detail else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = cfg["_root"]
    names = charset_list(cfg)
    ydir = root / cfg["export"]["yolo"]["dir"]
    cdir = root / cfg["export"]["crnn"]["dir"]
    metas = [json.loads(l) for l in open(root / cfg["paths"]["metadata_file"], encoding="utf-8")]
    print(f"{len(metas)} image variants, {len(set(m['page_id'] for m in metas))} pages\n")

    # YOLO labels 
    bad_fmt, bad_count, n_cls = 0, 0, Counter()
    heights, widths = defaultdict(list), defaultdict(list)
    for m in metas:
        lab = (ydir / "labels" / m["folder"] / f"{m['page_id']}.txt").read_text().strip().splitlines()
        if len(lab) != m["n_boxes"]:
            bad_count += 1
        for line in lab:
            c, cx, cy, w, h = line.split()
            c = int(c)
            vals = [float(cx), float(cy), float(w), float(h)]
            if not (0 <= c < len(names)) or any(v < 0 or v > 1 for v in vals) or vals[2] <= 0 or vals[3] <= 0:
                bad_fmt += 1
            n_cls[c] += 1
            heights[m["tier"]].append(vals[3] * m["height"])
            widths[m["tier"]].append(vals[2] * m["width"])
    check("YOLO label format valid (class ids, normalized coords)", bad_fmt == 0, f"{bad_fmt} bad lines")
    check("label file line count == n_boxes in metadata", bad_count == 0, f"{bad_count} mismatches")
    lost = sum(m["n_chars"] - m["n_boxes"] for m in metas)
    check("every rendered character has a box", lost == 0, f"{lost} characters lost")

    # class balance (train-like data) 
    total = sum(n_cls.values())
    present = len(n_cls)
    check("all classes appear", present == len(names), f"{present}/{len(names)} classes")
    groups = {"lowercase": cfg["charset"]["lowercase"], "uppercase": cfg["charset"]["uppercase"],
              "digits": cfg["charset"]["digits"]}
    for g, chars in groups.items():
        cnt = [n_cls.get(names.index(c), 0) for c in chars]
        share = sum(cnt) / total
        print(f"      {g:<10} share {share:5.1%}   per-class min {min(cnt)}  max {max(cnt)}")
    punct = sum(n_cls.get(names.index(c), 0) for c in cfg["charset"]["punctuation"])
    print(f"      punctuation share {punct / total:5.1%}")
    rare = sorted(n_cls.items(), key=lambda kv: kv[1])[:6]
    print("      rarest classes:", ", ".join(f"{names[c]!r}:{n}" for c, n in rare))

    # box sizes (small-object risk for YOLO) 
    print()
    for t, hs in heights.items():
        hs = np.array(hs)
        ws = np.array(widths[t])
        print(f"      tier {t:<9} box height px: median {np.median(hs):4.0f}  p5 {np.percentile(hs, 5):4.0f}  "
              f"min {hs.min():3.0f} | width median {np.median(ws):3.0f}  p5 {np.percentile(ws, 5):3.0f}")

    # tiers share pages
    by_page = defaultdict(list)
    for m in metas:
        by_page[m["page_id"]].append(m)
    tier_pages = [v for k, v in by_page.items() if k.startswith("test_tiers")]
    if tier_pages:
        same = all(len(v) == 3 and len({tuple(x["lines"]) for x in v}) == 1 for v in tier_pages)
        check("test_tiers: each page appears at 3 severities with identical text", same, f"{len(tier_pages)} pages")
        sev_ok = all(
            {x["tier"]: x["severity"] for x in v}["clean"] == 0.0
            and 0.25 <= {x["tier"]: x["severity"] for x in v}["moderate"] <= 0.5
            and 0.7 <= {x["tier"]: x["severity"] for x in v}["hard"] <= 1.0 for v in tier_pages)
        check("tier severities fall in configured ranges", sev_ok)

    # font split has no leakage 
    fonts_by_split = defaultdict(set)
    for m in metas:
        fonts_by_split[cfg["splits"][m["split"]]["fonts"]].add(m["font_body"])
    allsets = list(fonts_by_split.items())
    leak = any(a[1] & b[1] for i, a in enumerate(allsets) for b in allsets[i + 1:])
    check("font sets (train / val / unseen_test) do not overlap", not leak,
          ", ".join(f"{k}: {len(v)} fonts used" for k, v in fonts_by_split.items()))

    # CRNN labels vs metadata 
    rows = list(csv.DictReader(open(cdir / cfg["export"]["crnn"]["labels_file"], encoding="utf-8"),
                           delimiter="\t", quoting=csv.QUOTE_NONE))
    page_lines = {(m["page_id"], m["folder"]): m["lines"] for m in metas}
    mism = sum(1 for r in rows if page_lines[(r["page_id"], r["split"])][int(r["line"])] != r["text"])
    check("CRNN text == metadata line text", mism == 0, f"{len(rows)} crops, {mism} mismatches")
    exp_crops = sum(len(m["lines"]) for m in metas)
    check("every line has a crop", len(rows) == exp_crops, f"{len(rows)}/{exp_crops}")
    sample = [rows[i] for i in np.linspace(0, len(rows) - 1, min(60, len(rows))).astype(int)]
    hs = {cv2.imread(str(cdir / r["image_path"])).shape[0] for r in sample}
    check("crop height is fixed", hs == {cfg["export"]["crnn"]["line_height_px"]}, f"heights {hs}")
    widths_c = [cv2.imread(str(cdir / r["image_path"])).shape[1] for r in sample]
    print(f"      crop width px: min {min(widths_c)}  median {int(np.median(widths_c))}  max {max(widths_c)}")
    lens = [len(r["text"]) for r in rows]
    print(f"      line length chars: min {min(lens)}  median {int(np.median(lens))}  max {max(lens)}")
    ratio = [w / max(len(r["text"]), 1) for w, r in zip(widths_c, sample)]
    print(f"      crop px per character: min {min(ratio):.1f}  median {np.median(ratio):.1f}  "
          f"(CTC needs >1 time step per char after the CNN)")

    # reproducibility 
    inventories = {}
    chars = charset_list(cfg)
    for key, rel in cfg["paths"]["fonts"].items():
        inventories[key], _ = scan_fonts(root / rel, set(cfg["fonts"]["extensions"]), chars)
    build._init_worker(cfg, inventories)
    tgt = metas[0]
    split_idx = list(cfg["splits"]).index(tgt["split"])
    index = int(tgt["page_id"].split("_")[-1])
    before = cv2.imread(str(ydir / "images" / tgt["folder"] / f"{tgt['page_id']}.{cfg['export']['image_format']}"))
    build.generate((tgt["split"], split_idx, index))
    after = cv2.imread(str(ydir / "images" / tgt["folder"] / f"{tgt['page_id']}.{cfg['export']['image_format']}"))
    check("regenerating a page gives an identical image", before.shape == after.shape and (before == after).all(),
          tgt["page_id"])

    print(f"\n{sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()