"""Entry point: runs the whole data generation pipeline.

Usage
    python build.py                      # full build from config.yaml
    python build.py --pilot              # ~1% build to check the pipeline
    python build.py --pages 50           # force 50 pages per split
    python build.py --splits train val   # only some splits
    python build.py --clean              # delete output/ yolo + crnn + metadata first
    python build.py --workers 8
"""
import argparse
import json
import shutil
import time
from multiprocessing import Pool

import numpy as np

from generator import charset_list, load_config
from generator.degrade import degrade
from generator.export import export_variant, write_data_yaml
from generator.render import plan_page, render_plan, scan_fonts
from generator.text_layout import TextSampler, load_text_params

_G = {}


def _init_worker(cfg, inventories):
    tp = load_text_params(cfg)
    _G["cfg"] = cfg
    _G["inv"] = inventories
    _G["sampler"] = TextSampler(tp, cfg["charset"])
    _G["cindex"] = {c: i for i, c in enumerate(charset_list(cfg))}


def _variants(cfg, split, rng):
    """List of (folder, tier_name, severity) for this page."""
    sv = cfg["splits"][split]["severity"]
    if sv["mode"] == "random":
        lo, hi = sv["range"]
        return [(split, "random", float(rng.uniform(lo, hi)))]
    out = []
    for t in sv["tiers"]:
        v = cfg["degradation"]["tiers"][t]
        sev = float(v) if np.isscalar(v) else float(rng.uniform(v[0], v[1]))
        out.append((f"{split}_{t}", t, sev))
    return out


def generate(task):
    split, split_idx, index = task
    cfg, sampler, cindex = _G["cfg"], _G["sampler"], _G["cindex"]
    inv = _G["inv"][cfg["splits"][split]["fonts"]]
    ss = np.random.SeedSequence([cfg["seed"], split_idx, index])
    r_layout, r_sev, r_deg = [np.random.default_rng(s) for s in ss.spawn(3)]
    page_id = f"{split}_{index:06d}"

    plan = plan_page(r_layout, cfg, sampler, inv)
    ink, labels = render_plan(plan)

    metas, rows = [], []
    for folder, tier, sev in _variants(cfg, split, r_sev):
        img, lab2, dmeta = degrade(ink, labels, sev, r_deg, cfg)
        n_boxes, crows = export_variant(cfg, cindex, plan, img, lab2, folder, page_id,
                                        cfg["export"]["image_format"])
        metas.append({
            "page_id": page_id, "split": split, "folder": folder, "tier": tier,
            "width": plan.width, "height": plan.height, **plan.meta,
            "n_boxes": n_boxes, **dmeta, "lines": [ln["text"] for ln in plan.lines],
        })
        font = plan.meta["font_body"]
        for path, text, pid, li in crows:
            rows.append((path, text, font, folder, dmeta["severity"], pid, li))
    return metas, rows


def write_dataset_info(cfg, config_path, metas):
    """Record what is needed to reproduce this dataset: config, fonts and library versions."""
    import hashlib
    import platform

    import cv2
    import PIL
    import scipy
    import yaml

    root = cfg["_root"]
    h = hashlib.sha256()
    for key in ("train", "val", "unseen_test"):
        for p in sorted((root / cfg["paths"]["fonts"][key]).glob("*")):
            if p.suffix.lower() in {e.lower() for e in cfg["fonts"]["extensions"]}:
                h.update(f"{key}/{p.name}".encode())
                h.update(p.read_bytes())
    info = {
        "seed": cfg["seed"],
        "config_sha256": hashlib.sha256(open(config_path, "rb").read()).hexdigest(),
        "fonts_sha256": h.hexdigest(),
        "images": len(metas),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pillow": PIL.__version__,
                     "opencv": cv2.__version__, "scipy": scipy.__version__, "pyyaml": yaml.__version__},
        "note": ("Same config + same fonts + same library versions reproduces the data exactly on the same "
                 "machine. Other machines may differ by tiny pixel amounts (FreeType / JPEG library versions)."),
    }
    with open(root / cfg["paths"]["output_dir"] / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--pilot", action="store_true", help="small run (config: pilot.scale)")
    ap.add_argument("--pages", type=int, default=None, help="override pages per split")
    ap.add_argument("--splits", nargs="*", default=None, help="only these splits")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--clean", action="store_true", help="remove old yolo/crnn/metadata output first")
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = cfg["_root"]
    workers = args.workers or cfg.get("workers", 1)

    # fonts 
    chars = charset_list(cfg)
    inventories = {}
    for key, rel in cfg["paths"]["fonts"].items():
        inv, skipped = scan_fonts(root / rel, set(cfg["fonts"]["extensions"]), chars)
        n = sum(len(v) for v in inv.values())
        need = cfg["fonts"]["min_fonts_per_split"].get(key, 1)
        print(f"fonts[{key}]: {n} usable files "
              f"(regular {len(inv['regular'])}, bold {len(inv['bold'])}, italic {len(inv['italic'])})"
              + (f", skipped {len(skipped)} lacking glyphs" if skipped else ""))
        if n == 0:
            raise SystemExit(f"No usable fonts in {root / rel}. Add .ttf/.otf files there.")
        if n < need:
            print(f"  WARNING: fewer than {need} font files in '{key}'; results will generalize less.")
        inventories[key] = inv

    #  output 
    out_root = root / cfg["paths"]["output_dir"]
    if args.clean:
        for sub in ("yolo", "crnn"):
            shutil.rmtree(root / cfg["export"][sub]["dir"], ignore_errors=True)
        (root / cfg["paths"]["metadata_file"]).unlink(missing_ok=True)

    # tasks 
    tasks = []
    for split_idx, (name, sc) in enumerate(cfg["splits"].items()):
        if args.splits and name not in args.splits:
            continue
        n = sc["pages"]
        if args.pages is not None:
            n = args.pages
        elif args.pilot:
            n = max(20, int(n * cfg["pilot"]["scale"]))
        tasks += [(name, split_idx, i) for i in range(n)]
    print(f"generating {len(tasks)} pages with {workers} worker(s)")

    metas, rows, folders = [], [], set()
    t0 = time.time()
    step = max(1, len(tasks) // 20)
    if workers > 1:
        with Pool(workers, initializer=_init_worker, initargs=(cfg, inventories)) as pool:
            results = pool.imap_unordered(generate, tasks, chunksize=8)
            for k, (m, r) in enumerate(results, 1):
                metas += m
                rows += r
                if k % step == 0:
                    print(f"  {k}/{len(tasks)}  ({time.time() - t0:.0f}s)", flush=True)
    else:
        _init_worker(cfg, inventories)
        for k, t in enumerate(tasks, 1):
            m, r = generate(t)
            metas += m
            rows += r
            if k % step == 0:
                print(f"  {k}/{len(tasks)}  ({time.time() - t0:.0f}s)", flush=True)

    # bookkeeping files 
    metas.sort(key=lambda m: (m["folder"], m["page_id"]))
    rows.sort(key=lambda r: (r[3], r[5], r[6]))
    meta_path = root / cfg["paths"]["metadata_file"]
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
            folders.add(m["folder"])

    crnn_dir = root / cfg["export"]["crnn"]["dir"]
    crnn_dir.mkdir(parents=True, exist_ok=True)
    # NOTE: texts contain " and ' characters. Read this file with quoting disabled
    # (csv.QUOTE_NONE, or pandas read_csv(..., sep="\t", quoting=3, keep_default_na=False)).
    with open(crnn_dir / cfg["export"]["crnn"]["labels_file"], "w", encoding="utf-8") as f:
        f.write("image_path\ttext\tfont\tsplit\tseverity\tpage_id\tline\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")

    write_data_yaml(cfg, folders)
    write_dataset_info(cfg, args.config, metas)

    
    print(f"\ndone in {time.time() - t0:.0f}s")
    print(f"  variants (images) : {len(metas)}")
    print(f"  CRNN line crops   : {len(rows)}")
    print(f"  mean lines/page   : {np.mean([m['n_lines'] for m in metas]):.1f}")
    print(f"  mean chars/page   : {np.mean([m['n_chars'] for m in metas]):.0f}")
    lost = sum(m["n_chars"] - m["n_boxes"] for m in metas)
    print(f"  characters without a box: {lost}")
    print(f"  output folder     : {out_root}")
    if args.splits or args.pages is not None or args.pilot:
        print("  note: metadata.jsonl, labels.tsv and data.yaml cover only THIS run.")


if __name__ == "__main__":
    main()