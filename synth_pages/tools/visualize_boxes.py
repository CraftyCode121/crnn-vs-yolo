"""Overlay YOLO character boxes on sample images to check that labels line up.

    python tools/visualize_boxes.py --folder train --n 4
    python tools/visualize_boxes.py --folder test_tiers_hard --n 4 --labels
Writes PNGs to output/preview/.
"""
import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generator import charset_list, load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--folder", required=True, help="e.g. train, val, test_tiers_hard")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--labels", action="store_true", help="also draw the character under each box")
    args = ap.parse_args()

    cfg = load_config(args.config)
    names = charset_list(cfg)
    ydir = cfg["_root"] / cfg["export"]["yolo"]["dir"]
    out = cfg["_root"] / cfg["paths"]["output_dir"] / "preview"
    out.mkdir(parents=True, exist_ok=True)

    imgs = sorted((ydir / "images" / args.folder).glob("*"))[: args.n]
    if not imgs:
        raise SystemExit(f"no images in {ydir / 'images' / args.folder}")
    for p in imgs:
        img = cv2.imread(str(p))
        H, W = img.shape[:2]
        lab = ydir / "labels" / args.folder / f"{p.stem}.txt"
        for line in lab.read_text().splitlines():
            if not line.strip():
                continue
            c, cx, cy, w, h = line.split()
            cx, cy, w, h = float(cx) * W, float(cy) * H, float(w) * W, float(h) * H
            x0, y0, x1, y1 = int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2)
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 220), 1)
            if args.labels:
                cv2.putText(img, names[int(c)], (x0, max(y0 - 2, 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 0, 0), 1)
        cv2.imwrite(str(out / f"{args.folder}_{p.stem}.png"), img)
        print("wrote", out / f"{args.folder}_{p.stem}.png")


if __name__ == "__main__":
    main()