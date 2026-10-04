"""Writes YOLO and CRNN formats from one rendered+degraded page.

YOLO : one box per character (tight to the ink, 1 px padding), class id = index
       in the charset order. Boxes come from the warped label map.
CRNN : one rectified, fixed-height crop per text line. The crop rectangle is
       oriented along the line (fitted from its character boxes), so skewed
       pages do not leak neighbouring lines into the crop.
"""
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy import ndimage

from . import charset_list


def boxes_from_labels(labels, n_chars):
    """Return {char_id: (x0, y0, x1, y1)} for every character that still has pixels."""
    counts = np.bincount(labels.ravel(), minlength=n_chars + 1)
    objs = ndimage.find_objects(labels, max_label=n_chars)
    boxes = {}
    for i, sl in enumerate(objs, start=1):
        if sl is None or counts[i] == 0:
            continue
        boxes[i] = (sl[1].start, sl[0].start, sl[1].stop, sl[0].stop)
    return boxes


def yolo_lines(boxes, chars, class_of, W, H, pad=1):
    out = []
    for cid, (x0, y0, x1, y1) in sorted(boxes.items()):
        x0, y0, x1, y1 = max(x0 - pad, 0), max(y0 - pad, 0), min(x1 + pad, W), min(y1 + pad, H)
        cls = class_of[chars[cid - 1]["ch"]]
        out.append(f"{cls} {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}")
    return out


def line_crop(img, line_boxes, out_h, pad):
    """Rectified crop of one line, resized to a fixed height."""
    pts = np.array([[x, y] for (x0, y0, x1, y1) in line_boxes for (x, y) in
                    ((x0, y0), (x1, y0), (x1, y1), (x0, y1))], dtype=np.float64)
    centers = np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in line_boxes], dtype=np.float64)
    u = np.array([1.0, 0.0])
    if len(centers) >= 4:
        c0 = centers - centers.mean(axis=0)
        _, _, vt = np.linalg.svd(c0, full_matrices=False)
        u = vt[0]
        if u[0] < 0:
            u = -u
        if abs(np.degrees(np.arctan2(u[1], u[0]))) > 20:   # implausible, fall back to horizontal
            u = np.array([1.0, 0.0])
    n = np.array([-u[1], u[0]])
    mean = pts.mean(axis=0)
    pu, pn = (pts - mean) @ u, (pts - mean) @ n
    umin, umax, nmin, nmax = pu.min(), pu.max(), pn.min(), pn.max()
    w, h = int(round(umax - umin + 2 * pad)), int(round(nmax - nmin + 2 * pad))
    w, h = max(w, 4), max(h, 4)
    c = mean + u * (umin + umax) / 2 + n * (nmin + nmax) / 2
    # destination (x, y) -> source: c + u*(x - w/2) + n*(y - h/2)
    M = np.array([[u[0], n[0], c[0] - u[0] * w / 2 - n[0] * h / 2],
                  [u[1], n[1], c[1] - u[1] * w / 2 - n[1] * h / 2]], dtype=np.float32)
    crop = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REPLICATE)
    new_w = max(8, int(round(w * out_h / h)))
    return cv2.resize(crop, (new_w, out_h), interpolation=cv2.INTER_AREA if out_h < h else cv2.INTER_LINEAR)


def export_variant(cfg, charset_index, plan, img, labels, folder, page_id, ext):
    """Write one degraded variant. Returns (n_boxes, list of CRNN rows)."""
    H, W = labels.shape
    root = cfg["_root"] / cfg["paths"]["output_dir"]
    yolo_dir = cfg["_root"] / cfg["export"]["yolo"]["dir"]
    crnn_dir = cfg["_root"] / cfg["export"]["crnn"]["dir"]
    ccfg = cfg["export"]["crnn"]

    boxes = boxes_from_labels(labels, len(plan.chars))

    # YOLO 
    img_dir = yolo_dir / "images" / folder
    lab_dir = yolo_dir / "labels" / folder
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(img_dir / f"{page_id}.{ext}"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    with open(lab_dir / f"{page_id}.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(yolo_lines(boxes, plan.chars, charset_index, W, H)) + "\n")

    #  CRNN 
    by_line = {}
    for cid, ch in enumerate(plan.chars, start=1):
        by_line.setdefault(ch["line"], []).append(cid)
    crop_dir = crnn_dir / folder
    crop_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for li, ids in by_line.items():
        if any(i not in boxes for i in ids):      # a character lost all its pixels: skip line
            continue
        crop = line_crop(img, [boxes[i] for i in ids], ccfg["line_height_px"], ccfg["line_padding_px"])
        name = f"{page_id}_L{li:02d}.{ext}"
        cv2.imwrite(str(crop_dir / name), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        rows.append((f"{folder}/{name}", plan.lines[li]["text"], page_id, li))
    return len(boxes), rows


def write_data_yaml(cfg, folders):
    names = charset_list(cfg)
    yolo_dir = (cfg["_root"] / cfg["export"]["yolo"]["dir"]).resolve()
    tests = sorted(f for f in folders if f.startswith("test"))
    data = {
        "path": str(yolo_dir),
        "train": "images/train",
        "val": "images/val",
        "test": [f"images/{t}" for t in tests],
        "names": {i: c for i, c in enumerate(names)},
    }
    yolo_dir.mkdir(parents=True, exist_ok=True)
    with open(yolo_dir / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)