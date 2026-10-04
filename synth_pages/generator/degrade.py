"""Geometric then photometric degradations, driven by one severity value in [0, 1].

effective_strength = max_value * severity (times a small random factor).
Geometric effects are applied to the ink layer AND the label map with the same
mapping (nearest-neighbour for labels), so boxes derived afterwards stay correct.
"""
import cv2
import numpy as np


def _u(rng, lo=0.6, hi=1.0):
    return rng.uniform(lo, hi)


# Geometric

def _fit_matrix(H3, bbox, W, H, pad):
    """Scale/shift H3 so the transformed content bbox stays inside the canvas."""
    x0, y0, x1, y1 = bbox
    corners = np.array([[x0, y0, 1], [x1, y0, 1], [x1, y1, 1], [x0, y1, 1]], dtype=float).T
    t = H3 @ corners
    t = t[:2] / t[2]
    minx, miny = t.min(axis=1)
    maxx, maxy = t.max(axis=1)
    bw, bh = maxx - minx, maxy - miny
    f = min(1.0, (W - 2 * pad) / bw, (H - 2 * pad) / bh)
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    nminx, nmaxx = cx - bw * f / 2, cx + bw * f / 2
    nminy, nmaxy = cy - bh * f / 2, cy + bh * f / 2
    dx = dy = 0.0
    if nminx < pad:
        dx = pad - nminx
    elif nmaxx > W - pad:
        dx = (W - pad) - nmaxx
    if nminy < pad:
        dy = pad - nminy
    elif nmaxy > H - pad:
        dy = (H - pad) - nmaxy
    F = np.array([[f, 0, cx - f * cx + dx], [0, f, cy - f * cy + dy], [0, 0, 1]], dtype=float)
    return F @ H3, f


def _smooth_field(rng, H, W, amp):
    """Smooth random displacement field with max |value| == amp."""
    small = rng.standard_normal((H // 16 + 3, W // 16 + 3)).astype(np.float32)
    small = cv2.GaussianBlur(small, (0, 0), 1.5)
    m = np.abs(small).max()
    if m > 0:
        small = small / m
    return cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR) * amp


def apply_geometric(ink, labels, s, rng, g):
    H, W = ink.shape
    meta = {"skew_deg": 0.0, "stretch_x": 1.0, "stretch_y": 1.0, "warp_px": 0.0, "persp": 0.0, "fit_scale": 1.0}
    if s <= 0:
        return ink, labels, meta
    ys, xs = np.nonzero(labels)
    if len(xs) == 0:
        return ink, labels, meta
    bbox = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2

    ang = rng.uniform(-1, 1) * g["skew_deg"]["max"] * s if g["skew_deg"]["enabled"] else 0.0
    sm = g["stretch"]["max"] * s if g["stretch"]["enabled"] else 0.0
    sx, sy = 1 + rng.uniform(-1, 1) * sm, 1 + rng.uniform(-1, 1) * sm
    amp = g["warp_px"]["max"] * s * _u(rng, 0.5, 1.0) if g["warp_px"]["enabled"] else 0.0
    pj = g["perspective"]["max"] * s if g["perspective"]["enabled"] else 0.0

    th = np.deg2rad(ang)
    c, sn = np.cos(th), np.sin(th)
    T1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=float)
    S = np.diag([sx, sy, 1.0])
    R = np.array([[c, -sn, 0], [sn, c, 0], [0, 0, 1]], dtype=float)
    T2 = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]], dtype=float)
    M3 = T2 @ R @ S @ T1

    P = np.eye(3)
    if pj > 0:
        src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
        jit = (rng.uniform(-1, 1, (4, 2)) * pj * np.array([W, H])).astype(np.float32)
        P = cv2.getPerspectiveTransform(src, src + jit)

    H3, f = _fit_matrix(P @ M3, bbox, W, H, pad=6 + amp)
    Hi = np.linalg.inv(H3)

    X, Y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    den = Hi[2, 0] * X + Hi[2, 1] * Y + Hi[2, 2]
    mx = (Hi[0, 0] * X + Hi[0, 1] * Y + Hi[0, 2]) / den
    my = (Hi[1, 0] * X + Hi[1, 1] * Y + Hi[1, 2]) / den
    if amp > 0:
        mx = mx + _smooth_field(rng, H, W, amp)
        my = my + _smooth_field(rng, H, W, amp)
    mx, my = mx.astype(np.float32), my.astype(np.float32)

    ink2 = cv2.remap(ink, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    lab2 = cv2.remap(labels.astype(np.float32), mx, my, cv2.INTER_NEAREST,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(np.int32)
    meta.update({"skew_deg": round(float(ang), 3), "stretch_x": round(float(sx), 4),
                 "stretch_y": round(float(sy), 4), "warp_px": round(float(amp), 3),
                 "persp": round(float(pj), 4), "fit_scale": round(float(f), 4)})
    return ink2, lab2, meta


# Photometric

def _lowfreq(rng, H, W, scale=24, sigma=2.0):
    """Smooth noise in roughly [-1, 1]."""
    small = rng.standard_normal((H // scale + 3, W // scale + 3)).astype(np.float32)
    small = cv2.GaussianBlur(small, (0, 0), sigma)
    small /= max(np.abs(small).max(), 1e-6)
    return cv2.resize(small, (W, H), interpolation=cv2.INTER_CUBIC)


def apply_photometric(ink, s, rng, p):
    H, W = ink.shape
    alpha = ink.astype(np.float32) / 255.0
    meta = {}

    # ink bleed: slight stroke thickening
    if p["ink_bleed_px"]["enabled"] and s > 0:
        px = p["ink_bleed_px"]["max"] * s * _u(rng, 0.5, 1.0)
        if px > 0.25:
            alpha = np.clip(cv2.GaussianBlur(alpha, (0, 0), 0.4 + 0.5 * px) * (1 + 1.6 * px), 0, 1)
        meta["ink_bleed"] = round(float(px), 3)

    # paper: base brightness, yellowing, blotches, grain
    base = _u(rng, 0.94, 1.0)
    paper = np.full((H, W, 3), base, dtype=np.float32)
    if p["paper_texture"]["enabled"] and s > 0:
        t = p["paper_texture"]["max"] * s * _u(rng, 0.6, 1.0)
        tint = np.array([1.0, 1.0 - 0.07 * t * 1.5, 1.0 - 0.28 * t], dtype=np.float32)   # RGB, warm
        paper = paper * tint
        paper *= (1 + 0.10 * t * _lowfreq(rng, H, W, 24, 2.0))[..., None]
        paper *= (1 + 0.05 * t * _lowfreq(rng, H, W, 6, 1.0))[..., None]
        paper += rng.standard_normal((H, W, 1)).astype(np.float32) * 0.025 * t
        meta["paper"] = round(float(t), 3)

    # ink colour: near-black, fades a little with severity
    ink_v = _u(rng, 0.05, 0.18) + 0.18 * s * _u(rng, 0.2, 1.0)
    ink_rgb = np.array([ink_v, ink_v * _u(rng, 0.95, 1.1), ink_v * _u(rng, 0.9, 1.15)], dtype=np.float32)
    img = paper * (1 - alpha[..., None] * (1 - ink_rgb))

    # stains (multiply blend)
    if p["stains"]["enabled"] and s > 0:
        n = int(rng.integers(0, int(round(p["stains"]["max_count"] * s)) + 1))
        for _ in range(n):
            m = np.zeros((H, W), dtype=np.float32)
            r = float(rng.uniform(15, 110))
            cx0, cy0 = float(rng.uniform(0, W)), float(rng.uniform(0, H))
            for _ in range(int(rng.integers(2, 5))):
                cv2.circle(m, (int(cx0 + rng.normal(0, r * 0.4)), int(cy0 + rng.normal(0, r * 0.4))),
                           int(r * rng.uniform(0.5, 1.0)), 1.0, -1)
            if rng.random() < 0.4:   # coffee-ring edge
                cv2.circle(m, (int(cx0), int(cy0)), int(r), 1.6, max(2, int(r * 0.08)))
            m = cv2.GaussianBlur(m, (0, 0), r * 0.3 + 1)
            m = np.clip(m, 0, 1.2) * p["stains"]["max_opacity"] * s * _u(rng, 0.5, 1.0)
            color = np.array([0.78, 0.64, 0.45], dtype=np.float32) * _u(rng, 0.8, 1.1)
            img *= 1 - m[..., None] * (1 - np.clip(color, 0, 1))
        meta["stains"] = n

    # uneven lighting
    if p["uneven_lighting"]["enabled"] and s > 0:
        a = p["uneven_lighting"]["max"] * s * _u(rng, 0.5, 1.0)
        th = rng.uniform(0, 2 * np.pi)
        xn = (np.arange(W, dtype=np.float32) / W - 0.5)[None, :]
        yn = (np.arange(H, dtype=np.float32) / H - 0.5)[:, None]
        grad = np.cos(th) * xn + np.sin(th) * yn
        vig = 1 - 0.6 * (xn ** 2 + yn ** 2)
        light = 1 + a * grad - 0.25 * a * (1 - vig)
        img *= light[..., None]
        meta["lighting"] = round(float(a), 3)

    # contrast loss
    if p["contrast_loss"]["enabled"] and s > 0:
        k = p["contrast_loss"]["max"] * s * _u(rng, 0.4, 1.0)
        mean = img.mean()
        img = mean + (img - mean) * (1 - k)
        meta["contrast_loss"] = round(float(k), 3)

    img = np.clip(img, 0, 1)

    # low resolution
    if p["low_res_scale"]["enabled"] and s > 0:
        sc = 1 - (1 - p["low_res_scale"]["min_scale"]) * s * _u(rng, 0.6, 1.0)
        if sc < 0.98:
            small = cv2.resize(img, (max(8, int(W * sc)), max(8, int(H * sc))), interpolation=cv2.INTER_AREA)
            img = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
        meta["low_res"] = round(float(sc), 3)

    # blur
    if p["blur_sigma"]["enabled"] and s > 0:
        sg = p["blur_sigma"]["max"] * s * _u(rng, 0.4, 1.0)
        if sg > 0.2:
            img = cv2.GaussianBlur(img, (0, 0), sg)
        meta["blur"] = round(float(sg), 3)

    # sensor noise
    if p["gaussian_noise"]["enabled"] and s > 0:
        sg = p["gaussian_noise"]["max_sigma"] / 255.0 * s * _u(rng, 0.4, 1.0)
        img = img + rng.standard_normal((H, W, 1)).astype(np.float32) * sg \
            + rng.standard_normal((H, W, 3)).astype(np.float32) * sg * 0.3
        meta["noise"] = round(float(sg * 255), 2)

    out = (np.clip(img, 0, 1) * 255).astype(np.uint8)

    # JPEG artifacts
    if p["jpeg_quality"]["enabled"] and s > 0.05:
        q = int(round(95 - (95 - p["jpeg_quality"]["min_quality"]) * s * _u(rng, 0.6, 1.0)))
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(out, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok:
            out = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        meta["jpeg_q"] = q
    return out, meta


def degrade(ink, labels, severity, rng, cfg):
    """Return (rgb uint8 image, warped labels, meta dict)."""
    d = cfg["degradation"]
    ink2, lab2, gmeta = apply_geometric(ink, labels, severity, rng, d["geometric"])
    img, pmeta = apply_photometric(ink2, severity, rng, d["photometric"])
    meta = {"severity": round(float(severity), 4), **gmeta, **pmeta}
    return img, lab2, meta