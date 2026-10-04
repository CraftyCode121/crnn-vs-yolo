"""Download fonts (optional), then split them BY FAMILY into
assets/fonts/{train,val,unseen_test}.

All styles of a family (Regular, Bold, Italic ...) go to the same split, so the
unseen_test fonts are genuinely unseen.

    # download ~100 families from Google Fonts and split them (needs internet)
    python tools/split_fonts.py --download 100

    # use fonts already on your machine
    python tools/split_fonts.py --src /usr/share/fonts ~/.fonts

    # both at once, redoing the split from scratch
    python tools/split_fonts.py --download 120 --src /usr/share/fonts --clear

Downloads come from the open-source Google Fonts repository on GitHub (OFL
licensed fonts only). They are cached in assets/fonts/_downloaded/, so a re-run
does not download files twice.
"""
import argparse
import hashlib
import json
import os
import random
import re
import shutil
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generator import charset_list, load_config  # noqa: E402
from generator.render import font_supports  # noqa: E402

# Family grouping

_STYLE = r"(?:extra|semi|demi|ultra|x)?(?:bold|italic|oblique|regular|light|thin|medium|black|heavy|book|roman|condensed|narrow)"
_STYLE_TOKEN = re.compile(rf"^{_STYLE}+$")
_STYLE_TAIL = re.compile(rf"{_STYLE}+$")
DEFAULT_EXCLUDE = r"script|dingbat|symbol|emoji|icon|brush|handwrit|blackletter|fraktur|barcode|braille|ornament"


def family_key(path):
    stem = re.sub(r"\[.*?\]", "", path.stem).lower()
    tokens = [t for t in re.split(r"[-_ ]+", stem) if t and not _STYLE_TOKEN.match(t)]
    key = re.sub(r"[^a-z0-9]", "", "".join(tokens))
    prev = None
    while prev != key:                       # strip style words glued to the name (e.g. 'dejavusansbold')
        prev = key
        key = _STYLE_TAIL.sub("", key) or key
    key = re.sub(r"\d{1,2}$", "", key) or key      # optical sizes, e.g. lmroman10 / lmroman12
    return key or stem


# Downloading (Google Fonts repository on GitHub)

GH_COMMIT = "https://api.github.com/repos/google/fonts/commits/{ref}"
GH_TREE = "https://api.github.com/repos/google/fonts/git/trees/{ref}?recursive=1"
GH_RAW = "https://raw.githubusercontent.com/google/fonts/{ref}/{path}"
_FONT_MAGIC = (b"\x00\x01\x00\x00", b"true", b"OTTO", b"ttcf")
CATEGORY_FILE = "_sources.json"         # filename -> {path, category, commit}, written by the downloader
MANIFEST_FILE = "fonts_manifest.json"   # the exact font set + split, for reproducing it elsewhere


def _get(url, token=None, retries=3, timeout=40):
    """GET a URL with retries; returns bytes."""
    headers = {"User-Agent": "synth-pages-font-downloader"}
    if token and "api.github.com" in url:
        headers["Authorization"] = f"Bearer {token}"
    last = None
    for i in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as r:
                return r.read()
        except Exception as e:           # network error, 403 rate limit, 404
            last = e
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"{url}: {last}")


def _pick_files(files, k):
    """Choose up to k files covering regular / italic / bold / bold-italic."""
    buckets = defaultdict(list)
    for f in files:
        n = f.lower()
        key = ("italic" in n or "oblique" in n, "bold" in n or "black" in n)
        plain = 0 if ("regular" in n or "[" in n or "-" not in n) else 1     # prefer the plain weight
        buckets[key].append((plain, f))
    out = []
    for key in [(False, False), (True, False), (False, True), (True, True)]:
        if buckets.get(key):
            out.append(sorted(buckets[key])[0][1])
    return out[:k]


def download_google_fonts(n_families, cache, seed, branch, max_files, categories, commit=None, workers=8):
    cache.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("GITHUB_TOKEN")
    fallback = ("\nCould not download automatically. Manual options:\n"
                "  git clone --depth 1 https://github.com/google/fonts\n"
                "  python tools/split_fonts.py --src fonts/ofl\n"
                "or download families from https://fonts.google.com and use --src <folder>.")
    print("listing the Google Fonts repository ...")
    try:
        # Pin the whole run to ONE commit, so file list and downloads cannot change under us.
        ref = commit or json.loads(_get(GH_COMMIT.format(ref=branch), token))["sha"]
        print(f"pinned to google/fonts commit {ref[:10]}")
        tree = json.loads(_get(GH_TREE.format(ref=ref), token))
    except Exception as e:
        raise SystemExit(f"error: {e}{fallback}")
    if tree.get("truncated"):
        print("  note: GitHub returned a partial file list; some families may be missing (fine for this purpose).")

    fams = defaultdict(list)
    for item in tree.get("tree", []):
        parts = item.get("path", "").split("/")
        if item.get("type") == "blob" and len(parts) == 3 and parts[0] == "ofl" and parts[2].lower().endswith(".ttf"):
            fams[parts[1]].append(parts[2])
    if not fams:
        raise SystemExit(f"error: no fonts found in the repository listing (branch '{branch}'?).{fallback}")

    names = sorted(fams)
    random.Random(seed).shuffle(names)
    allowed = {c.strip().upper() for c in categories.split(",") if c.strip()}

    def category(fam):
        try:
            txt = _get(GH_RAW.format(ref=ref, path=f"ofl/{fam}/METADATA.pb")).decode("utf-8", "ignore")
        except Exception:
            return fam, None                       # unreachable: skip this family
        m = re.search(r'category:\s*"([A-Z_]+)"', txt)
        return fam, (m.group(1) if m else "?")     # '?' = no category info: accept

    accepted, i, cat_of = [], 0, {}
    with ThreadPoolExecutor(workers) as ex:
        while len(accepted) < n_families and i < len(names):
            batch = names[i: i + max(8, 2 * (n_families - len(accepted)))]
            i += len(batch)
            for fam, cat in ex.map(category, batch):
                if cat is not None and (cat == "?" or cat in allowed) and len(accepted) < n_families:
                    accepted.append(fam)
                    cat_of[fam] = "UNKNOWN" if cat == "?" else cat
        print(f"selected {len(accepted)} families (categories: {', '.join(sorted(allowed))})")

        jobs = [(fam, f) for fam in accepted for f in _pick_files(fams[fam], max_files)]
        sidecar_path = cache / CATEGORY_FILE
        sidecar = json.loads(sidecar_path.read_text()) if sidecar_path.exists() else {}
        sidecar.update({f: {"path": f"ofl/{fam}/{f}", "category": cat_of[fam], "commit": ref} for fam, f in jobs})
        sidecar_path.write_text(json.dumps(sidecar, indent=1))

        def fetch(job):
            fam, fname = job
            target = cache / fname
            if target.exists() and target.stat().st_size > 1000:
                return "cached"
            try:
                data = _get(GH_RAW.format(ref=ref, path=urllib.parse.quote(f"ofl/{fam}/{fname}")))
            except Exception:
                return "failed"
            if data[:4] not in _FONT_MAGIC:
                return "failed"
            target.write_bytes(data)
            return "ok"

        stats = defaultdict(int)
        for k, res in enumerate(ex.map(fetch, jobs), 1):
            stats[res] += 1
            if k % 25 == 0 or k == len(jobs):
                print(f"  downloaded {k}/{len(jobs)} files", flush=True)
    print(f"download finished: {stats['ok']} new, {stats['cached']} already cached, {stats['failed']} failed")
    if stats["failed"] and not (stats["ok"] or stats["cached"]):
        raise SystemExit(f"error: every download failed.{fallback}")
    return ref


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def restore_from_manifest(manifest_path, dest, clear):
    """Re-create the exact font set recorded in a manifest (downloads pinned files, checks hashes)."""
    man = json.loads(Path(manifest_path).read_text())
    for d in dest.values():
        d.mkdir(parents=True, exist_ok=True)
        if clear:
            for f in d.iterdir():
                if f.suffix.lower() in (".ttf", ".otf"):
                    f.unlink()
    token = os.environ.get("GITHUB_TOKEN")
    ok, bad_hash, missing = 0, [], []
    entries = man["fonts"]

    def restore(e):
        target = dest[e["split"]] / e["file"]
        if target.exists() and _sha256(target) == e["sha256"]:
            return "ok"
        if not e.get("origin"):
            return "local"
        try:
            data = _get(GH_RAW.format(ref=e["commit"], path=urllib.parse.quote(e["origin"])))
        except Exception:
            return "failed"
        if hashlib.sha256(data).hexdigest() != e["sha256"]:
            return "hash"
        target.write_bytes(data)
        return "ok"

    with ThreadPoolExecutor(8) as ex:
        for e, res in zip(entries, ex.map(restore, entries)):
            if res == "ok":
                ok += 1
            elif res == "hash":
                bad_hash.append(e["file"])
            else:
                missing.append((e["file"], res))
    print(f"restored {ok}/{len(entries)} fonts from {manifest_path}")
    if bad_hash:
        print(f"  WARNING: {len(bad_hash)} files downloaded but their hash differs from the manifest: {bad_hash[:5]} ...")
    if missing:
        local = [f for f, r in missing if r == "local"]
        failed = [f for f, r in missing if r == "failed"]
        if local:
            print(f"  {len(local)} fonts came from a local folder on the original machine and cannot be "
                  f"downloaded; ask the author for them (they are listed in the manifest): {local[:5]} ...")
        if failed:
            print(f"  {len(failed)} downloads failed: {failed[:5]} ...")
    if bad_hash or missing:
        raise SystemExit(2)


# Splitting

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", nargs="*", default=[], help="folders to search for .ttf/.otf files")
    ap.add_argument("--download", type=int, default=0, metavar="N",
                    help="download N font families from Google Fonts (needs internet)")
    ap.add_argument("--categories", default="SANS_SERIF,SERIF,MONOSPACE",
                    help="Google Fonts categories to keep (add DISPLAY or HANDWRITING for more variety)")
    ap.add_argument("--max-files-per-family", type=int, default=4)
    ap.add_argument("--branch", default="main", help="branch of github.com/google/fonts")
    ap.add_argument("--commit", default=None, help="pin to this google/fonts commit SHA instead of the branch head")
    ap.add_argument("--from-manifest", default=None, metavar="FILE",
                    help="recreate exactly the font set recorded in a fonts_manifest.json (no new selection)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--dest", default=None, help="base folder holding train/ val/ unseen_test/ (default from config)")
    ap.add_argument("--val-frac", type=float, default=0.10, help="share of FAMILIES for validation")
    ap.add_argument("--test-frac", type=float, default=0.12, help="share of FAMILIES for unseen_test")
    ap.add_argument("--exclude", default=DEFAULT_EXCLUDE, help="regex of file names to skip (decorative fonts)")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--clear", action="store_true", help="delete fonts already in the destination folders first")
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = cfg["_root"]
    chars = charset_list(cfg)
    if args.dest:
        dest = {k: Path(args.dest) / k for k in ("train", "val", "unseen_test")}
    else:
        dest = {k: root / cfg["paths"]["fonts"][k] for k in ("train", "val", "unseen_test")}
    if args.from_manifest:
        restore_from_manifest(args.from_manifest, dest, args.clear)
        return
    for d in dest.values():
        d.mkdir(parents=True, exist_ok=True)
    if args.clear:
        for d in dest.values():
            for f in d.iterdir():
                if f.suffix.lower() in (".ttf", ".otf"):
                    f.unlink()

    src = list(args.src)
    if args.download:
        cache = dest["train"].parent / "_downloaded"
        download_google_fonts(args.download, cache, args.seed, args.branch,
                              args.max_files_per_family, args.categories, commit=args.commit)
        src.insert(0, str(cache))
    if not src:
        raise SystemExit("Nothing to do: pass --download N and/or --src <folder> ...")

    exts = {e.lower() for e in cfg["fonts"]["extensions"]}
    excl = re.compile(args.exclude, re.I) if args.exclude else None
    dest_resolved = [d.resolve() for d in dest.values()]

    families = defaultdict(list)
    seen_files, n_total, n_skip_glyph, n_skip_name = 0, 0, 0, 0
    for s in src:
        for p in sorted(Path(s).expanduser().rglob("*")):
            if p.suffix.lower() not in exts or not p.is_file():
                continue
            if any(str(p.resolve()).startswith(str(d)) for d in dest_resolved):
                continue
            n_total += 1
            if excl and excl.search(p.name):
                n_skip_name += 1
                continue
            if not font_supports(str(p), chars):
                n_skip_glyph += 1
                continue
            families[family_key(p)].append(p)
            seen_files += 1

    F = len(families)
    print(f"found {n_total} font files -> usable {seen_files} in {F} families "
          f"(skipped {n_skip_name} by name filter, {n_skip_glyph} lacking needed glyphs)")
    if F == 0:
        raise SystemExit("No usable fonts found. Check --src / --download, and retry.")

    # ---- stratified split: every font category is spread over train / val / unseen_test ----
    cat_file = dest["train"].parent / "_downloaded" / CATEGORY_FILE
    known = json.loads(cat_file.read_text()) if cat_file.exists() else {}

    def category_of(files):
        for p in files:
            if p.name in known:
                return known[p.name]["category"]
        return "MONOSPACE" if re.search(r"mono|courier|typewriter|code", files[0].name, re.I) else "UNKNOWN"

    groups = defaultdict(list)
    for k in sorted(families):
        groups[category_of(families[k])].append(k)
    rng = random.Random(args.seed)
    for g in groups.values():
        rng.shuffle(g)

    def allocate(total, sizes):
        """Share `total` slots over groups in proportion to their sizes (largest remainder)."""
        tot = sum(sizes.values())
        quota = {g: total * n / tot for g, n in sizes.items()}
        out = {g: int(q) for g, q in quota.items()}
        for g in sorted(quota, key=lambda g: quota[g] - out[g], reverse=True)[: total - sum(out.values())]:
            out[g] += 1
        return out

    n_test = min(max(1, round(F * args.test_frac)), max(F - 2, 1)) if F >= 3 else 0
    n_val = min(max(1, round(F * args.val_frac)), max(F - n_test - 1, 1)) if F >= 3 else 0
    sizes = {g: len(v) for g, v in groups.items()}
    t_alloc = allocate(n_test, sizes) if n_test else {g: 0 for g in sizes}
    v_alloc = allocate(n_val, {g: sizes[g] - t_alloc[g] for g in sizes}) if n_val else {g: 0 for g in sizes}
    split_of, cat_split = {}, defaultdict(lambda: defaultdict(int))
    for g, keys_g in groups.items():
        for i, k in enumerate(keys_g):
            sp = "unseen_test" if i < t_alloc[g] else ("val" if i < t_alloc[g] + v_alloc[g] else "train")
            split_of[k] = sp
            cat_split[g][sp] += 1

    counts = defaultdict(lambda: [0, 0])
    manifest = []
    for k, files in families.items():
        sp = split_of[k]
        counts[sp][0] += 1
        for p in files:
            target = dest[sp] / p.name
            j = 1
            while target.exists():
                target = dest[sp] / f"{p.stem}_{j}{p.suffix}"
                j += 1
            shutil.copy2(p, target)
            counts[sp][1] += 1
            src_info = known.get(p.name, {})
            manifest.append({"file": target.name, "split": sp, "family": k,
                             "category": src_info.get("category", category_of(files)),
                             "origin": src_info.get("path"), "commit": src_info.get("commit"),
                             "sha256": _sha256(target)})
    manifest.sort(key=lambda e: (e["split"], e["file"]))
    man_path = dest["train"].parent / MANIFEST_FILE
    man_path.write_text(json.dumps({"seed": args.seed, "val_frac": args.val_frac, "test_frac": args.test_frac,
                                    "fonts": manifest}, indent=1))

    need = cfg["fonts"]["min_fonts_per_split"]
    print("\nsplit          families  files   (config wants at least)")
    for sp in ("train", "val", "unseen_test"):
        fam, fil = counts[sp]
        flag = "" if fil >= need.get(sp, 1) else "   <-- fewer than recommended, get more fonts"
        print(f"{sp:<14} {fam:>8} {fil:>6}   ({need.get(sp, 1)}){flag}")
    print("\ncategory breakdown (families per split):")
    print(f"{'category':<12} {'train':>6} {'val':>5} {'unseen_test':>12}")
    for g in sorted(cat_split):
        print(f"{g:<12} {cat_split[g]['train']:>6} {cat_split[g]['val']:>5} {cat_split[g]['unseen_test']:>12}")
    style_n = defaultdict(lambda: defaultdict(int))
    for k, files in families.items():
        for p in files:
            n = p.name.lower()
            st = "italic" if ("italic" in n or "oblique" in n) else ("bold" if any(t in n for t in ("bold", "black", "heavy")) else "regular")
            style_n[split_of[k]][st] += 1
    print("\nfile styles per split:")
    for sp in ("train", "val", "unseen_test"):
        print(f"  {sp:<12} " + ", ".join(f"{st} {style_n[sp][st]}" for st in ("regular", "bold", "italic")))
    print(f"\nfonts copied into: {dest['train'].parent}")
    print(f"manifest written : {man_path}   (share this file to reproduce the exact font set)")


if __name__ == "__main__":
    main()