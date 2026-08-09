"""Mayo / AAPM low-dose CT paired loader.

Three v1 bugs are fixed here, and they matter more than any architecture change:

1.  **Normalisation broke the pair.**  v1 rescaled each slice by *its own*
    0.5/99.5 percentiles, so the low-dose and normal-dose images of the same
    slice received different affine maps.  No denoiser can undo that, and it
    silently caps every metric.  Both images now go through one fixed,
    global map.
2.  **Pairing was a sorted zip.**  The v1 run printed
    ``[WARN] No filename-key overlap; using sorted-zip (16628 pairs)`` — i.e. it
    never verified that LD[i] and ND[i] are the same slice.  We try a ladder of
    structural keys and then *verify* by correlation on a random sample.
3.  **The split was by slice.**  Taking the last 10 % of a slice list leaks
    neighbouring slices of the same patient into validation.  The split is now
    by patient.
"""
from __future__ import annotations

import glob
import os
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

IMG_EXTS = ("*.png", "*.tif", "*.tiff", "*.jpg", "*.jpeg", "*.bmp", "*.npy")
_DOSE_TOKENS = ("quarterdose", "fulldose", "quarter", "full", "lowdose",
                "normaldose", "lddose", "nddose", "dose", "quater")


def _list_images(root: str) -> list[str]:
    out: list[str] = []
    for ext in IMG_EXTS:
        out.extend(glob.glob(os.path.join(root, "**", ext), recursive=True))
    return sorted(out)


def _canon(s: str) -> str:
    s = re.sub(r"[^a-z0-9]", "", s.lower())
    for tok in _DOSE_TOKENS:
        s = s.replace(tok, "")
    return s


def _keys(path: str, root: str) -> list[str]:
    rel = Path(path).relative_to(root)
    parts = list(rel.parts)
    stem = Path(parts[-1]).stem
    nums_file = re.findall(r"\d+", stem)
    nums_dir = re.findall(r"\d+", "".join(parts[:-1]))
    return [
        "/".join(_canon(p) for p in parts),                       # full rel path
        _canon(parts[-2]) + "/" + _canon(stem) if len(parts) > 1 else _canon(stem),
        (":".join(nums_dir) + "#" + (nums_file[-1] if nums_file else stem)),
        _canon(stem),
    ]


def _patient_of(path: str, root: str) -> str:
    rel = Path(path).relative_to(root)
    m = re.search(r"[Ll]\d{3}", str(rel))
    if m:
        return m.group(0).upper()
    return _canon(rel.parts[0]) if len(rel.parts) > 1 else "P0"


def build_pairs(ld_root: str, nd_root: str, verbose=True):
    ld_files, nd_files = _list_images(ld_root), _list_images(nd_root)
    if not ld_files:
        raise FileNotFoundError(f"no images under {ld_root}")
    if not nd_files:
        raise FileNotFoundError(f"no images under {nd_root}")

    for level in range(4):
        ldm, ndm = {}, {}
        for p in ld_files:
            ldm.setdefault(_keys(p, ld_root)[level], p)
        for p in nd_files:
            ndm.setdefault(_keys(p, nd_root)[level], p)
        common = sorted(set(ldm) & set(ndm))
        if len(common) >= 0.8 * min(len(ld_files), len(nd_files)):
            pairs = [(ldm[k], ndm[k]) for k in common]
            if verbose:
                print(f"[data] paired {len(pairs)} slices by key level {level}")
            return pairs
    n = min(len(ld_files), len(nd_files))
    print(f"[data] WARNING: no structural key matched; falling back to sorted zip "
          f"({n} pairs). Verify the correlation report below before trusting results.")
    return list(zip(ld_files[:n], nd_files[:n]))


# ------------------------------------------------------------- raw -> HU --
def _load_raw(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        a = np.load(path)
        return a[..., 0] if a.ndim == 3 else a
    from PIL import Image
    im = Image.open(path)
    if im.mode in ("RGB", "RGBA"):
        im = im.convert("L")
    return np.array(im)


def detect_encoding(path: str) -> str:
    a = _load_raw(path)
    if np.issubdtype(a.dtype, np.floating) or a.min() < 0:
        return "raw_hu"
    if a.dtype == np.uint8 or a.max() <= 255:
        return "uint8_linear"
    return "uint16_hu_offset"


def raw_to_norm(a: np.ndarray, enc: str, hu_min: float, hu_max: float) -> np.ndarray:
    a = a.astype(np.float32)
    if enc == "uint8_linear":
        hu = a / 255.0 * (hu_max - hu_min) + hu_min
    elif enc == "uint16_hu_offset":
        hu = a - 1024.0
    else:
        hu = a
    x = (hu - hu_min) / (hu_max - hu_min) * 2.0 - 1.0
    return np.clip(x, -1.0, 1.0)


# ------------------------------------------------------------- Dataset ----
class MayoPairs(Dataset):
    def __init__(self, pairs, cfg, augment=False):
        self.pairs = pairs
        self.size = cfg.img_size
        self.hu_min, self.hu_max = cfg.hu_min, cfg.hu_max
        self.enc = cfg.src_encoding
        self.augment = augment
        if self.enc == "auto":
            self.enc = detect_encoding(pairs[0][1])

    def __len__(self):
        return len(self.pairs)

    def _one(self, path):
        x = raw_to_norm(_load_raw(path), self.enc, self.hu_min, self.hu_max)
        t = torch.from_numpy(np.ascontiguousarray(x))[None, None]
        if t.shape[-1] != self.size or t.shape[-2] != self.size:
            mode = "area" if t.shape[-1] > self.size else "bilinear"
            kw = {} if mode == "area" else {"align_corners": False}
            t = F.interpolate(t, (self.size, self.size), mode=mode, **kw)
        return t[0]

    def __getitem__(self, i):
        lp, np_ = self.pairs[i]
        ld, nd = self._one(lp), self._one(np_)
        if self.augment:
            if random.random() < 0.5:
                ld, nd = ld.flip(-1), nd.flip(-1)
            if random.random() < 0.5:
                ld, nd = ld.flip(-2), nd.flip(-2)
            k = random.randint(0, 3)
            if k:
                ld, nd = torch.rot90(ld, k, (-2, -1)), torch.rot90(nd, k, (-2, -1))
        return ld.contiguous(), nd.contiguous()


def verify_pairing(ds: MayoPairs, n=48, thresh=0.90) -> float:
    """Mean Pearson r between paired LD/ND slices.  A correct pairing on Mayo
    quarter-dose data sits well above 0.98; anything under ``thresh`` means the
    files are mis-matched and every downstream number is meaningless."""
    idx = random.Random(0).sample(range(len(ds)), min(n, len(ds)))
    rs = []
    for i in idx:
        ld, nd = ds[i]
        a, b = ld.flatten().numpy(), nd.flatten().numpy()
        if a.std() < 1e-6 or b.std() < 1e-6:
            continue
        rs.append(float(np.corrcoef(a, b)[0, 1]))
    r = float(np.mean(rs)) if rs else 0.0
    tag = "OK" if r >= thresh else "SUSPECT"
    print(f"[data] pairing check: mean r(LD, ND) = {r:.4f}  [{tag}]")
    if r < thresh:
        print("[data] >>> LD/ND slices look mis-paired. Fix the folder layout or "
              "set --ld_dirname/--nd_dirname before training. <<<")
    return r


def make_datasets(cfg, verbose=True):
    pairs = build_pairs(cfg.ld_root(), cfg.nd_root(), verbose)
    ld_root = cfg.ld_root()
    by_pat: dict[str, list] = {}
    for p in pairs:
        by_pat.setdefault(_patient_of(p[0], ld_root), []).append(p)
    pats = sorted(by_pat)
    if verbose:
        print(f"[data] {len(pats)} patient group(s): "
              f"{', '.join(f'{k}({len(by_pat[k])})' for k in pats[:12])}"
              f"{' ...' if len(pats) > 12 else ''}")

    n_val, n_test = cfg.val_patients, cfg.test_patients
    if len(pats) >= n_val + n_test + 1:
        test_p, val_p = pats[:n_test], pats[n_test:n_test + n_val]
        train_p = pats[n_test + n_val:]
        sel = lambda ps: [q for k in ps for q in by_pat[k]]
        tr, va, te = sel(train_p), sel(val_p), sel(test_p)
        if verbose:
            print(f"[data] patient split -> train {train_p}\n"
                  f"                        val   {val_p}\n"
                  f"                        test  {test_p}")
    else:
        print("[data] WARNING: too few patient groups to split by patient; "
              "falling back to a contiguous slice split (results will be "
              "optimistic — report this).")
        n = len(pairs)
        te, va, tr = pairs[:n // 10], pairs[n // 10:n // 5], pairs[n // 5:]

    if cfg.max_train_slices:
        tr = tr[:cfg.max_train_slices]
    train = MayoPairs(tr, cfg, augment=cfg.augment)
    val = MayoPairs(va, cfg, augment=False)
    test = MayoPairs(te, cfg, augment=False)
    if verbose:
        print(f"[data] slices: train={len(train)} val={len(val)} test={len(test)}"
              f"  encoding={train.enc}")
        verify_pairing(train)
    return train, val, test


def make_loaders(cfg, verbose=True):
    from torch.utils.data import DataLoader
    train, val, test = make_datasets(cfg, verbose)
    nw = cfg.num_workers
    common = dict(num_workers=nw, pin_memory=torch.cuda.is_available(),
                  persistent_workers=nw > 0, prefetch_factor=4 if nw > 0 else None)
    tl = DataLoader(train, batch_size=cfg.batch_size, shuffle=True,
                    drop_last=True, **common)
    vl = DataLoader(val, batch_size=cfg.eval_batch, shuffle=False, **common)
    sl = DataLoader(test, batch_size=cfg.eval_batch, shuffle=False, **common)
    return (train, val, test), (tl, vl, sl)
