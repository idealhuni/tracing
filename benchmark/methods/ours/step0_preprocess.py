#!/usr/bin/env python3
"""Step 0: Preprocess raw TIFF -> isotropic float32 volume."""
import argparse
import gc
import warnings
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage
from skimage.filters import threshold_triangle

warnings.filterwarnings('ignore')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir',           required=True)
    ap.add_argument('--data-path',         required=True)
    ap.add_argument('--voxel-xy',          type=float, required=True)
    ap.add_argument('--voxel-z',           type=float, required=True)
    ap.add_argument('--target-voxel-iso',  type=float, default=0.35)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    voxel_xy           = args.voxel_xy
    VOXEL_Z            = args.voxel_z
    TARGET_VOXEL_ISO   = args.target_voxel_iso

    NORMALIZE_ENABLE     = True
    CLIP_LOW_PERCENTILE  = None
    CLIP_HIGH_PERCENTILE = 99.9

    OUT_TIF  = out_dir / 'stack_preprocessed.tif'
    OUT_META = out_dir / 'preprocess_meta.npz'

    # ── Load ────────────────────────────────────────────────────
    print(f'Loading: {args.data_path}')
    stack_raw = tifffile.imread(args.data_path)
    if stack_raw.ndim == 2:
        stack_raw = stack_raw[np.newaxis]
    print(f'  Shape  : {stack_raw.shape}  (Z, Y, X)')
    print(f'  dtype  : {stack_raw.dtype}')
    print(f'  Memory : {stack_raw.nbytes / 1e9:.2f} GB')
    print(f'  XY voxel: {voxel_xy:.7f} um/px')
    print(f'  Z  voxel: {VOXEL_Z:.4f} um/slice')

    # ── 해상도 계획 ───────────────────────────────────────────────
    voxel_iso = TARGET_VOXEL_ISO if voxel_xy <= TARGET_VOXEL_ISO else voxel_xy
    zoom_xy   = voxel_xy / voxel_iso   # ≤ 1.0 (downsample only)
    zoom_z    = VOXEL_Z  / voxel_iso
    nZ, nY, nX = stack_raw.shape
    out_shape = (int(round(nZ * zoom_z)), int(round(nY * zoom_xy)), int(round(nX * zoom_xy)))
    print(f'  target voxel_iso : {TARGET_VOXEL_ISO} um  →  actual: {voxel_iso:.4f} um')
    print(f'  zoom_xy={zoom_xy:.4f}  zoom_z={zoom_z:.4f}')
    print(f'  {stack_raw.shape} → {out_shape}  (~{np.prod(out_shape)*4/1e9:.2f} GB)')

    # ── float32 → Normalize ──────────────────────────────────────
    stack_f = stack_raw.astype(np.float32)
    if NORMALIZE_ENABLE:
        p_low = (float(np.percentile(stack_f, CLIP_LOW_PERCENTILE))
                 if CLIP_LOW_PERCENTILE is not None
                 else float(threshold_triangle(stack_f)))
        p_high = float(np.percentile(stack_f, CLIP_HIGH_PERCENTILE))
        print(f'Clip range: {p_low:.1f} - {p_high:.1f}')
        stack_norm = np.clip(stack_f, p_low, p_high)
        stack_norm = (stack_norm - p_low) / (p_high - p_low)
    else:
        dtype_max = (float(np.iinfo(stack_raw.dtype).max)
                     if np.issubdtype(stack_raw.dtype, np.integer) else 1.0)
        p_low, p_high = 0.0, dtype_max
        stack_norm = stack_f / dtype_max
    del stack_f; gc.collect()
    print(f'Normalized: {stack_norm.min():.4f} - {stack_norm.max():.4f}')

    # ── Isotropic Resampling (fractional zoom) ───────────────────
    if abs(zoom_z - 1.0) < 0.02 and abs(zoom_xy - 1.0) < 0.02:
        stack_iso = stack_norm.copy()
        print('Already at target resolution — zoom skipped.')
    else:
        print(f'Resampling {stack_norm.shape} → {out_shape} ...')
        stack_iso = ndimage.zoom(stack_norm, (zoom_z, zoom_xy, zoom_xy), order=1, prefilter=False)
    del stack_norm; gc.collect()
    print(f'Output: {stack_iso.shape}  {stack_iso.nbytes/1e9:.2f} GB')
    print(f'Isotropic voxel size: {voxel_iso:.4f} um')

    # ── Save ────────────────────────────────────────────────────
    tifffile.imwrite(str(OUT_TIF), stack_iso)
    np.savez(str(OUT_META),
        voxel_iso = np.float32(voxel_iso),
        p_low     = np.float32(p_low),
        p_high    = np.float32(p_high),
        aniso     = np.float32(zoom_z),
        n2v_used  = np.bool_(False),
    )
    print(f'Saved: {OUT_TIF}  {stack_iso.shape}  {stack_iso.dtype}')
    print(f'Saved: {OUT_META}')
    print(f'  voxel_iso  : {voxel_iso:.4f} um')
    print(f'  zoom_xy    : {zoom_xy:.4f}')
    print(f'  zoom_z     : {zoom_z:.4f}')


if __name__ == '__main__':
    main()
