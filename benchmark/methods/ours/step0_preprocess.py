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
    ap.add_argument('--out-dir',       required=True)
    ap.add_argument('--data-path',     required=True)
    ap.add_argument('--voxel-xy',      type=float, required=True)
    ap.add_argument('--voxel-z',       type=float, required=True)
    ap.add_argument('--downsample-xy', type=int,   required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    voxel_xy      = args.voxel_xy
    VOXEL_Z       = args.voxel_z
    DOWNSAMPLE_XY = args.downsample_xy
    DOWNSAMPLE_Z  = 1

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
    print(f'  Anisotropy (Z/XY): {VOXEL_Z / voxel_xy:.2f}x')
    print(f'  DOWNSAMPLE_XY: {DOWNSAMPLE_XY}')

    # ── Downsample -> float32 -> Normalize ──────────────────────
    stack_ds = stack_raw[::DOWNSAMPLE_Z, ::DOWNSAMPLE_XY, ::DOWNSAMPLE_XY]
    vxy   = voxel_xy * DOWNSAMPLE_XY
    vz    = VOXEL_Z  * DOWNSAMPLE_Z
    aniso = vz / vxy
    print(f'After downsample: {stack_ds.shape}  aniso={aniso:.2f}x')

    stack_f = stack_ds.astype(np.float32)

    if NORMALIZE_ENABLE:
        p_low = (float(np.percentile(stack_f, CLIP_LOW_PERCENTILE))
                 if CLIP_LOW_PERCENTILE is not None
                 else float(threshold_triangle(stack_f)))
        p_high = float(np.percentile(stack_f, CLIP_HIGH_PERCENTILE))
        print(f'Clip range: {p_low:.1f} - {p_high:.1f}')
        stack_norm = np.clip(stack_f, p_low, p_high)
        stack_norm = (stack_norm - p_low) / (p_high - p_low)
    else:
        dtype_max = (float(np.iinfo(stack_ds.dtype).max)
                     if np.issubdtype(stack_ds.dtype, np.integer) else 1.0)
        p_low, p_high = 0.0, dtype_max
        stack_norm = stack_f / dtype_max

    del stack_f; gc.collect()
    print(f'Normalized: {stack_norm.min():.4f} - {stack_norm.max():.4f}')

    # ── Z Rescaling (isotropic) ──────────────────────────────────
    if aniso > 1.2:
        print(f'Rescaling Z by {aniso:.2f}x ...')
        stack_iso = ndimage.zoom(stack_norm, (aniso, 1.0, 1.0), order=1, prefilter=False)
        voxel_iso = vxy
        print(f'  {stack_norm.shape} -> {stack_iso.shape}')
        print(f'  Memory: {stack_iso.nbytes / 1e9:.2f} GB')
    else:
        stack_iso = stack_norm.copy()
        voxel_iso = vxy
        print('Near-isotropic — Z rescaling skipped.')

    del stack_norm; gc.collect()
    print(f'Isotropic voxel size: {voxel_iso:.4f} um')

    # N2V skipped (N2V_ENABLE=False)

    # ── Save ────────────────────────────────────────────────────
    tifffile.imwrite(str(OUT_TIF), stack_iso)
    np.savez(str(OUT_META),
        voxel_iso = np.float32(voxel_iso),
        p_low     = np.float32(p_low),
        p_high    = np.float32(p_high),
        aniso     = np.float32(aniso),
        n2v_used  = np.bool_(False),
    )
    print(f'Saved: {OUT_TIF}  {stack_iso.shape}  {stack_iso.dtype}')
    print(f'Saved: {OUT_META}')
    print(f'  voxel_iso  : {voxel_iso:.4f} um')
    print(f'  clip range : {p_low:.1f} - {p_high:.1f}')
    print(f'  anisotropy : {aniso:.2f}x')


if __name__ == '__main__':
    main()
