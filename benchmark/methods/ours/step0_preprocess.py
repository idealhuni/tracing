#!/usr/bin/env python3
"""Step 0: Preprocess raw TIFF -> isotropic float32 volume."""
import argparse
import gc
import time
import warnings
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage
from skimage.filters import threshold_triangle

warnings.filterwarnings('ignore')

# ── LBN (Local Background Normalization) 파라미터 ───────────────────────────
LBN_SIGMA_UM  = 40.0   # rolling-ball background Gaussian σ (µm)
LBN_THRESHOLD = 0.05   # 배경 공간 불균일성 임계값 (정규화된 [0,1] 범위의 분율)
LBN_BLOCKS    = 6      # 불균일성 측정용 축당 블록 수


def _bg_nonuniformity(stack: np.ndarray, n_blocks: int = LBN_BLOCKS) -> float:
    """블록별 배경(p5)의 공간적 표준편차를 신호 범위의 분율로 반환.
    값이 높을수록 배경이 불균일 → LBN 적용 권장."""
    step = max(1, min(stack.shape) // 48)
    sub  = stack[::step, ::step, ::step]
    bz   = max(1, sub.shape[0] // n_blocks)
    by_  = max(1, sub.shape[1] // n_blocks)
    bx   = max(1, sub.shape[2] // n_blocks)
    bg   = []
    for z0 in range(0, sub.shape[0], bz):
        for y0 in range(0, sub.shape[1], by_):
            for x0 in range(0, sub.shape[2], bx):
                blk = sub[z0:z0+bz, y0:y0+by_, x0:x0+bx]
                if blk.size >= 4:
                    bg.append(float(np.percentile(blk, 5)))
    bg = np.array(bg, dtype=np.float32)
    sig_range = max(float(stack.max() - stack.min()), 1e-6)
    return float(np.std(bg) / sig_range)


def _rolling_ball(stack: np.ndarray, sigma_vox: float) -> np.ndarray:
    """Gaussian rolling-ball 배경 차감. 음수 clip 후 float32 반환."""
    bg = ndimage.gaussian_filter(stack.astype(np.float32), sigma=float(sigma_vox))
    return np.maximum(stack - bg, 0.0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir',           required=True)
    ap.add_argument('--data-path',         required=True)
    ap.add_argument('--voxel-xy',          type=float, required=True)
    ap.add_argument('--voxel-z',           type=float, required=True)
    ap.add_argument('--target-voxel-iso',  type=float, default=0.35)
    ap.add_argument('--xy-max',            type=int,   default=4096)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    voxel_xy           = args.voxel_xy
    VOXEL_Z            = args.voxel_z
    TARGET_VOXEL_ISO   = args.target_voxel_iso
    XY_MAX             = args.xy_max

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
    MAX_OUTPUT_VOXELS = 700_000_000  # step1 GPU 메모리 한계 기준

    nZ, nY, nX = stack_raw.shape

    # 1) target voxel iso (XY 업샘플 금지)
    voxel_iso = TARGET_VOXEL_ISO if voxel_xy <= TARGET_VOXEL_ISO else voxel_xy

    # 2) 볼륨 캡: 물리적 부피 기반으로 voxel_iso 자동 조정
    zoom_xy = min(1.0, voxel_xy / voxel_iso)
    zoom_z  = VOXEL_Z / voxel_iso
    est_voxels = int(nZ * zoom_z * nY * zoom_xy * nX * zoom_xy)
    if est_voxels > MAX_OUTPUT_VOXELS:
        viso_min  = (nZ * nY * nX * VOXEL_Z * voxel_xy**2 / MAX_OUTPUT_VOXELS) ** (1/3)
        voxel_iso = max(voxel_iso, viso_min)
        zoom_xy   = min(1.0, voxel_xy / voxel_iso)
        zoom_z    = VOXEL_Z / voxel_iso
        print(f'  Volume cap: {est_voxels/1e6:.0f}M > {MAX_OUTPUT_VOXELS/1e6:.0f}M '
              f'→ voxel_iso adjusted to {voxel_iso:.4f} um')

    # 3) XY pixel hard cap (안전망 — 물리적 의미 없는 극단적 케이스만 차단)
    if max(nY, nX) * zoom_xy > XY_MAX:
        voxel_iso = max(voxel_iso, voxel_xy * max(nY, nX) / XY_MAX)
        zoom_xy   = min(1.0, voxel_xy / voxel_iso)
        zoom_z    = VOXEL_Z / voxel_iso
        print(f'  XY hard cap ({XY_MAX}px): voxel_iso adjusted to {voxel_iso:.4f} um')

    out_shape = (int(round(nZ * zoom_z)), int(round(nY * zoom_xy)), int(round(nX * zoom_xy)))
    print(f'  target voxel_iso : {TARGET_VOXEL_ISO} um  →  actual: {voxel_iso:.4f} um')
    print(f'  zoom_xy={zoom_xy:.4f}  zoom_z={zoom_z:.4f}')
    print(f'  {stack_raw.shape} → {out_shape}  (~{np.prod(out_shape)*4/1e9:.2f} GB)')

    # ── float32 → Normalize ──────────────────────────────────────
    stack_f = stack_raw.astype(np.float32)
    if NORMALIZE_ENABLE:
        p_high = float(np.percentile(stack_f, CLIP_HIGH_PERCENTILE))
        if CLIP_LOW_PERCENTILE is not None:
            p_low = float(np.percentile(stack_f, CLIP_LOW_PERCENTILE))
            norm_mode = f'{CLIP_LOW_PERCENTILE}th-pct'
        else:
            p_low_tri = float(threshold_triangle(stack_f))
            # 저대비 감지: triangle이 p_high의 30% 이상 → 노이즈 과증폭 위험
            # 예) neuron4: 36.35/41.0=0.89 (위험), s10mm: 3.49/93=0.04 (정상)
            if p_low_tri / max(p_high, 1e-6) > 0.3:
                p_low = float(np.percentile(stack_f, 1))
                norm_mode = (f'1st-pct (auto: triangle={p_low_tri:.1f} '
                             f'was {100*p_low_tri/p_high:.0f}% of p_high={p_high:.1f})')
            else:
                p_low = p_low_tri
                norm_mode = f'triangle'
        print(f'Clip range: {p_low:.1f} - {p_high:.1f}  [{norm_mode}]')
        stack_norm = np.clip(stack_f, p_low, p_high)
        stack_norm = (stack_norm - p_low) / (p_high - p_low + 1e-10)
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

    # ── Local Background Normalization (LBN) ────────────────────────────────
    sigma_vox   = LBN_SIGMA_UM / voxel_iso
    bg_score    = _bg_nonuniformity(stack_iso)
    lbn_applied = bool(bg_score > LBN_THRESHOLD)
    print(f'LBN check: bg_nonuniformity={bg_score:.4f} (threshold={LBN_THRESHOLD})'
          f' → {"APPLY" if lbn_applied else "SKIP"}')
    if lbn_applied:
        print(f'  rolling-ball σ={LBN_SIGMA_UM}µm = {sigma_vox:.1f}vox ...', end=' ', flush=True)
        t_lbn     = time.time()
        stack_lbn = _rolling_ball(stack_iso, sigma_vox)
        del stack_iso; gc.collect()
        pos       = stack_lbn[stack_lbn > 0]
        p999      = float(np.percentile(pos, 99.9)) if pos.size > 0 else 1.0
        stack_iso = np.clip(stack_lbn / p999, 0.0, 1.0).astype(np.float32)
        del stack_lbn, pos; gc.collect()
        print(f'{time.time()-t_lbn:.1f}s  '
              f'range=[{stack_iso.min():.4f}, {stack_iso.max():.4f}]')

    # ── Save ────────────────────────────────────────────────────
    tifffile.imwrite(str(OUT_TIF), stack_iso)
    np.savez(str(OUT_META),
        voxel_iso    = np.float32(voxel_iso),
        p_low        = np.float32(p_low),
        p_high       = np.float32(p_high),
        aniso        = np.float32(zoom_z),
        n2v_used     = np.bool_(False),
        lbn_applied  = np.bool_(lbn_applied),
        lbn_sigma_um = np.float32(LBN_SIGMA_UM if lbn_applied else 0.0),
    )
    print(f'Saved: {OUT_TIF}  {stack_iso.shape}  {stack_iso.dtype}')
    print(f'Saved: {OUT_META}')
    print(f'  voxel_iso  : {voxel_iso:.4f} um')
    print(f'  zoom_xy    : {zoom_xy:.4f}')
    print(f'  zoom_z     : {zoom_z:.4f}')


if __name__ == '__main__':
    main()
