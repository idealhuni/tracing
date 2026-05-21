#!/usr/bin/env python3
"""Step 2: Downsample + EDT -> prep_riem.npz"""
import argparse
import gc
import json
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, maximum_filter, zoom

warnings.filterwarnings('ignore')

# ── Config (fixed) ───────────────────────────────────────────
TARGET_VOXEL_DOWN_UM  = 0.7   # FMM 해상도 목표 (voxel_iso 기반 동적 계산)
BORDER_PAD_Z_MIN      = 0
BORDER_PAD_Z_MAX      = 8
BORDER_ARTIFACT_RATIO = 1.5
BORDER_PAD_XY         = 1
EDT_THRESHOLD         = 0.20
EDT_SMOOTH_SIG        = 1.0
EDT_RADIUS_SCALE      = 0.7


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)

    TUBULARITY_NPZ = out_dir / 'tubularity_anchored.npz'
    SOMA_NPZ       = out_dir / 'soma.npz'
    SOMA_JSON      = out_dir / 'soma.json'
    OUT_NPZ        = out_dir / 'prep_riem.npz'

    # ── Load + Downsample ────────────────────────────────────────
    t0 = time.time()

    tub          = np.load(str(TUBULARITY_NPZ))
    T            = tub['T_combined'].astype(np.float32)
    radius       = tub['radius_map'].astype(np.float32)
    orient_field = tub['orient_field'].astype(np.float32)
    voxel_iso    = float(tub['voxel_iso'])
    del tub; gc.collect()

    MAX_FMM_VOXELS = 60_000_000  # M tensor(9 float64) 기준 ~4.3 GB
    NZ, NY, NX = T.shape
    DOWNSAMPLE = max(1, round(TARGET_VOXEL_DOWN_UM / voxel_iso))
    # 볼륨 캡: 예상 voxel 수가 초과하면 DOWNSAMPLE 상향
    est_vox = (NZ // DOWNSAMPLE) * (NY // DOWNSAMPLE) * (NX // DOWNSAMPLE)
    if est_vox > MAX_FMM_VOXELS:
        DOWNSAMPLE = max(DOWNSAMPLE, int(np.ceil((NZ * NY * NX / MAX_FMM_VOXELS) ** (1/3))))
        print(f'  Volume cap: {est_vox/1e6:.0f}M > {MAX_FMM_VOXELS/1e6:.0f}M '
              f'→ DOWNSAMPLE adjusted to {DOWNSAMPLE}')
    print(f'voxel_iso={voxel_iso:.4f} um  DOWNSAMPLE={DOWNSAMPLE}  '
          f'-> voxel_down={voxel_iso*DOWNSAMPLE:.4f} um')

    soma_data = np.load(str(SOMA_NPZ))
    soma_mask = soma_data['soma_mask']
    del soma_data; gc.collect()

    with open(str(SOMA_JSON)) as f:
        soma = json.load(f)
    soma_vox  = np.array(soma['centroid_vox'], dtype=np.float32)
    soma_r_um = float(soma['radius_um'])

    factor = 1.0 / DOWNSAMPLE

    # Z 방향 먼저 max-pool로 gap 채우기 (광학적 Z-PSF blur 보상)
    # 7 voxel × 0.35µm ≈ 2.45µm — Z PSF FWHM ~1µm + 슬라이스 간 gap 커버
    T_z_filled = maximum_filter(T, size=(7, 1, 1))
    T_maxpool  = maximum_filter(T_z_filled, size=(1, DOWNSAMPLE, DOWNSAMPLE))
    del T_z_filled
    T_down    = T_maxpool[::DOWNSAMPLE, ::DOWNSAMPLE, ::DOWNSAMPLE].astype(np.float32)
    del T_maxpool; gc.collect()

    Zd, Yd, Xd = T_down.shape  # stride 기준 shape → 모든 zoom 결과를 이 shape에 맞춤

    def _zoom_match(arr, fac, order):
        """zoom 후 T_down shape에 맞게 crop/pad (off-by-one 방지)."""
        out = zoom(arr, fac, order=order)
        # 각 축 독립적으로 crop (초과) 또는 pad (부족)
        slices, pads = [], []
        for i, (s, t) in enumerate(zip(out.shape, (Zd, Yd, Xd) if arr.ndim == 3
                                       else (Zd, Yd, Xd)[:arr.ndim])):
            slices.append(slice(0, min(s, t)))
            pads.append((0, max(0, t - s)))
        out = out[tuple(slices)]
        if any(p[1] > 0 for p in pads):
            out = np.pad(out, pads, mode='edge')
        return out

    radius_down    = _zoom_match(radius,    factor, order=1).astype(np.float32)
    soma_mask_down = _zoom_match(soma_mask, factor, order=0).astype(bool)
    del T, radius, soma_mask; gc.collect()

    print('Downsampling orient_field...', flush=True)
    vz = _zoom_match(orient_field[..., 0], factor, order=1).astype(np.float32)
    vy = _zoom_match(orient_field[..., 1], factor, order=1).astype(np.float32)
    vx = _zoom_match(orient_field[..., 2], factor, order=1).astype(np.float32)
    del orient_field; gc.collect()

    norm = np.sqrt(vz**2 + vy**2 + vx**2) + 1e-8
    vz /= norm; vy /= norm; vx /= norm
    orient_down = np.stack([vz, vy, vx], axis=-1).astype(np.float16)
    del vz, vy, vx, norm; gc.collect()

    voxel_down    = voxel_iso * DOWNSAMPLE
    soma_vox_down = (soma_vox * factor).astype(np.float32)

    # ── Border pad auto-detect ───────────────────────────────────
    T_z = T_down.mean(axis=(1, 2))
    mid_s = Zd // 4
    mid_e = 3 * Zd // 4
    interior_ref = np.percentile(T_z[mid_s:mid_e], 75)

    def _detect_pad(profile, ref, ratio, max_pad):
        pad = 0
        for v in profile:
            if v > ref * ratio: pad += 1
            else: break
        return min(pad, max_pad)

    pad_top = _detect_pad(T_z,       interior_ref, BORDER_ARTIFACT_RATIO, BORDER_PAD_Z_MAX)
    pad_bot = _detect_pad(T_z[::-1], interior_ref, BORDER_ARTIFACT_RATIO, BORDER_PAD_Z_MAX)
    BORDER_PAD_Z = max(pad_top, pad_bot, BORDER_PAD_Z_MIN)

    if BORDER_PAD_Z > 0:
        T_down[:BORDER_PAD_Z]  = 0.0
        T_down[-BORDER_PAD_Z:] = 0.0
    if BORDER_PAD_XY > 0:
        p = BORDER_PAD_XY
        T_down[:, :p, :]  = 0.0; T_down[:, -p:, :] = 0.0
        T_down[:, :, :p]  = 0.0; T_down[:, :, -p:] = 0.0

    print(f'Border Z: top={pad_top}  bot={pad_bot}  pad={BORDER_PAD_Z} vox')
    print(f'T_down     : {T_down.shape}  voxel={voxel_down:.3f} um  ({time.time()-t0:.1f}s)')
    print(f'orient_down: {orient_down.shape}  dtype={orient_down.dtype}')
    print(f'soma_mask_down: {soma_mask_down.sum():,} voxels')

    # ── EDT radius ───────────────────────────────────────────────
    t0 = time.time()
    fg_mask  = T_down > EDT_THRESHOLD
    edt_raw  = distance_transform_edt(fg_mask).astype(np.float32) * voxel_down
    edt_down = gaussian_filter(edt_raw, sigma=EDT_SMOOTH_SIG).astype(np.float32)
    edt_down *= EDT_RADIUS_SCALE
    del edt_raw; gc.collect()

    fg_edt = edt_down[fg_mask]
    print(f'EDT done in {time.time()-t0:.1f}s')
    print(f'EDT (fg): min={fg_edt.min():.3f}  median={np.median(fg_edt):.3f}'
          f'  p99={np.percentile(fg_edt,99):.3f}  max={fg_edt.max():.3f} um')

    # ── Save ────────────────────────────────────────────────────
    np.savez_compressed(str(OUT_NPZ),
        T_down         = T_down,
        radius_down    = radius_down,
        edt_down       = edt_down,
        orient_down    = orient_down,
        soma_mask_down = soma_mask_down,
        voxel_down     = np.float32(voxel_down),
        voxel_iso      = np.float32(voxel_iso),
        downsample     = np.int32(DOWNSAMPLE),
        soma_vox       = soma_vox,
        soma_vox_down  = soma_vox_down,
        soma_r_um      = np.float32(soma_r_um),
        border_pad_z   = np.int32(BORDER_PAD_Z),
        border_pad_xy  = np.int32(BORDER_PAD_XY),
    )
    print(f'Saved: {OUT_NPZ}')
    print(f'  T_down         {T_down.shape}  float32')
    print(f'  edt_down       {edt_down.shape}  float32')
    print(f'  orient_down    {orient_down.shape}  float16')
    print(f'  soma_mask_down {soma_mask_down.shape}  bool')
    print(f'  voxel_down     {voxel_down:.3f} um')
    print(f'  border_pad_z   {BORDER_PAD_Z} vox')


if __name__ == '__main__':
    main()
