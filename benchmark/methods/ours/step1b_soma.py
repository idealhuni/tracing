#!/usr/bin/env python3
"""Step 1b: Soma detection + segmentation + anchor -> soma.npz, tubularity_anchored.npz"""
import argparse
import gc
import json
import warnings
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import (binary_closing, binary_erosion, binary_fill_holes,
                            binary_opening, distance_transform_edt,
                            gaussian_filter, label as sp_label)
from skimage.filters import threshold_otsu
from skimage.morphology import ball

warnings.filterwarnings('ignore')

# ── Config (fixed) ───────────────────────────────────────────
SOMA_VOXEL            = None
SOMA_HOLLOW           = False
SOMA_SIGMA_UM         = 4.0
SOMA_SEARCH_RADIUS_UM = 30.0
SOMA_MAX_TUBULARITY   = 0.25
SOMA_OPEN_RADIUS_UM   = 2.0
SOMA_CLOSE_RADIUS_UM  = 1.5
SOMA_ERODE_RADIUS_UM  = 0
SOMA_BORDER_PAD_UM    = 15.0  # µm — exclusion margin from image edges for soma detection
SOMA_ANCHOR_OFFSET    = 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--soma-voxel', nargs=3, type=int, default=None,
                    metavar=('Z', 'Y', 'X'),
                    help='Pin soma seed voxel (skips auto-detection)')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)

    if args.soma_voxel is not None:
        global SOMA_VOXEL
        SOMA_VOXEL = tuple(args.soma_voxel)

    TUBULARITY_NPZ          = out_dir / 'tubularity.npz'
    STACK_TIF               = out_dir / 'stack_preprocessed.tif'
    OUT_SOMA_NPZ            = out_dir / 'soma.npz'
    OUT_SOMA_JSON           = out_dir / 'soma.json'
    OUT_TUBULARITY_ANCHORED = out_dir / 'tubularity_anchored.npz'

    # ── Load ────────────────────────────────────────────────────
    tub        = np.load(str(TUBULARITY_NPZ))
    T_combined = tub['T_combined']
    I_OOF_raw  = tub['I_OOF_raw']
    voxel_iso  = float(tub['voxel_iso'])
    stack      = tifffile.imread(str(STACK_TIF)).astype(np.float32)
    print(f'Loaded: T_combined {T_combined.shape}  voxel_iso={voxel_iso:.4f} um')

    SOMA_SIGMA           = SOMA_SIGMA_UM         / voxel_iso
    SOMA_SEARCH_RADIUS   = max(10, int(round(SOMA_SEARCH_RADIUS_UM / voxel_iso)))
    SOMA_BORDER_PAD      = max(10, int(round(SOMA_BORDER_PAD_UM    / voxel_iso)))
    SOMA_OPEN_RADIUS_VX  = max(1,  int(round(SOMA_OPEN_RADIUS_UM   / voxel_iso)))
    SOMA_CLOSE_RADIUS_VX = max(1,  int(round(SOMA_CLOSE_RADIUS_UM  / voxel_iso)))
    SOMA_ERODE_RADIUS_VX = max(0,  int(round(SOMA_ERODE_RADIUS_UM  / voxel_iso)))
    print(f'Soma mode: {"hollow" if SOMA_HOLLOW else "filled"}')
    print(f'  sigma={SOMA_SIGMA:.1f}  search_r={SOMA_SEARCH_RADIUS}'
          f'  open={SOMA_OPEN_RADIUS_VX}  close={SOMA_CLOSE_RADIUS_VX}'
          f'  erode={SOMA_ERODE_RADIUS_VX}  border_pad={SOMA_BORDER_PAD}')

    # ── Soma Detection ───────────────────────────────────────────
    if SOMA_VOXEL is not None:
        soma_voxel = tuple(int(v) for v in SOMA_VOXEL)
        print(f'Soma pinned manually: {soma_voxel}')
    else:
        stack_f32   = stack.astype(np.float32)
        stack_norm  = (stack_f32 - stack_f32.min()) / (stack_f32.max() - stack_f32.min() + 1e-6)
        z_sums      = stack_norm.sum(axis=(1, 2))
        nz          = stack_norm.shape[0]
        front_frac  = float(z_sums[:max(1, nz // 10)].sum()) / max(float(z_sums.sum()), 1e-8)
        if front_frac > 0.3:
            # Front-loaded Z-gradient (2-photon depth attenuation): equalise slices
            print(f'  Z-gradient detected (front-10% = {front_frac:.2f}) -> per-Z normalisation')
            soma_score = stack_norm / np.maximum(z_sums[:, None, None], 1e-8)
        else:
            print(f'  No Z-gradient (front-10% = {front_frac:.2f}) -> intensity score')
            soma_score = stack_norm
        soma_smooth = gaussian_filter(soma_score, sigma=SOMA_SIGMA)
        p = SOMA_BORDER_PAD
        NZ, NY, NX = soma_smooth.shape
        border_mask = np.ones_like(soma_smooth, dtype=bool)
        border_mask[p:NZ-p, p:NY-p, p:NX-p] = False
        soma_smooth[border_mask] = 0.0
        soma_voxel = tuple(int(v) for v in
                           np.unravel_index(soma_smooth.argmax(), soma_smooth.shape))
        print(f'Auto-detected soma voxel: {soma_voxel}')
        print(f'  soma_score at peak : {soma_score[soma_voxel]:.4f}')
        print(f'  T_combined at peak : {T_combined[soma_voxel]:.4f}')

    # ── Soma Segmentation ────────────────────────────────────────
    z0, y0, x0 = soma_voxel
    m  = SOMA_SEARCH_RADIUS + 1
    zs = slice(max(0, z0 - m), min(stack.shape[0], z0 + m))
    ys = slice(max(0, y0 - m), min(stack.shape[1], y0 + m))
    xs = slice(max(0, x0 - m), min(stack.shape[2], x0 + m))

    crop   = stack[zs, ys, xs].astype(np.float32)
    t_crop = I_OOF_raw[zs, ys, xs]

    cz = z0 - zs.start
    cy = y0 - ys.start
    cx = x0 - xs.start
    ZZ, YY, XX = np.ogrid[:crop.shape[0], :crop.shape[1], :crop.shape[2]]
    sphere = (ZZ - cz)**2 + (YY - cy)**2 + (XX - cx)**2 <= SOMA_SEARCH_RADIUS**2

    soma_candidates = sphere & (t_crop < SOMA_MAX_TUBULARITY)
    thresh          = threshold_otsu(crop[soma_candidates])
    bin_crop        = (crop > thresh) & soma_candidates
    print(f'Otsu threshold: {thresh:.4f}  ({100 * bin_crop.sum() / sphere.sum():.1f}% of sphere)')

    labeled, n_comp = sp_label(bin_crop)

    if not SOMA_HOLLOW:
        lbl = labeled[cz, cy, cx]
        if lbl == 0:
            print('WARNING: soma seed not in any component')
            soma_mask_crop = np.zeros_like(bin_crop)
        else:
            soma_mask_crop = labeled == lbl
            print(f'Connected components: {n_comp}  -> component #{lbl}')
    else:
        bridged = False
        for close_r in [3, 6, 10, 15]:
            sealed = binary_closing(bin_crop, structure=ball(close_r))
            filled = binary_fill_holes(sealed)
            labeled2, _ = sp_label(filled)
            lbl2 = labeled2[cz, cy, cx]
            if lbl2 > 0:
                soma_mask_crop = labeled2 == lbl2
                print(f'Hollow soma: sealed (close_r={close_r} vox) + fill_holes')
                bridged = True
                break
        if not bridged:
            print('WARNING: hollow soma bridging failed')
            soma_mask_crop = np.zeros_like(bin_crop)

    soma_mask_crop = binary_opening(soma_mask_crop, structure=ball(SOMA_OPEN_RADIUS_VX))
    soma_mask_crop = binary_closing(soma_mask_crop, structure=ball(SOMA_CLOSE_RADIUS_VX))
    soma_mask_crop = binary_erosion(soma_mask_crop, structure=ball(SOMA_ERODE_RADIUS_VX))
    soma_mask_crop = binary_fill_holes(soma_mask_crop)

    soma_mask = np.zeros(stack.shape, dtype=bool)
    soma_mask[zs, ys, xs] = soma_mask_crop

    soma_voxels_coords = np.argwhere(soma_mask)
    n_soma_vox         = len(soma_voxels_coords)

    if n_soma_vox == 0:
        FALLBACK_R = max(3, SOMA_OPEN_RADIUS_VX // 2)
        print(f'WARNING: soma empty after morphology. Fallback r={FALLBACK_R} vox.')
        ZZ2, YY2, XX2 = np.ogrid[:soma_mask.shape[0], :soma_mask.shape[1], :soma_mask.shape[2]]
        z0f, y0f, x0f = soma_voxel
        soma_mask = ((ZZ2-z0f)**2 + (YY2-y0f)**2 + (XX2-x0f)**2) <= FALLBACK_R**2
        soma_voxels_coords = np.argwhere(soma_mask)
        n_soma_vox = len(soma_voxels_coords)

    soma_centroid_vox = (soma_voxels_coords.mean(axis=0) if n_soma_vox > 0
                         else np.array(soma_voxel, dtype=float))
    soma_volume_um3   = n_soma_vox * voxel_iso ** 3
    _z_ctr    = int(round(soma_centroid_vox[0]))
    _z_ctr    = max(0, min(_z_ctr, soma_mask.shape[0] - 1))
    _xy_area  = soma_mask[_z_ctr].sum() * voxel_iso ** 2
    soma_equiv_r_um = float(np.sqrt(_xy_area / np.pi))

    print(f'Soma voxels      : {n_soma_vox:,}')
    print(f'Soma volume      : {soma_volume_um3:.1f} um3')
    print(f'Equiv radius     : {soma_equiv_r_um:.2f} um')
    print(f'Centroid (vox)   : {tuple(soma_centroid_vox.round(1))}')

    # ── Save soma ────────────────────────────────────────────────
    verts = np.zeros((0, 3), dtype=np.float32)
    faces = np.zeros((0, 3), dtype=np.int32)

    np.savez_compressed(str(OUT_SOMA_NPZ),
        soma_mask         = soma_mask,
        soma_centroid_vox = soma_centroid_vox,
        soma_equiv_r_um   = np.array(soma_equiv_r_um),
        voxel_iso         = np.array(voxel_iso),
        mesh_verts        = verts,
        mesh_faces        = faces,
    )
    with open(str(OUT_SOMA_JSON), 'w') as f:
        json.dump({
            'centroid_vox': soma_centroid_vox.tolist(),
            'radius_um':    float(soma_equiv_r_um),
        }, f, indent=2)
    print(f'Saved: {OUT_SOMA_NPZ}')
    print(f'Saved: {OUT_SOMA_JSON}')

    # ── Soma Anchor ──────────────────────────────────────────────
    import time as _time
    t0 = _time.time()
    coords = np.argwhere(soma_mask)
    pad = 5
    NZ2, NY2, NX2 = soma_mask.shape
    z0b = max(0,   int(coords[:,0].min()) - pad)
    y0b = max(0,   int(coords[:,1].min()) - pad)
    x0b = max(0,   int(coords[:,2].min()) - pad)
    z1b = min(NZ2, int(coords[:,0].max()) + pad + 1)
    y1b = min(NY2, int(coords[:,1].max()) + pad + 1)
    x1b = min(NX2, int(coords[:,2].max()) + pad + 1)

    mask_crop = soma_mask[z0b:z1b, y0b:y1b, x0b:x1b]
    dist_crop = distance_transform_edt(mask_crop).astype(np.float32)
    dist_crop /= (dist_crop.max() + 1e-8)
    print(f'EDT: {mask_crop.shape} crop  {_time.time()-t0:.1f}s')

    anchor_val = float(T_combined.max()) + SOMA_ANCHOR_OFFSET
    T_anchored = T_combined.copy()
    T_crop     = T_anchored[z0b:z1b, y0b:y1b, x0b:x1b]
    T_crop[mask_crop] = (dist_crop * anchor_val)[mask_crop]
    T_anchored[z0b:z1b, y0b:y1b, x0b:x1b] = T_crop
    T_anchored /= T_anchored.max()

    cz2 = int(soma_centroid_vox[0])
    cy2 = int(soma_centroid_vox[1])
    cx2 = int(soma_centroid_vox[2])
    print(f'Before: T_combined[soma centroid] = {T_combined[cz2, cy2, cx2]:.4f}')
    print(f'After : T_anchored[soma centroid] = {T_anchored[cz2, cy2, cx2]:.4f}')

    tub_rest = np.load(str(TUBULARITY_NPZ))
    np.savez(str(OUT_TUBULARITY_ANCHORED),
        T_combined   = T_anchored,
        I_OOF_raw    = I_OOF_raw,
        orient_field = tub_rest['orient_field'],
        radius_map   = tub_rest['radius_map'],
        scale_idx    = tub_rest['scale_idx'],
        radii        = tub_rest['radii'],
        voxel_iso    = np.float32(voxel_iso),
        soma_mask    = soma_mask,
    )
    del tub_rest
    print(f'Saved: {OUT_TUBULARITY_ANCHORED}')


if __name__ == '__main__':
    main()
