#!/usr/bin/env python3
"""Step 3: Riemannian FMM traceback -> neurons_auto.swc"""
import argparse
import gc
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion as _bin_erode, gaussian_filter1d
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu

warnings.filterwarnings('ignore')

# ── Config (fixed) ───────────────────────────────────────────
COST_TARGET_RATIO     = 22026  # α=10 (e^10 ≈ 22026, alpha sweep 최적값)
MIN_DIST_UM           = 15.0   # 20→15µm: seed 밀도 증가 → recall 개선
GAMMA                 = 0.99
SIGMA_PERP            = 1.0
MAX_TIPS              = 4000
MIN_RADIUS_UM         = 0.1
MERGE_DOT_MIN         = 0.99
MIN_TORTUOSITY        = 1.01
MIN_PATH_LEN_UM_FLOOR = 10.0
MAX_PATH_LEN_UM       = 600.0
SIGMA_Z_SMOOTH        = 1.0
MAX_Z_ARM_UM          = 7.0
Z_PATH_THR            = 0.65
COS_THR_SOMA          = 0.2
MIN_PRIMARY_REACH_UM  = 50.0
MAX_FALLBACK_UM       = 1.5
GAP_LEN_UM            = 2.0
GAP_T_MULT            = 3.0
MIN_T_TIP_RATIO       = 1.00   # tip detection threshold (Otsu 배수) — 낮출수록 더 많은 seed
MIN_MEAN_T_RATIO      = 0.40   # 경로 평균 T 기준 (Otsu 배수) — tip ratio와 독립
MIN_SEG_T_RATIO       = 0.12   # 경로 최소 T 기준 (Otsu 배수) — tip ratio와 독립
SMOOTH_SIGMA_VOX      = 1.0    # SWC 좌표 Gaussian smoothing sigma (voxel 단위, 계단 제거)
PRUNE_MIN_LEN_UM      = 20.0   # leaf pruning: 이 길이 미만인 branch 후보
PRUNE_MIN_MEAN_T_RATIO = 0.70  # leaf pruning: mean T < Otsu × 이 값이면 제거


def path_length_um(path, voxel):
    if len(path) < 2: return 0.0
    arr = np.array(path, dtype=np.float32)
    return float(np.linalg.norm(np.diff(arr, axis=0), axis=1).sum()) * voxel


def traceback_discrete(tip_vox, geo_dist, soma_mask, Zd, Yd, Xd,
                       border_z=1, orient_field=None, cos_thr=0.35,
                       soma_pos_vox=None, soma_r_vox=0.0,
                       max_fallback_steps=3, max_steps=200000):
    cur  = (int(tip_vox[0]), int(tip_vox[1]), int(tip_vox[2]))
    path = []
    fallback_count = 0
    for _ in range(max_steps):
        path.append(cur)
        if soma_mask[cur]: break
        z, y, x = cur
        if len(path) > 1 and (z < border_z or z >= Zd - border_z):
            break
        if orient_field is not None and soma_pos_vox is not None and soma_r_vox > 0:
            dz2 = (z - soma_pos_vox[0])**2
            dy2 = (y - soma_pos_vox[1])**2
            dx2 = (x - soma_pos_vox[2])**2
            dist_vox = (dz2 + dy2 + dx2) ** 0.5
            w = min(max((dist_vox - soma_r_vox) / soma_r_vox, 0.0), 1.0)
            eff_thr = COS_THR_SOMA + w * (cos_thr - COS_THR_SOMA)
        else:
            eff_thr = cos_thr
        cur_orient = orient_field[z, y, x] if orient_field is not None else None
        # orient norm < 0.3 → 신뢰할 수 없는 방향 → orientation constraint 비활성화
        orient_valid = (cur_orient is not None and
                        float(np.dot(cur_orient, cur_orient)) >= 0.09)
        best_val = geo_dist[cur]
        best_nb  = None
        fb_val   = geo_dist[cur]
        fb_nb    = None
        for dz in range(-1, 2):
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if dz == dy == dx == 0: continue
                    nz, ny, nx = z+dz, y+dy, x+dx
                    if not (0 <= nz < Zd and 0 <= ny < Yd and 0 <= nx < Xd): continue
                    v = geo_dist[nz, ny, nx]
                    if v < fb_val:
                        fb_val, fb_nb = v, (nz, ny, nx)
                    if v < best_val:
                        if not orient_valid:
                            # orient 없음 → pure geodesic (fallback 카운트 안 함)
                            best_val, best_nb = v, (nz, ny, nx)
                        else:
                            step = np.array([dz, dy, dx], np.float32)
                            step /= np.linalg.norm(step)
                            if abs(float(np.dot(step, cur_orient))) >= eff_thr:
                                best_val, best_nb = v, (nz, ny, nx)
        if best_nb is not None:
            cur = best_nb
            fallback_count = 0
        else:
            fallback_count += 1
            if fallback_count > max_fallback_steps or fb_nb is None:
                break
            cur = fb_nb
    return path[::-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--orig-image', type=str, default=None,
                    help='원본 이미지 TIF 경로 (원본 공간에서 노드 위치 refinement)')
    ap.add_argument('--vxy', type=float, default=None, help='원본 이미지 XY voxel 크기 (µm)')
    ap.add_argument('--vz',  type=float, default=None, help='원본 이미지 Z voxel 크기 (µm)')
    ap.add_argument('--alpha', type=float, default=None, help='FMM ALPHA 직접 지정 (기본: 자동감지)')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)

    MORSE_NPZ = out_dir / 'prep_riem.npz'
    OUT_SWC   = out_dir / 'neurons_auto.swc'

    # ── Load ────────────────────────────────────────────────────
    _meta = np.load(str(out_dir / 'preprocess_meta.npz'))
    _crop_x0 = float(_meta['crop_x0']) if 'crop_x0' in _meta else 0.0
    _crop_y0 = float(_meta['crop_y0']) if 'crop_y0' in _meta else 0.0
    _crop_z0 = float(_meta['crop_z0']) if 'crop_z0' in _meta else 0.0
    _voxel_meta = float(_meta['voxel_iso'])
    crop_offset_um = np.array([_crop_x0, _crop_y0, _crop_z0]) * _voxel_meta  # (x, y, z) µm
    if crop_offset_um.any():
        print(f'Crop offset: x={crop_offset_um[0]:.2f} y={crop_offset_um[1]:.2f} z={crop_offset_um[2]:.2f} um')

    d              = np.load(str(MORSE_NPZ))
    T_down         = d['T_down'].astype(np.float32)
    radius_down    = d['radius_down'].astype(np.float32)
    edt_down       = d['edt_down'].astype(np.float32)
    orient_down    = d['orient_down'].astype(np.float32)
    soma_mask_down = d['soma_mask_down']
    voxel_down     = float(d['voxel_down'])
    soma_vox_down  = d['soma_vox_down'].astype(np.float64)
    soma_r_um      = float(d['soma_r_um'])
    BORDER_PAD_Z   = int(d['border_pad_z']) if 'border_pad_z' in d else 1

    Zd, Yd, Xd = T_down.shape
    print(f'T_down: {T_down.shape}  voxel={voxel_down:.3f} um')
    print(f'soma_r_um: {soma_r_um:.2f} um')

    # ── Auto-detect params ───────────────────────────────────────
    T_fg     = T_down[T_down > 0.02].ravel()
    otsu_val = float(threshold_otsu(T_fg))

    # T saturation 기반 seed 간격 자동 결정
    # Z PSF saturated 이미지(neuron2/4): T_down >Otsu 비율 높음 → 20µm 유지
    # 저노이즈 이미지(s06b/s10mm): >Otsu 비율 낮음 → 15µm로 더 촘촘하게
    sat_frac = float((T_down > otsu_val).mean())
    MIN_DIST_UM_actual = MIN_DIST_UM if sat_frac > 0.05 else 15.0

    MIN_T_TIP    = round(float(np.clip(otsu_val * MIN_T_TIP_RATIO,  0.10, 0.55)), 2)

    # Alpha 자동 감지: T 분포 contrast 기반 (target_k=50 calibrated from 4-sample sweep)
    # alpha = log(50) / (2 * contrast), where contrast = median(T_fg) - median(T_bg)
    _T_fg_vals = T_down[T_down > otsu_val * 0.5].ravel()
    _T_bg_vals = T_down[(T_down > 0.01) & (T_down < otsu_val * 0.3)].ravel()
    if len(_T_fg_vals) > 200 and len(_T_bg_vals) > 200:
        _contrast  = float(np.median(_T_fg_vals) - np.median(_T_bg_vals))
        _alpha_auto = float(np.log(50.0) / (2 * max(_contrast, 0.01)))
        ALPHA = round(float(np.clip(_alpha_auto, 4.0, 10.5)), 1)  # 10.5: sweep 결과 기반 상한
    else:
        ALPHA = round(float(np.clip(np.log(COST_TARGET_RATIO), 4.0, 12.0)), 1)
    if args.alpha is not None:
        ALPHA = round(float(args.alpha), 1)  # --alpha override 우선
    MIN_DIST_VOX = int(round(MIN_DIST_UM_actual / voxel_down))
    MIN_MEAN_T   = round(float(np.clip(otsu_val * MIN_MEAN_T_RATIO, 0.08, 0.40)), 2)
    MIN_SEG_T    = round(float(np.clip(otsu_val * MIN_SEG_T_RATIO,  0.02, 0.12)), 3)
    MIN_PATH_LEN_UM = round(float(max(MIN_PATH_LEN_UM_FLOOR, soma_r_um * 0.5)), 1)
    MERGE_DIST_UM   = round(float(max(4.0, soma_r_um * 0.3)), 1)

    print('=' * 56)
    print('  Auto-detected parameters')
    print('=' * 56)
    print(f'  ALPHA        = {ALPHA}  (log({COST_TARGET_RATIO})  cost@T=1={np.exp(-ALPHA):.4f})')
    print(f'  MIN_T_TIP    = {MIN_T_TIP}  (Otsu={otsu_val:.3f})')
    print(f'  sat_frac     = {sat_frac:.3f}  → MIN_DIST_UM={MIN_DIST_UM_actual:.0f}')
    print(f'  MIN_DIST_VOX = {MIN_DIST_VOX}  ({MIN_DIST_UM_actual} um)'
          f'  MIN_T_TIP_RATIO = {MIN_T_TIP_RATIO}')
    print(f'  MIN_MEAN_T   = {MIN_MEAN_T}')
    print(f'  MIN_SEG_T    = {MIN_SEG_T}')
    print(f'  MIN_PATH_LEN = {MIN_PATH_LEN_UM} um')
    print(f'  MERGE_DIST   = {MERGE_DIST_UM} um')
    print('=' * 56)

    # ── FileHFM setup ────────────────────────────────────────────
    import agd
    from agd import Eikonal, Metrics

    script_dir = Path(__file__).parent
    hfm_txt    = script_dir / 'FileHFM_binary_dir.txt'
    if not hfm_txt.exists():
        # fallback: tracer_aniso directory
        hfm_txt = script_dir.parent.parent.parent / 'tracer_aniso' / 'FileHFM_binary_dir.txt'
    BIN_DIR = hfm_txt.read_text().strip()
    agd.Eikonal.LibraryCall.binary_dir['FileHFM'] = BIN_DIR
    print(f'FileHFM: {BIN_DIR}')

    # ── Metric tensor M(x) ──────────────────────────────────────
    t0 = time.time()
    T64     = T_down.astype(np.float64)
    cost2   = np.exp(-ALPHA * T64) ** 2
    sigma_p = SIGMA_PERP * (1.0 - GAMMA * T64)
    v       = orient_down.astype(np.float64)

    M = np.zeros((3, 3, Zd, Yd, Xd), dtype=np.float64)
    for i in range(3):
        M[i, i] += SIGMA_PERP
        for j in range(3):
            M[i, j] += (sigma_p - SIGMA_PERP) * v[..., i] * v[..., j]
    M *= cost2[np.newaxis, np.newaxis]
    del cost2, v, sigma_p, T64; gc.collect()

    sp_min = SIGMA_PERP * (1.0 - GAMMA * float(T_down.max()))
    print(f'M built in {time.time()-t0:.1f}s  mem={M.nbytes/1e9:.2f} GB')
    print(f'Anisotropy ratio: {SIGMA_PERP/sp_min:.1f}:1')

    # ── Riemannian FMM ───────────────────────────────────────────
    t0 = time.time()
    soma_seeds = np.argwhere(soma_mask_down).astype(np.float64)
    print(f'Seeds: {len(soma_seeds):,} soma voxels')

    metric = Metrics.Riemann(M)
    del M; gc.collect()

    hfm = Eikonal.dictIn({
        'model':        'Riemann3',
        'dims':         np.array([Zd, Yd, Xd]),
        'gridScale':    1.0,
        'metric':       metric,
        'seeds':        soma_seeds,
        'exportValues': True,
        'verbosity':    1,
    })
    out           = hfm.Run()
    geodesic_dist = out['values'].astype(np.float32)
    if geodesic_dist.shape != T_down.shape:
        raise RuntimeError(
            f'FMM output shape {geodesic_dist.shape} != T_down shape {T_down.shape}. '
            f'FMM may have crashed (OOM). Try reducing volume or increasing DOWNSAMPLE.')
    print(f'FMM done in {time.time()-t0:.1f}s')
    print(f'Reachable: {np.isfinite(geodesic_dist).sum():,}')

    # ── Tip detection: Hybrid (T peaks + Geo peaks) ─────────────
    # Fix A+B hybrid:
    # Group T: T local max → reliable traceback (밝은 위치)
    # Group G: geodesic local max (T mask) → true branch terminal (distal coverage)
    # Merge: geo 우선, T peaks 중 geo와 MIN_DIST 내 겹치는 것 제거 → union
    geo_finite = geodesic_dist.copy()
    geo_finite[~np.isfinite(geo_finite)] = 0

    T_for_tips = gaussian_filter1d(T_down, sigma=SIGMA_Z_SMOOTH, axis=0)

    # Group T: 기존 T-based peaks
    _pT = peak_local_max(T_for_tips, min_distance=MIN_DIST_VOX,
                         threshold_abs=MIN_T_TIP, exclude_border=False)
    coords_T = _pT if _pT.dtype != bool else np.argwhere(_pT)

    # Group G: geodesic local max (T >= MIN_T_TIP 마스크 적용)
    _geo_masked = np.where(
        (T_for_tips >= MIN_T_TIP) & np.isfinite(geodesic_dist) & ~soma_mask_down,
        geodesic_dist.astype(np.float32), 0.0)
    _pG = peak_local_max(_geo_masked, min_distance=MIN_DIST_VOX,
                         threshold_abs=1e-6, exclude_border=False)
    coords_G = _pG if _pG.dtype != bool else np.argwhere(_pG)

    # Merge: geo 우선. T peaks 중 geo peak과 MIN_DIST_VOX 이내인 것 제거
    if len(coords_G) > 0 and len(coords_T) > 0:
        _d, _ = cKDTree(coords_G.astype(float)).query(coords_T.astype(float))
        coords_T_extra = coords_T[_d >= MIN_DIST_VOX]
        tip_coords_all = np.vstack([coords_G, coords_T_extra])
    elif len(coords_G) > 0:
        tip_coords_all = coords_G
    else:
        tip_coords_all = coords_T

    tip_vals    = T_down[tip_coords_all[:,0], tip_coords_all[:,1], tip_coords_all[:,2]]
    tip_geo_all = geodesic_dist[tip_coords_all[:,0], tip_coords_all[:,1], tip_coords_all[:,2]]
    reachable   = np.isfinite(tip_geo_all)
    tip_coords_r = tip_coords_all[reachable]
    tip_vals_r   = tip_vals[reachable]
    tip_geo_r    = tip_geo_all[reachable]

    sort_idx     = np.argsort(tip_geo_r)[::-1]
    tip_coords_s = tip_coords_r[sort_idx][:MAX_TIPS]
    tip_vals_s   = tip_vals_r[sort_idx][:MAX_TIPS]
    tip_geo_s    = tip_geo_r[sort_idx][:MAX_TIPS]

    # ── Geo/Euclidean ratio filter: FP seeds via winding paths 제거 ──
    # 인접 구조물 terminal은 geodesic >> euclidean (dim 구간 경유)
    # ratio = geo_cost / euclid_vox → outlier seeds 제거
    _euclid_vox = np.linalg.norm(
        tip_coords_s.astype(float) - soma_vox_down.astype(float), axis=1) + 1e-6
    _geo_ratio  = tip_geo_s / _euclid_vox
    _ratio_thr  = np.median(_geo_ratio) * 4.0   # median × 4 (adaptive per-sample)
    _keep_ge    = _geo_ratio <= _ratio_thr
    n_ge_filtered = int((~_keep_ge).sum())
    if n_ge_filtered > 0:
        tip_coords_s = tip_coords_s[_keep_ge]
        tip_vals_s   = tip_vals_s[_keep_ge]
        tip_geo_s    = tip_geo_s[_keep_ge]

    print(f'Tips detected: {len(tip_coords_all):,}'
          f'  (G={len(coords_G)}, T_extra={len(tip_coords_all)-len(coords_G)}, reachable: {reachable.sum():,})')
    print(f'Tips selected: {len(tip_coords_s)}'
          f'  geo={tip_geo_s[-1]:.1f}-{tip_geo_s[0]:.1f}'
          f'  T={tip_vals_s.min():.3f}-{tip_vals_s.max():.3f}'
          f'  geo/euclid_filtered={n_ge_filtered}')

    # ── Traceback ────────────────────────────────────────────────
    t0 = time.time()

    border_mask = ((tip_coords_s[:,0] >= BORDER_PAD_Z) &
                   (tip_coords_s[:,0] <  Zd - BORDER_PAD_Z))
    n_border_removed = (~border_mask).sum()
    tip_coords_s = tip_coords_s[border_mask]
    tip_vals_s   = tip_vals_s[border_mask]
    if n_border_removed:
        print(f'Border tips removed: {n_border_removed}')

    soma_r_vox         = soma_r_um / voxel_down
    MAX_FALLBACK_STEPS = max(1, int(MAX_FALLBACK_UM / voxel_down))
    GAP_LEN_VOX        = max(1, int(GAP_LEN_UM / voxel_down))
    _GAP_THR           = MIN_SEG_T * GAP_T_MULT
    print(f'Fallback: {MAX_FALLBACK_STEPS} steps  Gap: {GAP_LEN_UM} um ({GAP_LEN_VOX} vox)  thr={_GAP_THR:.4f}')

    all_paths = {}
    n_short = n_trimmed = n_too_short_trim = n_straight = n_too_long = 0
    n_z_path = n_no_soma = n_gap_trim = 0

    for i, tip in enumerate(tip_coords_s):
        key = (int(tip[0]), int(tip[1]), int(tip[2]))
        if not np.isfinite(geodesic_dist[key]): continue
        path = traceback_discrete(key, geodesic_dist, soma_mask_down,
                                  Zd, Yd, Xd, border_z=BORDER_PAD_Z,
                                  orient_field=orient_down, cos_thr=0.60,
                                  soma_pos_vox=soma_vox_down,
                                  soma_r_vox=soma_r_vox,
                                  max_fallback_steps=MAX_FALLBACK_STEPS)

        if not soma_mask_down[path[0]]:
            n_no_soma += 1; continue
        if path_length_um(path, voxel_down) < MIN_PATH_LEN_UM:
            n_short += 1; continue

        # Full-path T-gap filter
        _t_full = np.array([T_down[k] for k in path], dtype=np.float32)
        _run = 0; _gap_at = None
        for _ki, _tv in enumerate(_t_full):
            if _tv < _GAP_THR:
                _run += 1
                if _run >= GAP_LEN_VOX and _gap_at is None:
                    _gap_at = _ki - _run + 1
            else:
                _run = 0
        if _gap_at is not None and _gap_at > 0:
            path = path[:_gap_at]
            if path_length_um(path, voxel_down) < MIN_PATH_LEN_UM:
                n_gap_trim += 1; continue
            n_gap_trim += 1

        # Z-path filter
        _n_tip_vox = max(2, int(MAX_Z_ARM_UM / voxel_down))
        _tip_seg   = path[-_n_tip_vox:]
        if len(_tip_seg) >= 2:
            _diffs = np.diff(np.array(_tip_seg, dtype=np.float32), axis=0)
            _total = float(np.linalg.norm(_diffs, axis=1).sum())
            if _total > 0 and float(np.abs(_diffs[:, 0]).sum()) / _total > Z_PATH_THR:
                n_z_path += 1; continue

        t_vals  = np.array([T_down[k] for k in path], dtype=np.float32)
        trimmed = False
        half    = len(path) // 2

        bad = np.where(t_vals[half:] < MIN_SEG_T)[0]
        if len(bad) > 0:
            path = path[:half + bad[0]]; t_vals = t_vals[:half + bad[0]]; trimmed = True
            if path_length_um(path, voxel_down) < MIN_PATH_LEN_UM:
                n_too_short_trim += 1; continue

        half = len(path) // 2   # recalculate after first trim
        if len(t_vals) > half:
            rm = np.cumsum(t_vals[half:]) / np.arange(1, len(t_vals)-half+1)
            bm = np.where(rm < MIN_MEAN_T)[0]
            if len(bm) > 0:
                path = path[:half + bm[0]]; t_vals = t_vals[:half + bm[0]]; trimmed = True
                if path_length_um(path, voxel_down) < MIN_PATH_LEN_UM:
                    n_too_short_trim += 1; continue

        if trimmed: n_trimmed += 1

        path_len_um = path_length_um(path, voxel_down)
        euclid_um   = float(np.linalg.norm(
            np.array(path[-1], dtype=np.float32) -
            np.array(path[0],  dtype=np.float32))) * voxel_down
        if path_len_um / (euclid_um + 1e-8) < MIN_TORTUOSITY:
            n_straight += 1; continue
        if path_len_um > MAX_PATH_LEN_UM:
            n_too_long += 1; continue

        all_paths[i] = path

    print(f'Traceback: {time.time()-t0:.1f}s')
    print(f'Paths kept: {len(all_paths)}')
    print(f'  trimmed:{n_trimmed}  no_soma:{n_no_soma}  short:{n_short}'
          f'  gap_trim:{n_gap_trim}  too_short_trim:{n_too_short_trim}')
    print(f'  straight:{n_straight}  too_long:{n_too_long}  z_path:{n_z_path}')


    # ── Path coordinate smoothing (계단 제거) ────────────────────
    # voxel 좌표는 node_id_map dedup에 그대로 사용, µm 좌표만 smooth
    smooth_xyz = {}  # voxel key → (x_um, y_um, z_um) smoothed
    for path in all_paths.values():
        arr = np.array(path, dtype=np.float32)          # (N,3): z,y,x
        if len(arr) >= 4 and SMOOTH_SIGMA_VOX > 0:
            s = np.stack([gaussian_filter1d(arr[:, i], sigma=SMOOTH_SIGMA_VOX)
                          for i in range(3)], axis=1)
        else:
            s = arr
        for idx, key in enumerate(path):
            if key not in smooth_xyz:
                smooth_xyz[key] = (float(s[idx, 2] * voxel_down),
                                   float(s[idx, 1] * voxel_down),
                                   float(s[idx, 0] * voxel_down))

    # ── Tree construction ────────────────────────────────────────
    _soma_surface = soma_mask_down & ~_bin_erode(soma_mask_down, iterations=1)
    _surf_coords  = np.argwhere(_soma_surface).astype(np.float32)
    sc = _surf_coords.mean(axis=0) if len(_surf_coords) > 0 else soma_vox_down

    soma_x = float(sc[2]) * voxel_down
    soma_y = float(sc[1]) * voxel_down
    soma_z = float(sc[0]) * voxel_down

    swc_rows    = [(1, 1, soma_x, soma_y, soma_z, soma_r_um, -1)]
    node_id_map = {}
    next_id     = 2

    sorted_keys = sorted(all_paths.keys(),
        key=lambda k: float(geo_finite[tip_coords_s[k][0],
                                       tip_coords_s[k][1],
                                       tip_coords_s[k][2]]),
        reverse=False)

    for bi in sorted_keys:
        prev_swc_id  = 1
        soma_surf_key = None
        for key in all_paths[bi]:
            if soma_mask_down[key]:
                if key in node_id_map:
                    prev_swc_id = node_id_map[key]
                else:
                    node_id_map[key] = 1
                    prev_swc_id = 1
                soma_surf_key = key
                continue
            if key in node_id_map:
                prev_swc_id   = node_id_map[key]
                soma_surf_key = None
                continue
            z, y, x = key
            r = max(float(edt_down[z, y, x]), MIN_RADIUS_UM)
            cx, cy, cz = smooth_xyz.get(key, (x*voxel_down, y*voxel_down, z*voxel_down))
            if soma_surf_key is not None:
                sx, sy, sz = smooth_xyz.get(soma_surf_key,
                    (soma_surf_key[2]*voxel_down,
                     soma_surf_key[1]*voxel_down,
                     soma_surf_key[0]*voxel_down))
                swc_rows.append((next_id, 3, sx, sy, sz, r, prev_swc_id))
                soma_surf_key = None
            else:
                swc_rows.append((next_id, 3, cx, cy, cz, r, prev_swc_id))
            node_id_map[key] = next_id; prev_swc_id = next_id; next_id += 1

    print(f'SWC nodes: {next_id-1:,}  (branches: {len(all_paths)})')
    swc_rows_dict = {r[0]: r for r in swc_rows}

    # ── Primary merge ────────────────────────────────────────────
    primary_nodes = {nid: swc_rows_dict[nid]
                     for nid in [r[0] for r in swc_rows if r[6] == 1]}

    def branch_dir(nid):
        n = swc_rows_dict[nid]; sc_ = swc_rows_dict[1]
        v = np.array([n[2]-sc_[2], n[3]-sc_[3], n[4]-sc_[4]])
        return v / (np.linalg.norm(v) + 1e-8)

    def node_pos(nid):
        n = swc_rows_dict[nid]; return np.array([n[2], n[3], n[4]])

    pids   = list(primary_nodes.keys())
    p_pos  = np.array([node_pos(p) for p in pids])
    p_dirs = np.array([branch_dir(p) for p in pids])

    parent_uf = {p: p for p in pids}
    def find(x):
        while parent_uf[x] != x: parent_uf[x] = parent_uf[parent_uf[x]]; x = parent_uf[x]
        return x
    def union(a, b): parent_uf[find(a)] = find(b)

    if len(pids) > 1:
        for i, j in cKDTree(p_pos).query_pairs(MERGE_DIST_UM):
            cos = float(p_dirs[i] @ p_dirs[j])
            if cos >= MERGE_DOT_MIN:
                print(f'  MERGE: node {pids[i]} & {pids[j]}  cos={cos:.3f}')
                union(pids[i], pids[j])

    groups = defaultdict(list)
    for p in pids: groups[find(p)].append(p)

    def subtree_tip_count(nid):
        count, stack = 0, [nid]
        while stack:
            cur = stack.pop()
            kids = [r[0] for r in swc_rows if r[6] == cur]
            if not kids: count += 1
            stack.extend(kids)
        return count

    to_remove, merged_count = set(), 0
    for rep, members in groups.items():
        if len(members) == 1: continue
        best = max(members, key=subtree_tip_count)
        for m in members:
            if m != best: to_remove.add(m)
        merged_count += len(members) - 1

    def get_subtree_ids(root):
        result, stack = set(), [root]
        while stack:
            cur = stack.pop(); result.add(cur)
            stack.extend([r[0] for r in swc_rows if r[6] == cur])
        return result

    remove_ids = set()
    for nid in to_remove: remove_ids |= get_subtree_ids(nid)
    swc_rows = [r for r in swc_rows if r[0] not in remove_ids]

    print(f'Primary before merge: {len(pids)}  merged: {merged_count}')
    print(f'SWC nodes after merge: {len(swc_rows):,}')

    # ── Primary reach filter ─────────────────────────────────────
    _swc_dict = {r[0]: r for r in swc_rows}
    _ch = defaultdict(list)
    for r in swc_rows:
        if r[6] != -1: _ch[r[6]].append(r[0])

    def _max_reach(pid):
        tips, stack = [], [pid]
        while stack:
            cur = stack.pop()
            kids = _ch.get(cur, [])
            if not kids: tips.append(cur)
            stack.extend(kids)
        best = 0.0
        for tip in tips:
            length, cur = 0.0, tip
            while cur != -1 and cur in _swc_dict:
                par = _swc_dict[cur][6]
                if par == -1 or par not in _swc_dict: break
                p1 = np.array(_swc_dict[cur][2:5])
                p2 = np.array(_swc_dict[par][2:5])
                length += np.linalg.norm(p1 - p2)
                cur = par
            best = max(best, length)
        return best

    def _get_subtree(root):
        result, stack = set(), [root]
        while stack:
            cur = stack.pop(); result.add(cur)
            stack.extend(_ch.get(cur, []))
        return result

    primaries_final = [r[0] for r in swc_rows if r[6] == 1]
    reach_remove    = set()
    for pid in primaries_final:
        reach = _max_reach(pid)
        if reach < MIN_PRIMARY_REACH_UM:
            reach_remove.add(pid)
            print(f'  Removing primary {pid}: max_reach={reach:.1f} um')

    if reach_remove:
        reach_remove_ids = set()
        for pid in reach_remove:
            reach_remove_ids |= _get_subtree(pid)
        swc_rows = [r for r in swc_rows if r[0] not in reach_remove_ids]
        print(f'Reach filter: removed {len(reach_remove)} primaries ({len(reach_remove_ids)} nodes)')
    else:
        print('Reach filter: nothing removed')
    print(f'SWC nodes before pruning: {len(swc_rows):,}')

    # ── Leaf pruning ──────────────────────────────────────────────
    _root_id    = next((r[0] for r in swc_rows if r[6] == -1), None)
    prune_t_thr = otsu_val * PRUNE_MIN_MEAN_T_RATIO
    total_pruned_nodes = 0

    for _pass in range(30):
        _sd  = {r[0]: r for r in swc_rows}
        _chp = defaultdict(list)
        for r in swc_rows:
            if r[6] != -1:
                _chp[r[6]].append(r[0])

        leaves = [r[0] for r in swc_rows
                  if r[0] != _root_id and not _chp.get(r[0])]

        prune_ids = set()
        for leaf in leaves:
            branch = []
            cur = leaf
            while cur != _root_id and cur in _sd:
                branch.append(cur)
                par = _sd[cur][6]
                if par == -1 or par not in _sd or par == _root_id:
                    break
                if len(_chp.get(par, [])) > 1:
                    break  # branch point: 여기서 멈춤
                cur = par

            if len(branch) < 2:
                continue

            blen = sum(
                np.sqrt((_sd[branch[i]][2] - _sd[branch[i+1]][2])**2 +
                        (_sd[branch[i]][3] - _sd[branch[i+1]][3])**2 +
                        (_sd[branch[i]][4] - _sd[branch[i+1]][4])**2)
                for i in range(len(branch) - 1)
            )
            if blen >= PRUNE_MIN_LEN_UM:
                continue

            t_sum = 0.0
            for nid in branch:
                r = _sd[nid]
                iz = int(np.clip(round(r[4] / voxel_down), 0, Zd - 1))
                iy = int(np.clip(round(r[3] / voxel_down), 0, Yd - 1))
                ix = int(np.clip(round(r[2] / voxel_down), 0, Xd - 1))
                t_sum += float(T_down[iz, iy, ix])
            if (t_sum / len(branch)) < prune_t_thr:
                prune_ids.update(branch)

        if not prune_ids:
            break
        swc_rows = [r for r in swc_rows if r[0] not in prune_ids]
        total_pruned_nodes += len(prune_ids)

    _chp_final = defaultdict(list)
    for r in swc_rows:
        if r[6] != -1:
            _chp_final[r[6]].append(r[0])
    n_tips_final = sum(1 for r in swc_rows
                       if r[0] != _root_id and not _chp_final.get(r[0]))
    print(f'Leaf pruning: {total_pruned_nodes} nodes removed'
          f'  (len<{PRUNE_MIN_LEN_UM}µm & T<{prune_t_thr:.3f} [{PRUNE_MIN_MEAN_T_RATIO}×Otsu])')
    print(f'SWC nodes after pruning: {len(swc_rows):,}  tips: {n_tips_final}')

    # ── 원본 이미지 공간에서 노드 위치 refinement ────────────────────
    # gold standard가 트레이싱된 원본 이미지 공간에서 로컬 강도 최대값으로 이동
    # → 위치 정확도 개선 → 2µm threshold F1 향상
    if args.orig_image and args.vxy and Path(args.orig_image).exists():
        import tifffile as _tf
        from scipy import ndimage as _ndi
        from scipy.ndimage import map_coordinates as _map_coords
        print(f'원본 이미지 refinement: {args.orig_image}')
        _img = _tf.imread(args.orig_image).astype(np.float32)
        if _img.ndim == 4:
            _img = _img[0] if _img.shape[0] < _img.shape[1] else _img[:, 0]
        _Zd, _Yd, _Xd = _img.shape
        _vxy = float(args.vxy); _vz = float(args.vz) if args.vz else 1.0

        # ── Method 2: 수직 단면 최대 강도 스냅 (먼저 실행) ─────────────
        # cross-section max snap → XY 정밀 위치, Z는 노드별 독립 snap
        # M1 COM이 이후에 Z 진동을 자연스럽게 안정화
        _SRCH     = 2.0   # µm 기준 탐색 반경
        _SRCH_MAX_PX = 7  # XY 픽셀 상한: 고해상도(조밀) 샘플에서 인접 구조물 snap 방지
        _GRID = 11
        _srch_vxy = min(_SRCH / _vxy, float(_SRCH_MAX_PX))  # px (자동 적응)
        _srch_vz  = _SRCH / _vz                              # px (µm 기준 고정)
        _swc_dict = {r[0]: r for r in swc_rows}
        _ch2 = defaultdict(list)
        for r in swc_rows:
            if r[6] != -1: _ch2[r[6]].append(r[0])

        refined2 = []; n_ref2 = 0
        for row in swc_rows:
            if row[1] == 1: refined2.append(row); continue
            x, y, z = row[2], row[3], row[4]
            neighbors = []
            if row[6] != -1 and row[6] in _swc_dict:
                p = _swc_dict[row[6]]
                neighbors.append(np.array([p[2]-x, p[3]-y, p[4]-z]))
            for cid in _ch2.get(row[0], []):
                c = _swc_dict[cid]
                neighbors.append(np.array([c[2]-x, c[3]-y, c[4]-z]))
            if not neighbors: refined2.append(row); continue
            d = np.zeros(3)
            for v in neighbors:
                nrm = np.linalg.norm(v)
                if nrm > 1e-6: d += v / nrm
            nrm = np.linalg.norm(d)
            if nrm < 1e-6: refined2.append(row); continue
            d /= nrm
            arb = np.array([1,0,0]) if abs(d[0]) < 0.9 else np.array([0,1,0])
            u = np.cross(d, arb); u /= np.linalg.norm(u)
            v2 = np.cross(d, u)
            ts = np.linspace(-1, 1, _GRID)
            gi = (x/_vxy) + (ts[:,None]*u[0]*_srch_vxy + ts[None,:]*v2[0]*_srch_vxy)
            gj = (y/_vxy) + (ts[:,None]*u[1]*_srch_vxy + ts[None,:]*v2[1]*_srch_vxy)
            gk = (z/_vz)  + (ts[:,None]*u[2]*_srch_vz  + ts[None,:]*v2[2]*_srch_vz)
            gi = np.clip(gi, 0, _Xd-1); gj = np.clip(gj, 0, _Yd-1); gk = np.clip(gk, 0, _Zd-1)
            vals = _map_coords(_img, [gk.ravel(), gj.ravel(), gi.ravel()],
                               order=1, mode='nearest').reshape(_GRID, _GRID)
            bi, bj = np.unravel_index(vals.argmax(), vals.shape)
            t1, t2 = ts[bi], ts[bj]
            dx = (t1*u[0] + t2*v2[0]) * _srch_vxy * _vxy
            dy = (t1*u[1] + t2*v2[1]) * _srch_vxy * _vxy
            dz = (t1*u[2] + t2*v2[2]) * _srch_vz  * _vz
            if abs(dx)+abs(dy)+abs(dz) > 0.05: n_ref2 += 1
            refined2.append((row[0], row[1], x+dx, y+dy, z+dz, row[5], row[6]))

        swc_rows = refined2
        print(f'  → Method2: {n_ref2}/{len(swc_rows)} 노드 위치 조정 완료'
              f' (_SRCH_XY={_srch_vxy:.1f}px={_srch_vxy*_vxy:.2f}µm, Z={_srch_vz:.1f}px={_srch_vz*_vz:.2f}µm)')

        # ── Method 1: 3D COM refinement (나중에 실행 → M2 Z 진동 안정화) ─
        # M2의 per-node 독립 snap 이후 3D COM으로 Z 위치 자연스럽게 정규화
        _NH = 5  # ±5 voxel 탐색
        _ox, _oy, _oz = crop_offset_um
        refined_swc = []
        n_ref = 0
        for row in swc_rows:
            if row[1] == 1:
                refined_swc.append(row); continue
            x, y, z = row[2], row[3], row[4]
            xi = int(np.clip(round(x / _vxy), 0, _Xd - 1))
            yi = int(np.clip(round(y / _vxy), 0, _Yd - 1))
            zi = int(np.clip(round(z / _vz),  0, _Zd - 1))
            x0,x1 = max(0,xi-_NH), min(_Xd,xi+_NH+1)
            y0,y1 = max(0,yi-_NH), min(_Yd,yi+_NH+1)
            z0,z1 = max(0,zi-_NH), min(_Zd,zi+_NH+1)
            patch = _img[z0:z1, y0:y1, x0:x1].astype(np.float64)
            if patch.max() < 1e-3:
                refined_swc.append(row); continue
            com = _ndi.center_of_mass(patch ** 2)
            dx = float(np.clip(com[2] - (xi - x0), -_NH, _NH))
            dy = float(np.clip(com[1] - (yi - y0), -_NH, _NH))
            dz = float(np.clip(com[0] - (zi - z0), -_NH, _NH))
            if abs(dx) + abs(dy) + abs(dz) > 0.05:
                n_ref += 1
            refined_swc.append((row[0], row[1],
                                 x + dx*_vxy, y + dy*_vxy, z + dz*_vz,
                                 row[5], row[6]))
        swc_rows = refined_swc
        print(f'  → Method1: {n_ref}/{len(swc_rows)} 노드 위치 조정 완료')

    # ── Dense resampling + M2 snap (이미지 기반 실제 노드 감지) ──
    # 선형 보간(인위적)이 아닌, 각 보간 위치에서 M2 cross-section snap으로
    # 실제 이미지 intensity maximum에 노드 배치 → genuine detection
    DENSE_STEP_UM = 0.5
    _sd_d = {r[0]: r for r in swc_rows}
    _new_id_d = max(r[0] for r in swc_rows) + 1
    dense_rows = []; n_dense = 0
    for row in swc_rows:
        pid = row[6]
        if pid == -1 or pid not in _sd_d:
            dense_rows.append(row); continue
        par = _sd_d[pid]
        dx=row[2]-par[2]; dy=row[3]-par[3]; dz=row[4]-par[4]
        dist=(dx*dx+dy*dy+dz*dz)**0.5
        if dist <= DENSE_STEP_UM * 1.5:
            dense_rows.append(row); continue
        d_dir = np.array([dx,dy,dz]); d_nrm = np.linalg.norm(d_dir)
        if d_nrm < 1e-6:
            dense_rows.append(row); continue
        d_dir /= d_nrm
        arb = np.array([1,0,0]) if abs(d_dir[0]) < 0.9 else np.array([0,1,0])
        u_d = np.cross(d_dir, arb); u_d /= np.linalg.norm(u_d)
        v_d = np.cross(d_dir, u_d)
        n_seg = max(1, int(dist / DENSE_STEP_UM))
        prev_pid_d = pid
        ts_d = np.linspace(-1, 1, _GRID)
        for i in range(1, n_seg):
            t = i / n_seg
            xi=par[2]+t*dx; yi=par[3]+t*dy; zi=par[4]+t*dz
            ri=par[5]+t*(row[5]-par[5])
            gi=(xi/_vxy)+(ts_d[:,None]*u_d[0]*_srch_vxy+ts_d[None,:]*v_d[0]*_srch_vxy)
            gj=(yi/_vxy)+(ts_d[:,None]*u_d[1]*_srch_vxy+ts_d[None,:]*v_d[1]*_srch_vxy)
            gk=(zi/_vz) +(ts_d[:,None]*u_d[2]*_srch_vz +ts_d[None,:]*v_d[2]*_srch_vz)
            gi=np.clip(gi,0,_Xd-1); gj=np.clip(gj,0,_Yd-1); gk=np.clip(gk,0,_Zd-1)
            vv=_map_coords(_img,[gk.ravel(),gj.ravel(),gi.ravel()],
                           order=1,mode='nearest').reshape(_GRID,_GRID)
            bi_d,bj_d=np.unravel_index(vv.argmax(),vv.shape)
            t1d,t2d=ts_d[bi_d],ts_d[bj_d]
            sx=(t1d*u_d[0]+t2d*v_d[0])*_srch_vxy*_vxy
            sy=(t1d*u_d[1]+t2d*v_d[1])*_srch_vxy*_vxy
            sz=(t1d*u_d[2]+t2d*v_d[2])*_srch_vz *_vz
            dense_rows.append((_new_id_d,row[1],xi+sx,yi+sy,zi+sz,ri,prev_pid_d))
            prev_pid_d=_new_id_d; _new_id_d+=1; n_dense+=1
        dense_rows.append((row[0],row[1],row[2],row[3],row[4],row[5],prev_pid_d))
    swc_rows = dense_rows
    print(f'  → Dense+M2: {n_dense} nodes added (total {len(swc_rows)})')

    # ── Post-Dense smoothing: M2 per-node snap 진동 제거 ─────────
    # Dense+M2는 각 노드를 독립 snap → 고주파 지그재그 발생
    # σ=1.5 Gaussian으로 진동 제거, 진짜 centerline trend는 보존
    _SMOOTH_DENSE = 1.5
    _sd_ps = {r[0]: r for r in swc_rows}
    _ch_ps = defaultdict(list)
    for r in swc_rows:
        if r[6] != -1: _ch_ps[r[6]].append(r[0])
    seg_starts_ps = [r[0] for r in swc_rows
                     if r[6] != -1 and (
                         _sd_ps[r[6]][1] == 1 or
                         len(_ch_ps.get(r[6], [])) >= 2)]
    smooth_ps = {}
    for start in seg_starts_ps:
        seg = []
        cur = start
        while cur is not None:
            seg.append(cur)
            kids = _ch_ps.get(cur, [])
            cur = kids[0] if len(kids) == 1 else None
        if len(seg) < 4:
            continue
        arr = np.array([[_sd_ps[n][2], _sd_ps[n][3], _sd_ps[n][4]]
                        for n in seg], dtype=np.float32)
        s = np.stack([gaussian_filter1d(arr[:, i], sigma=_SMOOTH_DENSE)
                      for i in range(3)], axis=1)
        for idx, n in enumerate(seg):
            if n not in smooth_ps and _sd_ps[n][1] != 1:
                smooth_ps[n] = (float(s[idx, 0]), float(s[idx, 1]), float(s[idx, 2]))
    swc_rows = [
        (r[0], r[1],
         smooth_ps[r[0]][0] if r[0] in smooth_ps else r[2],
         smooth_ps[r[0]][1] if r[0] in smooth_ps else r[3],
         smooth_ps[r[0]][2] if r[0] in smooth_ps else r[4],
         r[5], r[6])
        for r in swc_rows
    ]
    print(f'  → Post-dense smooth: {len(smooth_ps)}/{len(swc_rows)} nodes (σ={_SMOOTH_DENSE})')

    # ── Save SWC ─────────────────────────────────────────────────
    header = [
        f'# tracer_aniso AUTO — Riemannian FMM',
        f'# ALPHA={ALPHA}  MIN_T_TIP={MIN_T_TIP}  GAMMA={GAMMA}  SIGMA_PERP={SIGMA_PERP}',
        f'# MIN_SEG_T={MIN_SEG_T}  MIN_MEAN_T={MIN_MEAN_T}  MIN_TORTUOSITY={MIN_TORTUOSITY}',
        f'# PRUNE_LEN={PRUNE_MIN_LEN_UM}um  PRUNE_T_RATIO={PRUNE_MIN_MEAN_T_RATIO}  PRUNE_T={prune_t_thr:.3f}',
        f'# seed_tips={len(tip_coords_s)}  paths={len(all_paths)}  tips={n_tips_final}  nodes={len(swc_rows)}',
        '# id type x y z radius parent',
    ]
    ox, oy, oz = crop_offset_um
    lines = header + [
        f'{r[0]} {r[1]} {r[2]+ox:.4f} {r[3]+oy:.4f} {r[4]+oz:.4f} {r[5]:.4f} {r[6]}'
        for r in swc_rows
    ]
    with open(str(OUT_SWC), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'Saved: {OUT_SWC}')


if __name__ == '__main__':
    main()
