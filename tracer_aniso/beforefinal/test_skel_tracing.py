#!/usr/bin/env python3
"""
test_skel_tracing.py  —  소마 seeded 스켈레톤 트레이싱 테스트

두 가지 방법 비교:
  A) 현재 방식 개선: peak_local_max(geodesic_dist) 로 tip 탐지
     → T 동점 문제 해결, true distal endpoint 선택
  B) 완전 새 방식: T_down 마스크 → 3D skeletonize → 소마 rooted SWC

Usage:
    python test_skel_tracing.py          # 둘 다 실행
    python test_skel_tracing.py --method A
    python test_skel_tracing.py --method B
"""

import argparse, os, time
import numpy as np
from skimage.morphology import skeletonize
from skimage.measure import label as cc_label
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from scipy.ndimage import distance_transform_edt
import tifffile

HERE = os.path.dirname(os.path.abspath(__file__))
FN1  = os.path.join(HERE, 'output', 'FN1_01')

parser = argparse.ArgumentParser()
parser.add_argument('--method', choices=['A','B','both'], default='both')
parser.add_argument('--t-thr-factor', type=float, default=0.5,
                    help='T threshold = Otsu * factor  (method B, default 0.5)')
args = parser.parse_args()

# ── Load ─────────────────────────────────────────────────────
print('Loading prep data...', flush=True)
d          = np.load(os.path.join(FN1, 'prep_riem.npz'), allow_pickle=True)
T_down     = d['T_down'].astype(np.float32)
edt_down   = d['edt_down'].astype(np.float32)
orient_down= d['orient_down'].astype(np.float32)
soma_mask  = d['soma_mask_down']
voxel      = float(d['voxel_down'])
soma_vox   = d['soma_vox_down']
soma_r_um  = float(d['soma_r_um'])
Zd,Yd,Xd   = T_down.shape
sz,sy,sx   = int(soma_vox[0]), int(soma_vox[1]), int(soma_vox[2])

print(f'  shape={T_down.shape}  voxel={voxel:.3f}µm  soma=({sz},{sy},{sx})')

T_fg      = T_down[T_down > 0.02].ravel()
otsu_val  = float(threshold_otsu(T_fg))
MIN_T_TIP = round(float(np.clip(otsu_val, 0.20, 0.60)), 2)
print(f'  Otsu={otsu_val:.3f}  MIN_T_TIP={MIN_T_TIP}')

# geodesic_dist 로드 (FMM 결과)
geo_path = os.path.join(FN1, 'geodesic_dist.npy')
if os.path.exists(geo_path):
    geodesic_dist = np.load(geo_path)
    print(f'  geodesic_dist loaded from {geo_path}')
else:
    print(f'  [!] geodesic_dist.npy not found → method A 불가')
    print(f'      step3_auto.ipynb Cell 5 실행 후 아래 셀에서 저장하세요:')
    print(f'      np.save("output/FN1_01/geodesic_dist.npy", geodesic_dist)')
    geodesic_dist = None

# ── 공통 유틸 ─────────────────────────────────────────────────
def path_length_um(path, v):
    if len(path) < 2: return 0.0
    a = np.array(path, dtype=np.float32)
    return float(np.linalg.norm(np.diff(a, axis=0), axis=1).sum()) * v

def write_swc(rows, path, comment=''):
    header = [f'# test_skel_tracing.py  {comment}',
              '# id type x y z radius parent']
    with open(path, 'w') as f:
        f.write('\n'.join(header) + '\n')
        for r in rows:
            f.write(f'{r[0]} {r[1]} {r[2]:.4f} {r[3]:.4f} {r[4]:.4f} {r[5]:.4f} {r[6]}\n')
    print(f'  Saved: {path}  ({len(rows)} nodes)')

# ════════════════════════════════════════════════════════════
# METHOD A: geodesic_dist 피크 → 기존 traceback
# ════════════════════════════════════════════════════════════
if args.method in ('A', 'both') and geodesic_dist is not None:
    print('\n── Method A: geodesic_dist peaks as tips ──────────────')
    t0 = time.time()

    geo_finite = geodesic_dist.copy()
    geo_finite[~np.isfinite(geo_finite)] = 0.0

    MIN_DIST_VOX = int(round(10.0 / voxel))

    # geodesic_dist 피크 = 소마에서 가장 먼 점 = 진짜 branch endpoint
    # T > MIN_T_TIP 필터로 noise peak 제거
    geo_masked = geo_finite.copy()
    geo_masked[T_down < MIN_T_TIP * 0.5] = 0.0   # tube 밖은 0

    peaks = peak_local_max(geo_masked,
                           min_distance=MIN_DIST_VOX,
                           threshold_abs=float(soma_r_um / voxel),  # 소마 반경 밖만
                           exclude_border=False)
    print(f'  Peaks (geo dist): {len(peaks)}')

    # traceback with orientation guidance
    def traceback(tip_vox, max_steps=200000):
        cur  = (int(tip_vox[0]), int(tip_vox[1]), int(tip_vox[2]))
        path = []
        for _ in range(max_steps):
            path.append(cur)
            if soma_mask[cur]: break
            z,y,x = cur
            best_val   = geodesic_dist[cur]
            best_nb    = None
            fb_val     = geodesic_dist[cur]
            fb_nb      = None
            for dz in range(-1,2):
                for dy in range(-1,2):
                    for dx in range(-1,2):
                        if dz==dy==dx==0: continue
                        nz,ny,nx = z+dz,y+dy,x+dx
                        if not (0<=nz<Zd and 0<=ny<Yd and 0<=nx<Xd): continue
                        v = geodesic_dist[nz,ny,nx]
                        if v < fb_val:
                            fb_val, fb_nb = v, (nz,ny,nx)
                        if v < best_val:
                            step = np.array([dz,dy,dx], np.float32)
                            step /= np.linalg.norm(step)
                            cos = abs(float(np.dot(step, orient_down[nz,ny,nx])))
                            if cos >= 0.35:
                                best_val, best_nb = v, (nz,ny,nx)
            cur = best_nb if best_nb is not None else fb_nb
            if cur is None: break
        return path[::-1]

    all_paths = {}
    for i, tip in enumerate(peaks):
        key = (int(tip[0]),int(tip[1]),int(tip[2]))
        if not np.isfinite(geodesic_dist[key]): continue
        path = traceback(key)
        if path_length_um(path, voxel) < 5.0: continue
        all_paths[i] = path

    print(f'  Paths: {len(all_paths)}  ({time.time()-t0:.1f}s)')

    # SWC 빌드
    node_id_map = {}
    swc_rows    = [(1, 1,
                    float(sx)*voxel, float(sy)*voxel, float(sz)*voxel,
                    soma_r_um, -1)]
    next_id = 2
    for bi, path in all_paths.items():
        prev_id = 1
        for key in path:
            if soma_mask[key]: node_id_map[key]=1; prev_id=1; continue
            if key in node_id_map: prev_id=node_id_map[key]; continue
            z,y,x = key
            r = max(float(edt_down[z,y,x]), 0.1)
            swc_rows.append((next_id, 3, x*voxel, y*voxel, z*voxel, r, prev_id))
            node_id_map[key] = next_id; prev_id = next_id; next_id += 1

    write_swc(swc_rows,
              os.path.join(FN1, 'test_A_geo_tips.swc'),
              'method=A geo_dist_peaks')

# ════════════════════════════════════════════════════════════
# METHOD B: skeletonize → soma rooted tree
# ════════════════════════════════════════════════════════════
if args.method in ('B', 'both'):
    print('\n── Method B: skeletonize + soma rooted tree ───────────')
    t0 = time.time()

    T_THR = otsu_val * args.t_thr_factor
    print(f'  T threshold = {T_THR:.3f}  (Otsu={otsu_val:.3f} × {args.t_thr_factor})')

    # 1) tube mask: T > T_THR, 소마 포함
    tube_mask = (T_down >= T_THR) | soma_mask
    print(f'  Tube mask voxels: {tube_mask.sum():,}')

    # 2) 소마에 연결된 connected component만 유지
    labeled    = cc_label(tube_mask, connectivity=3)
    soma_lbl   = int(labeled[sz, sy, sx])
    connected  = (labeled == soma_lbl)
    print(f'  Connected to soma: {connected.sum():,} voxels')

    # 3) 3D skeletonize
    print('  Skeletonizing...', flush=True)
    skel = skeletonize(connected)
    n_skel = int(skel.sum())
    print(f'  Skeleton voxels: {n_skel:,}  ({time.time()-t0:.1f}s)')

    # 4) skeleton graph 추출
    # 26-connectivity로 각 voxel의 skeleton 이웃 수 계산
    skel_coords = np.argwhere(skel)
    skel_set    = set(map(tuple, skel_coords))

    def skel_neighbors(z,y,x):
        nb = []
        for dz in range(-1,2):
            for dy in range(-1,2):
                for dx in range(-1,2):
                    if dz==dy==dx==0: continue
                    n = (z+dz,y+dy,x+dx)
                    if n in skel_set: nb.append(n)
        return nb

    # 소마 voxel과 가장 가까운 skeleton voxel을 root로
    soma_voxels = np.argwhere(soma_mask & skel)
    if len(soma_voxels) == 0:
        # skeleton이 소마 mask와 겹치지 않으면 가장 가까운 voxel 사용
        dists = np.linalg.norm(skel_coords - np.array([sz,sy,sx]), axis=1)
        root_vox = tuple(skel_coords[np.argmin(dists)])
    else:
        root_vox = tuple(soma_voxels[0])
    print(f'  Skeleton root: {root_vox}')

    # 5) BFS로 soma rooted spanning tree 빌드
    from collections import deque
    parent_map  = {root_vox: None}
    queue       = deque([root_vox])
    visited     = {root_vox}
    while queue:
        cur = queue.popleft()
        for nb in skel_neighbors(*cur):
            if nb not in visited:
                visited.add(nb)
                parent_map[nb] = cur
                queue.append(nb)

    print(f'  Tree nodes: {len(parent_map):,}')

    # 6) SWC 변환
    # 각 skeleton voxel → SWC node, 반경은 EDT
    vox_to_id = {}
    swc_rows  = []
    next_id   = 1

    # root (soma)
    rz,ry,rx = root_vox
    soma_r   = max(float(edt_down[rz,ry,rx]), soma_r_um)
    swc_rows.append((1, 1, rx*voxel, ry*voxel, rz*voxel, soma_r, -1))
    vox_to_id[root_vox] = 1
    next_id = 2

    # BFS order로 나머지 노드 추가
    queue   = deque([root_vox])
    visited2 = {root_vox}
    while queue:
        cur = queue.popleft()
        for nb in skel_neighbors(*cur):
            if nb not in visited2 and nb in parent_map:
                visited2.add(nb)
                par_id = vox_to_id[cur]
                nz,ny,nx = nb
                r = max(float(edt_down[nz,ny,nx]), 0.1)
                swc_rows.append((next_id, 3, nx*voxel, ny*voxel, nz*voxel, r, par_id))
                vox_to_id[nb] = next_id
                next_id += 1
                queue.append(nb)

    print(f'  SWC nodes: {len(swc_rows):,}  ({time.time()-t0:.1f}s)')
    write_swc(swc_rows,
              os.path.join(FN1, 'test_B_skeleton.swc'),
              f'method=B T_thr={T_THR:.3f}')

print('\nDone.')
print()
print('FIJI로 확인:')
print('  python render_filled_tubes.py --swc output/FN1_01/test_A_geo_tips.swc')
print('  python render_filled_tubes.py --swc output/FN1_01/test_B_skeleton.swc')
