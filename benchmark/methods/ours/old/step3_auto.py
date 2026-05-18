# ── Config ──────────────────────────────────────────────────
import argparse as _ap, os
_a = _ap.ArgumentParser(); _a.add_argument('--out-dir', default='output'); _a = _a.parse_args()
OUT_DIR = _a.out_dir
os.makedirs(OUT_DIR, exist_ok=True)

MORSE_NPZ         = f'{OUT_DIR}/prep_riem.npz'
OUT_SWC           = f'{OUT_DIR}/neurons_auto.swc'

COST_TARGET_RATIO = 3000
MIN_DIST_UM       = 25.0

GAMMA             = 0.95
SIGMA_PERP        = 1.0
MAX_TIPS          = 300
MIN_RADIUS_UM     = 0.1
MERGE_DOT_MIN     = 0.92
MIN_TORTUOSITY    = 1.02
MIN_PATH_LEN_UM_FLOOR = 5.0

# ── Load ────────────────────────────────────────────────────
import numpy as np, time, gc
from tqdm import tqdm

d              = np.load(MORSE_NPZ)
T_down         = d['T_down'].astype(np.float32)
radius_down    = d['radius_down'].astype(np.float32)
edt_down       = d['edt_down'].astype(np.float32)
orient_down    = d['orient_down'].astype(np.float32)
soma_mask_down = d['soma_mask_down']
voxel_down     = float(d['voxel_down'])
soma_vox_down  = d['soma_vox_down'].astype(np.float64)
soma_r_um      = float(d['soma_r_um'])

Zd, Yd, Xd = T_down.shape
print(f'T_down    : {T_down.shape}  voxel={voxel_down:.3f} µm')
print(f'soma_r_um : {soma_r_um:.2f} µm')

# ── Parameter auto-detection ─────────────────────────────────
from skimage.filters import threshold_otsu

T_fg = T_down[T_down > 0.02].ravel()
otsu_val = float(threshold_otsu(T_fg))

MIN_T_TIP    = round(float(np.clip(otsu_val, 0.20, 0.60)), 2)
ALPHA        = round(float(np.clip(np.log(COST_TARGET_RATIO), 4.0, 12.0)), 1)
T_mean_tube  = float(T_down[T_down > MIN_T_TIP * 0.5].mean())

MIN_DIST_VOX    = int(round(MIN_DIST_UM / voxel_down))
MIN_MEAN_T      = round(MIN_T_TIP * 0.35, 2)
MIN_SEG_T       = round(MIN_T_TIP * 0.05, 3)
MIN_PATH_LEN_UM = round(float(max(MIN_PATH_LEN_UM_FLOOR, soma_r_um * 0.5)), 1)
MERGE_DIST_UM   = round(float(max(10.0, soma_r_um * 1.5)), 1)

print('=' * 56)
print('  Auto-detected parameters')
print('=' * 56)
print(f'  ALPHA        = {ALPHA:<6}  (log({COST_TARGET_RATIO})  cost@T=1={np.exp(-ALPHA):.4f})')
print(f'  MIN_T_TIP    = {MIN_T_TIP:<6}  (Otsu={otsu_val:.3f})')
print(f'  MIN_DIST_VOX = {MIN_DIST_VOX:<6}  ({MIN_DIST_UM} µm / {voxel_down:.3f} µm/vox)')
print(f'  MIN_MEAN_T   = {MIN_MEAN_T:<6}  (MIN_T_TIP × 0.35)')
print(f'  MIN_SEG_T    = {MIN_SEG_T:<6}  (MIN_T_TIP × 0.05)')
print(f'  MIN_PATH_LEN = {MIN_PATH_LEN_UM:<6}  µm')
print(f'  MERGE_DIST   = {MERGE_DIST_UM:<6}  µm')
print('=' * 56)

# ── FileHFM setup ────────────────────────────────────────────
import agd
from agd import Eikonal, Metrics

txt = 'FileHFM_binary_dir.txt'
with open(txt) as f:
    BIN_DIR = f.read().strip()
agd.Eikonal.LibraryCall.binary_dir['FileHFM'] = BIN_DIR
print(f'FileHFM: {BIN_DIR}')
print(f'  Riemann3: {os.path.exists(os.path.join(BIN_DIR, "FileHFM_Riemann3"))}')

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
print(f'Anisotropy ratio: {SIGMA_PERP/sp_min:.1f}:1  (at T=max)')

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
print(f'FMM done in {time.time()-t0:.1f}s')
print(f'Reachable: {np.isfinite(geodesic_dist).sum():,}')
print(f'Geo range: {geodesic_dist[np.isfinite(geodesic_dist)].min():.3f} – '
      f'{geodesic_dist[np.isfinite(geodesic_dist)].max():.3f}')

# ── Tip detection ────────────────────────────────────────────
from skimage.feature import peak_local_max

_peaks = peak_local_max(
    T_down,
    min_distance   = MIN_DIST_VOX,
    threshold_abs  = MIN_T_TIP,
    exclude_border = False,
)
tip_coords_all = _peaks if _peaks.dtype != bool else np.argwhere(_peaks)
tip_vals       = T_down[tip_coords_all[:,0], tip_coords_all[:,1], tip_coords_all[:,2]]
sort_idx       = np.argsort(tip_vals)[::-1]
tip_coords_s   = tip_coords_all[sort_idx][:MAX_TIPS]
tip_vals_s     = tip_vals[sort_idx][:MAX_TIPS]

reachable    = np.isfinite(geodesic_dist[tip_coords_s[:,0],
                                         tip_coords_s[:,1],
                                         tip_coords_s[:,2]])
tip_coords_s = tip_coords_s[reachable]
tip_vals_s   = tip_vals_s[reachable]

geo_finite = geodesic_dist.copy()
geo_finite[~np.isfinite(geo_finite)] = 0

print(f'Tips detected : {len(tip_coords_all):,}')
print(f'Tips selected : {len(tip_coords_s)}  T={tip_vals_s[-1]:.3f}–{tip_vals_s[0]:.3f}')

# ── Traceback ────────────────────────────────────────────────
t0 = time.time()

def traceback_discrete(tip_vox, geo_dist, soma_mask, Zd, Yd, Xd, max_steps=200000):
    cur  = (int(tip_vox[0]), int(tip_vox[1]), int(tip_vox[2]))
    path = []
    for _ in range(max_steps):
        path.append(cur)
        if soma_mask[cur]: break
        z, y, x  = cur
        best_val = geo_dist[cur]
        best_nb  = None
        for dz in range(-1, 2):
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if dz == dy == dx == 0: continue
                    nz, ny, nx = z+dz, y+dy, x+dx
                    if 0 <= nz < Zd and 0 <= ny < Yd and 0 <= nx < Xd:
                        v = geo_dist[nz, ny, nx]
                        if v < best_val:
                            best_val, best_nb = v, (nz, ny, nx)
        if best_nb is None: break
        cur = best_nb
    return path[::-1]

all_paths = {}
n_short = n_trimmed = n_too_short_trim = n_straight = 0

for i, tip in enumerate(tqdm(tip_coords_s, desc='traceback', unit='tip')):
    key = (int(tip[0]), int(tip[1]), int(tip[2]))
    if not np.isfinite(geodesic_dist[key]): continue
    path = traceback_discrete(key, geodesic_dist, soma_mask_down, Zd, Yd, Xd)

    if len(path) * voxel_down < MIN_PATH_LEN_UM:
        n_short += 1; continue

    t_vals  = np.array([T_down[k] for k in path], dtype=np.float32)
    trimmed = False
    half    = len(path) // 2

    bad = np.where(t_vals[half:] < MIN_SEG_T)[0]
    if len(bad) > 0:
        path = path[:half + bad[0]]; t_vals = t_vals[:half + bad[0]]; trimmed = True
        if len(path) * voxel_down < MIN_PATH_LEN_UM:
            n_too_short_trim += 1; continue

    if len(t_vals) > half:
        rm = np.cumsum(t_vals[half:]) / np.arange(1, len(t_vals)-half+1)
        bm = np.where(rm < MIN_MEAN_T)[0]
        if len(bm) > 0:
            path = path[:half + bm[0]]; t_vals = t_vals[:half + bm[0]]; trimmed = True
            if len(path) * voxel_down < MIN_PATH_LEN_UM:
                n_too_short_trim += 1; continue

    if trimmed: n_trimmed += 1

    path_len_um = len(path) * voxel_down
    euclid_um   = np.linalg.norm(
        np.array(path[-1], dtype=np.float32) -
        np.array(path[0],  dtype=np.float32)) * voxel_down
    if path_len_um / (euclid_um + 1e-8) < MIN_TORTUOSITY:
        n_straight += 1; continue

    all_paths[i] = path

print(f'Traceback : {time.time()-t0:.1f}s')
print(f'Paths kept: {len(all_paths)}')
print(f'  trimmed & kept: {n_trimmed}')
print(f'  filtered — short:{n_short}  too_short_trim:{n_too_short_trim}  straight:{n_straight}')
if all_paths:
    lens = [len(p)*voxel_down for p in all_paths.values()]
    print(f'  Length: min={min(lens):.1f}  max={max(lens):.1f}  mean={np.mean(lens):.1f} µm')

# ── Tree 구성 ────────────────────────────────────────────────
sc = soma_vox_down
soma_x = sc[2]*voxel_down; soma_y = sc[1]*voxel_down; soma_z = sc[0]*voxel_down

swc_rows    = [(1, 1, soma_x, soma_y, soma_z, soma_r_um, -1)]
node_id_map = {}
next_id     = 2

sorted_keys = sorted(all_paths.keys(),
    key=lambda k: float(geo_finite[tip_coords_s[k][0],
                                   tip_coords_s[k][1],
                                   tip_coords_s[k][2]]),
    reverse=False)

for bi in sorted_keys:
    prev_swc_id = 1
    for key in all_paths[bi]:
        if soma_mask_down[key]:
            node_id_map[key] = 1; prev_swc_id = 1; continue
        if key in node_id_map:
            prev_swc_id = node_id_map[key]; continue
        z, y, x = key
        r = max(float(edt_down[z, y, x]), MIN_RADIUS_UM)
        swc_rows.append((next_id, 3,
                         x*voxel_down, y*voxel_down, z*voxel_down,
                         r, prev_swc_id))
        node_id_map[key] = next_id; prev_swc_id = next_id; next_id += 1

print(f'SWC nodes: {next_id-1:,}  (branches: {len(all_paths)})')
swc_rows_dict = {r[0]: r for r in swc_rows}

# ── Primary merge ────────────────────────────────────────────
from scipy.spatial import cKDTree
from collections import defaultdict

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
        if abs(float(p_dirs[i] @ p_dirs[j])) >= MERGE_DOT_MIN:
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

print(f'Primary before merge : {len(pids)}')
print(f'Merged (removed)     : {merged_count}')
print(f'SWC nodes after merge: {len(swc_rows):,}')

# ── Save ─────────────────────────────────────────────────────
header = [
    f'# tracer_aniso AUTO — Riemannian FMM',
    f'# ALPHA={ALPHA}  MIN_T_TIP={MIN_T_TIP}  GAMMA={GAMMA}  SIGMA_PERP={SIGMA_PERP}',
    f'# MIN_SEG_T={MIN_SEG_T}  MIN_MEAN_T={MIN_MEAN_T}  MIN_TORTUOSITY={MIN_TORTUOSITY}',
    f'# tips={len(tip_coords_s)}  paths={len(all_paths)}  nodes={next_id-1}',
    '# id type x y z radius parent',
]
lines = header + [
    f'{r[0]} {r[1]} {r[2]:.4f} {r[3]:.4f} {r[4]:.4f} {r[5]:.4f} {r[6]}'
    for r in swc_rows
]
with open(OUT_SWC, 'w') as f:
    f.write('\n'.join(lines) + '\n')
print(f'Saved: {OUT_SWC}')
