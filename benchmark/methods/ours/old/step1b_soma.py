# ── Config ───────────────────────────────────────────────────
import argparse as _ap, os, numpy as np, tifffile
import warnings
warnings.filterwarnings('ignore')
_a = _ap.ArgumentParser(); _a.add_argument('--out-dir', default='output'); _a = _a.parse_args()
OUT_DIR = _a.out_dir
os.makedirs(OUT_DIR, exist_ok=True)

SOMA_VOXEL              = None
SOMA_SIGMA_UM           = 4.0   # Gaussian smoothing for soma detection (µm)
SOMA_SEARCH_RADIUS_UM   = 30.0  # search sphere radius around seed (µm)
SOMA_MAX_TUBULARITY     = 0.25
SOMA_OPEN_RADIUS_UM     = 2.0   # morphological opening — removes speckle (µm)
SOMA_CLOSE_RADIUS_UM    = 1.5   # morphological closing — fills small holes (µm)
SOMA_ERODE_RADIUS_UM    = 1.5   # final erosion — shrinks to nucleus (µm)

# Load
tub        = np.load(f'{OUT_DIR}/tubularity.npz')
T_combined = tub['T_combined']
I_OOF_raw  = tub['I_OOF_raw']
voxel_iso  = float(tub['voxel_iso'])
stack      = tifffile.imread(f'{OUT_DIR}/stack_preprocessed.tif').astype(np.float32)
print(f'Loaded: T_combined {T_combined.shape}  stack {stack.shape}  voxel_iso={voxel_iso:.4f} µm')

# ── Convert µm parameters → voxels ──────────────────────────
SOMA_SIGMA          = SOMA_SIGMA_UM         / voxel_iso
SOMA_SEARCH_RADIUS  = max(10, int(round(SOMA_SEARCH_RADIUS_UM / voxel_iso)))
SOMA_OPEN_RADIUS_VX = max(1,  int(round(SOMA_OPEN_RADIUS_UM   / voxel_iso)))
SOMA_CLOSE_RADIUS_VX= max(1,  int(round(SOMA_CLOSE_RADIUS_UM  / voxel_iso)))
SOMA_ERODE_RADIUS_VX= max(1,  int(round(SOMA_ERODE_RADIUS_UM  / voxel_iso)))
print(f'Soma params (vox): sigma={SOMA_SIGMA:.1f}  search_r={SOMA_SEARCH_RADIUS}'
      f'  open={SOMA_OPEN_RADIUS_VX}  close={SOMA_CLOSE_RADIUS_VX}  erode={SOMA_ERODE_RADIUS_VX}')

# ── Soma Detection ───────────────────────────────────────────
from scipy.ndimage import gaussian_filter

SOMA_BORDER_PAD = 10   # edge exclusion (voxels) to avoid border artifacts

if SOMA_VOXEL is not None:
    soma_voxel = tuple(int(v) for v in SOMA_VOXEL)
    print(f'Soma pinned manually: {soma_voxel}')
else:
    stack_f32   = stack.astype(np.float32)
    stack_norm  = (stack_f32 - stack_f32.min()) / (stack_f32.max() - stack_f32.min() + 1e-6)
    soma_score  = stack_norm * (1.0 - T_combined)
    soma_smooth = gaussian_filter(soma_score.astype(np.float32), sigma=SOMA_SIGMA)
    # mask out border to avoid edge artifacts
    p = SOMA_BORDER_PAD
    NZ, NY, NX = soma_smooth.shape
    border_mask = np.ones_like(soma_smooth, dtype=bool)
    border_mask[p:NZ-p, p:NY-p, p:NX-p] = False
    soma_smooth_masked = soma_smooth.copy()
    soma_smooth_masked[border_mask] = 0.0
    soma_voxel  = tuple(int(v) for v in
                        np.unravel_index(soma_smooth_masked.argmax(), soma_smooth_masked.shape))
    print(f'Auto-detected soma voxel : {soma_voxel}')
    print(f'  soma_score  at peak    : {soma_score[soma_voxel]:.4f}')
    print(f'  T_combined  at peak    : {T_combined[soma_voxel]:.4f}')
    print(f'  stack_norm  at peak    : {stack_norm[soma_voxel]:.4f}')
    print(f'\nIf this looks wrong, set SOMA_VOXEL = {soma_voxel} in Config and adjust.')

# ── Soma Segmentation ────────────────────────────────────────
from skimage.filters import threshold_otsu
from scipy.ndimage import label as sp_label, binary_fill_holes
from scipy.ndimage import binary_opening, binary_closing, binary_erosion
from skimage.morphology import ball

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
thresh   = threshold_otsu(crop[soma_candidates])
bin_crop = (crop > thresh) & soma_candidates
print(f'Otsu threshold : {thresh:.4f}  ({100 * bin_crop.sum() / sphere.sum():.1f}% of sphere)')

labeled, n_comp = sp_label(bin_crop)
lbl = labeled[cz, cy, cx]

if lbl == 0:
    print('WARNING: soma seed not in any component — try increasing SOMA_SEARCH_RADIUS '
          'or SOMA_MAX_TUBULARITY')
    soma_mask_crop = np.zeros_like(bin_crop)
else:
    soma_mask_crop = labeled == lbl
    print(f'Connected components in sphere : {n_comp}  → using component #{lbl}')

soma_mask_crop = binary_opening(soma_mask_crop, structure=ball(SOMA_OPEN_RADIUS_VX))
soma_mask_crop = binary_closing(soma_mask_crop, structure=ball(SOMA_CLOSE_RADIUS_VX))
soma_mask_crop = binary_erosion(soma_mask_crop, structure=ball(SOMA_ERODE_RADIUS_VX))
soma_mask_crop = binary_fill_holes(soma_mask_crop)

soma_mask = np.zeros(stack.shape, dtype=bool)
soma_mask[zs, ys, xs] = soma_mask_crop

soma_voxels_coords = np.argwhere(soma_mask)
n_soma_vox         = len(soma_voxels_coords)

if n_soma_vox == 0:
    # morphological ops eroded everything — fall back to a sphere around the seed
    FALLBACK_R = max(3, SOMA_OPEN_RADIUS_VX // 2)
    print(f'WARNING: soma mask empty after morphology (open_r={SOMA_OPEN_RADIUS_VX}). '
          f'Falling back to r={FALLBACK_R} vox sphere around seed {soma_voxel}.')
    ZZ2, YY2, XX2 = np.ogrid[:soma_mask.shape[0], :soma_mask.shape[1], :soma_mask.shape[2]]
    z0f, y0f, x0f = soma_voxel
    soma_mask = ((ZZ2-z0f)**2 + (YY2-y0f)**2 + (XX2-x0f)**2) <= FALLBACK_R**2
    soma_voxels_coords = np.argwhere(soma_mask)
    n_soma_vox = len(soma_voxels_coords)

soma_centroid_vox  = soma_voxels_coords.mean(axis=0) if n_soma_vox > 0 \
                     else np.array(soma_voxel, dtype=float)
soma_volume_um3    = n_soma_vox * voxel_iso ** 3
soma_equiv_r_um    = (3 * soma_volume_um3 / (4 * np.pi)) ** (1 / 3)

print(f'Soma voxels       : {n_soma_vox:,}')
print(f'Soma volume       : {soma_volume_um3:.1f} µm³')
print(f'Equivalent radius : {soma_equiv_r_um:.2f} µm')
print(f'Centroid (vox)    : {tuple(soma_centroid_vox.round(1))}')

# ── Save soma checkpoint ──────────────────────────────────────
import json
from skimage.measure import marching_cubes
from scipy.ndimage import gaussian_filter as gf

sc  = soma_voxels_coords
pad = 4
zs_v = slice(max(0, int(sc[:, 0].min()) - pad), int(sc[:, 0].max()) + pad + 1)
ys_v = slice(max(0, int(sc[:, 1].min()) - pad), int(sc[:, 1].max()) + pad + 1)
xs_v = slice(max(0, int(sc[:, 2].min()) - pad), int(sc[:, 2].max()) + pad + 1)

soma_crop   = soma_mask[zs_v, ys_v, xs_v].astype(np.float32)
soma_smooth = gf(soma_crop, sigma=1.5)
verts, faces, _, _ = marching_cubes(soma_smooth, level=0.5,
                                    spacing=(voxel_iso, voxel_iso, voxel_iso))

np.savez_compressed(
    f'{OUT_DIR}/soma.npz',
    soma_mask        = soma_mask,
    soma_centroid_vox= soma_centroid_vox,
    soma_equiv_r_um  = np.array(soma_equiv_r_um),
    voxel_iso        = np.array(voxel_iso),
    mesh_verts       = verts.astype(np.float32),
    mesh_faces       = faces.astype(np.int32),
)

with open(f'{OUT_DIR}/soma.json', 'w') as f:
    json.dump({
        'centroid_vox': soma_centroid_vox.tolist(),
        'radius_um':    float(soma_equiv_r_um),
    }, f, indent=2)

print(f'Saved: {OUT_DIR}/soma.npz  verts={verts.shape}  faces={faces.shape}')
print(f'Saved: {OUT_DIR}/soma.json  centroid={soma_centroid_vox.round(1).tolist()}  r={soma_equiv_r_um:.2f} µm')

# ── Soma Anchor ───────────────────────────────────────────────
from scipy.ndimage import distance_transform_edt
import time

SOMA_ANCHOR_OFFSET = 0.5

t0 = time.time()
coords = np.argwhere(soma_mask)
pad = 5
NZ, NY, NX = soma_mask.shape
z0b = max(0,  int(coords[:,0].min()) - pad)
y0b = max(0,  int(coords[:,1].min()) - pad)
x0b = max(0,  int(coords[:,2].min()) - pad)
z1b = min(NZ, int(coords[:,0].max()) + pad + 1)
y1b = min(NY, int(coords[:,1].max()) + pad + 1)
x1b = min(NX, int(coords[:,2].max()) + pad + 1)

mask_crop = soma_mask[z0b:z1b, y0b:y1b, x0b:x1b]
dist_crop = distance_transform_edt(mask_crop).astype(np.float32)
dist_crop /= (dist_crop.max() + 1e-8)
print(f'EDT: {mask_crop.shape} crop  {time.time()-t0:.1f}s')

t0 = time.time()
anchor_val = T_combined.max() + SOMA_ANCHOR_OFFSET
T_anchored = T_combined.copy()
T_crop     = T_anchored[z0b:z1b, y0b:y1b, x0b:x1b]
T_crop[mask_crop] = (dist_crop * anchor_val)[mask_crop]
T_anchored[z0b:z1b, y0b:y1b, x0b:x1b] = T_crop
T_anchored /= T_anchored.max()

cz = int(soma_centroid_vox[0])
cy = int(soma_centroid_vox[1])
cx = int(soma_centroid_vox[2])
print(f'Before: T_combined[soma] = {T_combined[cz, cy, cx]:.4f}')
print(f'After : T_anchored[soma] = {T_anchored[cz, cy, cx]:.4f}  (global max=1.0)')
print(f'Anchor: {time.time()-t0:.1f}s')

t0 = time.time()
tub_rest = np.load(f'{OUT_DIR}/tubularity.npz')
np.savez_compressed(f'{OUT_DIR}/tubularity_anchored.npz',
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
print(f'Saved: {OUT_DIR}/tubularity_anchored.npz  ({time.time()-t0:.0f}s)')
