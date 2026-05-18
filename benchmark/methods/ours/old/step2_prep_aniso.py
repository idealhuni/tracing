# ── Config ──────────────────────────────────────────────────
import argparse as _ap, os
_a = _ap.ArgumentParser(); _a.add_argument('--out-dir', default='output'); _a = _a.parse_args()
OUT_DIR = _a.out_dir
os.makedirs(OUT_DIR, exist_ok=True)

TUBULARITY_NPZ = f'{OUT_DIR}/tubularity_anchored.npz'
SOMA_JSON      = f'{OUT_DIR}/soma.json'
OUT_NPZ        = f'{OUT_DIR}/prep_riem.npz'
DOWNSAMPLE     = 4

# ── Load + Downsample ───────────────────────────────────────
import numpy as np, json, gc, time
from scipy.ndimage import zoom

t0 = time.time()

tub          = np.load(TUBULARITY_NPZ)
T            = tub['T_combined'].astype(np.float32)
radius       = tub['radius_map'].astype(np.float32)
orient_field = tub['orient_field'].astype(np.float32)
soma_mask    = tub['soma_mask']
voxel_iso    = float(tub['voxel_iso'])
del tub; gc.collect()

with open(SOMA_JSON) as f:
    soma = json.load(f)
soma_vox  = np.array(soma['centroid_vox'], dtype=np.float32)
soma_r_um = float(soma['radius_um'])

factor = 1.0 / DOWNSAMPLE

T_down         = zoom(T,         factor, order=1).astype(np.float32)
radius_down    = zoom(radius,    factor, order=1).astype(np.float32)
soma_mask_down = zoom(soma_mask, factor, order=0).astype(bool)
del T, radius, soma_mask; gc.collect()

print('Downsampling orient_field...', flush=True)
vz = zoom(orient_field[..., 0], factor, order=1).astype(np.float32)
vy = zoom(orient_field[..., 1], factor, order=1).astype(np.float32)
vx = zoom(orient_field[..., 2], factor, order=1).astype(np.float32)
del orient_field; gc.collect()

norm = np.sqrt(vz**2 + vy**2 + vx**2) + 1e-8
vz /= norm; vy /= norm; vx /= norm
orient_down = np.stack([vz, vy, vx], axis=-1).astype(np.float16)
del vz, vy, vx, norm; gc.collect()

voxel_down    = voxel_iso * DOWNSAMPLE
soma_vox_down = (soma_vox * factor).astype(np.float32)
Zd, Yd, Xd   = T_down.shape

print(f'T_down         : {T_down.shape}  voxel={voxel_down:.3f} µm  ({time.time()-t0:.1f}s)')
print(f'orient_down    : {orient_down.shape}  dtype={orient_down.dtype}')
print(f'soma_mask_down : {soma_mask_down.sum():,} voxels')
print(f'T range        : {T_down.min():.4f} – {T_down.max():.4f}')
print(f'soma_vox_down  : {soma_vox_down}  r={soma_r_um:.2f} µm')

# ── EDT radius ───────────────────────────────────────────────
from scipy.ndimage import distance_transform_edt, gaussian_filter

EDT_THRESHOLD  = 0.08
EDT_SMOOTH_SIG = 2.0

t0 = time.time()
fg_mask = T_down > EDT_THRESHOLD

edt_raw  = distance_transform_edt(fg_mask).astype(np.float32) * voxel_down
edt_down = gaussian_filter(edt_raw, sigma=EDT_SMOOTH_SIG).astype(np.float32)
del edt_raw; gc.collect()

fg_edt = edt_down[fg_mask]
print(f'EDT done in {time.time()-t0:.1f}s')
print(f'edt (foreground only):')
print(f'  min={fg_edt.min():.3f}  median={np.median(fg_edt):.3f}'
      f'  p99={np.percentile(fg_edt,99):.3f}  max={fg_edt.max():.3f} µm')
del fg_edt

# ── Save ────────────────────────────────────────────────────
np.savez_compressed(OUT_NPZ,
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
)
print(f'Saved: {OUT_NPZ}')
print(f'  T_down         {T_down.shape}  float32')
print(f'  edt_down       {edt_down.shape}  float32  ← EDT radius')
print(f'  radius_down    {radius_down.shape}  float32  ← OOF radius')
print(f'  orient_down    {orient_down.shape}  float16')
print(f'  soma_mask_down {soma_mask_down.shape}  bool')
print(f'  voxel_down     {voxel_down:.3f} µm')
