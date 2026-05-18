"""
Phase B fixes — apply to fix1/ notebooks only. Originals untouched.

Fix B1: step3_auto.ipynb     — soma_mask_down empty assert (load cell)
Fix B2: step2_prep_aniso.ipynb — orient_field NaN/Inf replace (load cell)
Fix B3: step0_preprocess.ipynb — VOXEL_Z range assert (load cell)
Fix B4: step3_auto.ipynb     — MIN_TORTUOSITY 1.02 → 1.0 (config cell)
"""
import json, pathlib

BASE = pathlib.Path(__file__).parent   # fix1/


def patch_cell(nb, match_fn, old, new, label):
    for cell in nb['cells']:
        src = ''.join(cell.get('source', []))
        if match_fn(src) and old in src:
            cell['source'] = src.replace(old, new).splitlines(keepends=True)
            print(f'  {label}: OK')
            return True
    print(f'  {label}: NOT FOUND — check strings')
    return False


def save(nb, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)


# ══════════════════════════════════════════════════════════════
# Fix B1 + B4 — step3_auto.ipynb
# ══════════════════════════════════════════════════════════════
path3 = BASE / 'step3_auto.ipynb'
with open(path3) as f:
    nb3 = json.load(f)

print('step3_auto.ipynb:')

# B4: MIN_TORTUOSITY 1.02 → 1.0  (config cell)
patch_cell(
    nb3,
    match_fn=lambda s: 'MIN_TORTUOSITY' in s and 'GAMMA' in s,
    old='MIN_TORTUOSITY = 1.02',
    new='MIN_TORTUOSITY = 1.0   # lowered from 1.02 — preserves near-straight dendrites',
    label='B4 MIN_TORTUOSITY 1.02→1.0',
)

# B1: soma_mask_down empty assert (load cell)
OLD_B1 = (
    "Zd, Yd, Xd = T_down.shape\n"
    "print(f'T_down      : {T_down.shape}  voxel={voxel_down:.3f} µm')\n"
    "print(f'soma_r_um   : {soma_r_um:.2f} µm')\n"
    "print(f'border_pad_z: {BORDER_PAD_Z} vox  ({BORDER_PAD_Z * voxel_down:.1f} µm)')"
)
NEW_B1 = (
    "Zd, Yd, Xd = T_down.shape\n"
    "print(f'T_down      : {T_down.shape}  voxel={voxel_down:.3f} µm')\n"
    "print(f'soma_r_um   : {soma_r_um:.2f} µm')\n"
    "print(f'border_pad_z: {BORDER_PAD_Z} vox  ({BORDER_PAD_Z * voxel_down:.1f} µm)')\n"
    "\n"
    "# Validate soma mask — empty mask → no FMM seeds → silent failure\n"
    "assert soma_mask_down.sum() > 0, (\n"
    "    f'soma_mask_down is empty after {DOWNSAMPLE}× downsampling! '\n"
    "    'Check DOWNSAMPLE or soma segmentation in step1b.'\n"
    ")\n"
    "print(f'soma_mask_down: {soma_mask_down.sum():,} voxels  OK')"
)
patch_cell(
    nb3,
    match_fn=lambda s: 'soma_mask_down' in s and 'soma_vox_down' in s,
    old=OLD_B1,
    new=NEW_B1,
    label='B1 soma_mask_down assert',
)

save(nb3, path3)


# ══════════════════════════════════════════════════════════════
# Fix B2 — step2_prep_aniso.ipynb
# ══════════════════════════════════════════════════════════════
path2 = BASE / 'step2_prep_aniso.ipynb'
with open(path2) as f:
    nb2 = json.load(f)

print('\nstep2_prep_aniso.ipynb:')

OLD_B2 = (
    "norm = np.sqrt(vz**2 + vy**2 + vx**2) + 1e-8\n"
    "vz /= norm; vy /= norm; vx /= norm\n"
    "orient_down = np.stack([vz, vy, vx], axis=-1).astype(np.float16)"
)
NEW_B2 = (
    "norm = np.sqrt(vz**2 + vy**2 + vx**2) + 1e-8\n"
    "vz /= norm; vy /= norm; vx /= norm\n"
    "orient_down = np.stack([vz, vy, vx], axis=-1).astype(np.float16)\n"
    "\n"
    "# Validate orientation field — NaN/Inf here corrupts the Riemannian metric\n"
    "_nan_count = int((~np.isfinite(orient_down.astype(np.float32))).sum())\n"
    "if _nan_count > 0:\n"
    "    print(f'WARNING: {_nan_count} NaN/Inf in orient_down — replacing with 0')\n"
    "    orient_down = np.where(np.isfinite(orient_down), orient_down,\n"
    "                           np.float16(0)).astype(np.float16)\n"
    "else:\n"
    "    print('orient_down: no NaN/Inf  OK')"
)
patch_cell(
    nb2,
    match_fn=lambda s: 'orient_down' in s and 'vz /= norm' in s,
    old=OLD_B2,
    new=NEW_B2,
    label='B2 orient_field NaN check',
)

save(nb2, path2)


# ══════════════════════════════════════════════════════════════
# Fix B3 — step0_preprocess.ipynb
# ══════════════════════════════════════════════════════════════
path0 = BASE / 'step0_preprocess.ipynb'
with open(path0) as f:
    nb0 = json.load(f)

print('\nstep0_preprocess.ipynb:')

OLD_B3 = (
    "print(f'  Z  voxel: {VOXEL_Z:.4f} um/slice')\n"
    "print(f'  Anisotropy (Z/XY): {VOXEL_Z / voxel_xy:.2f}x')"
)
NEW_B3 = (
    "# Validate VOXEL_Z — only manually set parameter; 2× error distorts entire Z axis\n"
    "assert 0.1 <= VOXEL_Z <= 10.0, (\n"
    "    f'VOXEL_Z={VOXEL_Z} out of plausible range [0.1, 10.0] µm/slice. '\n"
    "    'Check your acquisition settings.'\n"
    ")\n"
    "if VOXEL_Z < 0.5:\n"
    "    print(f'WARNING: VOXEL_Z={VOXEL_Z} µm very small — confirm slice thickness')\n"
    "if VOXEL_Z > 3.0:\n"
    "    print(f'WARNING: VOXEL_Z={VOXEL_Z} µm large — Z rescaling will expand volume significantly')\n"
    "print(f'  Z  voxel: {VOXEL_Z:.4f} um/slice')\n"
    "print(f'  Anisotropy (Z/XY): {VOXEL_Z / voxel_xy:.2f}x')"
)
patch_cell(
    nb0,
    match_fn=lambda s: 'VOXEL_Z' in s and 'Anisotropy' in s and 'voxel_xy' in s,
    old=OLD_B3,
    new=NEW_B3,
    label='B3 VOXEL_Z assert',
)

save(nb0, path0)

print('\nDone. All fixes applied to fix1/.')
