import numpy as np

NPZ_PATH         = 'output/neuron2/tubularity_anchored.npz'
META_PATH        = 'output/neuron2/preprocess_meta.npz'
BORDER_FADE_Z_UM = 5.0

meta      = np.load(META_PATH)
voxel_iso = float(meta['voxel_iso'])

d = np.load(NPZ_PATH)
T = d['T_combined'].astype(np.float32)
NZ = T.shape[0]

bz = max(1, int(round(BORDER_FADE_Z_UM / voxel_iso)))
bz = min(bz, NZ // 4)
fade = np.ones(NZ, dtype=np.float32)
ramp = np.linspace(0.0, 1.0, bz + 1)[1:]
fade[:bz]  = ramp
fade[-bz:] = ramp[::-1]
T *= fade[:, np.newaxis, np.newaxis]

print(f'voxel_iso={voxel_iso:.4f}  bz={bz} vox ({BORDER_FADE_Z_UM} um)')
print(f'T shape={T.shape}  range={T.min():.4f}-{T.max():.4f}')

np.savez_compressed(NPZ_PATH,
    T_combined   = T,
    I_OOF_raw    = d['I_OOF_raw'],
    orient_field = d['orient_field'],
    radius_map   = d['radius_map'],
    scale_idx    = d['scale_idx'],
    radii        = d['radii'],
    voxel_iso    = d['voxel_iso'],
    soma_mask    = d['soma_mask'],
)
print(f'Saved: {NPZ_PATH}')
