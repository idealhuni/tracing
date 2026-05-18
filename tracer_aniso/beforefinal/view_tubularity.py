"""
Tubularity 볼륨 napari 뷰어

Usage:
    python view_tubularity.py                    # neuron2 (기본)
    python view_tubularity.py FN1_01             # FN1_01
    python view_tubularity.py neuron2 full       # tubularity.npz 전체 해상도 (메모리 ~1GB)
"""
import sys, struct, zlib, io
import numpy as np
import napari

SAMPLE  = sys.argv[1] if len(sys.argv) > 1 else 'neuron2'
FULL    = len(sys.argv) > 2 and sys.argv[2] == 'full'
BASE    = f'output/{SAMPLE}'

# ── full-res tubularity.npz (손상된 zip → 직접 복구) ─────────────────────────
def load_tubularity_full(path):
    """end-of-central-directory 없는 손상 npz에서 T_combined 복구."""
    with open(path, 'rb') as f:
        raw = f.read()

    entries = {}
    offset  = 0
    while True:
        idx = raw.find(b'PK\x03\x04', offset)
        if idx == -1:
            break
        if idx + 30 > len(raw):
            break
        fname_len  = struct.unpack_from('<H', raw, idx + 26)[0]
        extra_len  = struct.unpack_from('<H', raw, idx + 28)[0]
        comp_size  = struct.unpack_from('<I', raw, idx + 18)[0]
        fname      = raw[idx + 30: idx + 30 + fname_len].decode('utf-8', errors='replace')
        data_start = idx + 30 + fname_len + extra_len
        entries[fname] = (data_start, comp_size)
        offset = idx + 1

    print('Entries found:', list(entries.keys()))
    target = 'T_combined.npy'
    if target not in entries:
        target = list(entries.keys())[0]
    start, csize = entries[target]
    print(f'Decompressing {target} ({csize/1e6:.0f} MB compressed)...')

    dec  = zlib.decompressobj(wbits=-15)
    buf  = dec.decompress(raw[start: start + csize])

    # parse npy header
    assert buf[:6] == b'\x93NUMPY', 'Not a valid npy file'
    hlen  = struct.unpack_from('<H', buf, 8)[0]
    hdr   = buf[10: 10 + hlen].decode('latin1')
    import ast, re
    info  = ast.literal_eval(hdr.strip().rstrip(','))
    shape = info['shape']
    dtype = np.dtype(info['descr'])
    data_offset = 10 + hlen
    arr = np.frombuffer(buf[data_offset:], dtype=dtype).reshape(shape)
    return arr.copy()


# ── downsampled prep_riem.npz (항상 사용 가능) ────────────────────────────────
def load_prep_riem(path):
    d = np.load(path)
    return {
        'T_down':      d['T_down'],
        'radius_down': d['radius_down'],
        'edt_down':    d['edt_down'],
        'voxel_down':  float(d['voxel_down']),
    }


# ── 로드 ─────────────────────────────────────────────────────────────────────
if FULL:
    print(f'Loading full-res tubularity from {BASE}/tubularity.npz ...')
    T = load_tubularity_full(f'{BASE}/tubularity.npz')
    voxel = 1.0
    layers = [('T_combined (full)', T, 'image', 'hot', (0, T.max()))]
else:
    print(f'Loading downsampled prep_riem from {BASE}/prep_riem.npz ...')
    data  = load_prep_riem(f'{BASE}/prep_riem.npz')
    voxel = data['voxel_down']
    layers = [
        ('T_down',      data['T_down'],      'image', 'hot',    (0, 1)),
        ('radius_down', data['radius_down'],  'image', 'turbo',  (0, data['radius_down'].max())),
        ('edt_down',    data['edt_down'],     'image', 'viridis',(0, data['edt_down'].max())),
    ]

# ── napari ────────────────────────────────────────────────────────────────────
viewer = napari.Viewer(title=f'Tubularity — {SAMPLE}')
for name, arr, kind, cmap, clim in layers:
    viewer.add_image(
        arr,
        name        = name,
        colormap    = cmap,
        contrast_limits = clim,
        scale       = (voxel,) * arr.ndim,
        blending    = 'additive',
    )
    print(f'  {name}: {arr.shape}  [{arr.min():.3f}, {arr.max():.3f}]')

print(f'\nvoxel size: {voxel:.3f} µm')
print('napari 열림 — Ctrl+G: 격자, Ctrl+D: 3D 뷰')
napari.run()
