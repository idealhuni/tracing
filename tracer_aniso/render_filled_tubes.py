#!/usr/bin/env python3
"""
render_filled_tubes.py  —  SWC tube rasterizer for FIJI overlay

Rasterizes each SWC segment as a truncated cone using per-node radii,
then alpha-blends onto the raw grayscale stack.

Usage:
    python render_filled_tubes.py                # ds=1, full res (~3-4 GB RAM)
    python render_filled_tubes.py --ds 2         # ds=2, ~500 MB RAM (recommended)
    python render_filled_tubes.py --mask-only    # save label mask only, no raw blend

Output (in output/FN1_01/):
    tube_overlay.tif   -- pre-blended RGB TIFF, open directly in FIJI
    tube_mask.tif      -- uint8 label mask (0=bg,2=axon,3=basal,4=apical,1=soma)
                          use with raw for manual FIJI overlay

FIJI workflow (tube_overlay.tif):
    File → Open → tube_overlay.tif
    Image → Adjust → Brightness/Contrast  (scroll Z slider)

FIJI workflow (tube_mask.tif + raw):
    1. Open stack_preprocessed.tif (raw)
    2. Open tube_mask.tif
    3. mask window → Image → Lookup Tables → pick a colormap
    4. Image → Color → Merge Channels → set opacity to 50%
"""

import argparse, os, sys, time
import numpy as np
import tifffile

HERE   = os.path.dirname(os.path.abspath(__file__))
FN1    = os.path.join(HERE, 'output', 'FN1_01')

parser = argparse.ArgumentParser()
parser.add_argument('--swc',   default=os.path.join(FN1, 'neurons_auto.swc'))
parser.add_argument('--stack', default=os.path.join(FN1, 'stack_preprocessed.tif'))
parser.add_argument('--out',   default=FN1)
parser.add_argument('--ds',    type=int,   default=1,   help='downsample factor (1=full res)')
parser.add_argument('--alpha', type=float, default=0.5, help='tube opacity (0-1)')
parser.add_argument('--mask-only', action='store_true', help='skip raw load, save mask only')
args = parser.parse_args()

# ── Type colors (RGB 0-255) ──────────────────────────────────────────────────
TYPE_COLOR = {
    1: (255, 255,   0),   # soma     → yellow
    2: (255,  60,  60),   # axon     → red
    3: (60,  220,  60),   # basal    → green
    4: ( 80, 120, 255),   # apical   → blue
}
TYPE_NAME = {1:'soma', 2:'axon', 3:'basal', 4:'apical'}

# ── Parse SWC ────────────────────────────────────────────────────────────────
print('Parsing SWC...', flush=True)
soma_r_actual = None
nodes = {}   # id → (type, x_um, y_um, z_um, r_um, parent_id)
with open(args.swc) as f:
    for line in f:
        line = line.strip()
        if not line: continue
        if line.startswith('#'):
            # parse "soma_r_actual=10.77 µm" from header
            if 'soma_r_actual=' in line:
                try:
                    soma_r_actual = float(line.split('soma_r_actual=')[1].split()[0])
                except Exception:
                    pass
            continue
        p = line.split()
        nid, ntype = int(p[0]), int(p[1])
        x, y, z, r = float(p[2]), float(p[3]), float(p[4]), float(p[5])
        par = int(p[6])
        nodes[nid] = (ntype, x, y, z, r, par)

if soma_r_actual is None:
    soma_r_actual = 10.77   # fallback default
print(f'  {len(nodes):,} nodes   soma_r_actual={soma_r_actual:.2f} µm')

# ── Voxel size ───────────────────────────────────────────────────────────────
meta_path = os.path.join(FN1, 'preprocess_meta.npz')
meta      = np.load(meta_path)
voxel_iso = float(meta['voxel_iso'])   # µm/voxel
voxel_ds  = voxel_iso * args.ds
print(f'  voxel_iso={voxel_iso:.4f} µm  ds={args.ds}  voxel_ds={voxel_ds:.4f} µm')

# ── Load raw (optional) ──────────────────────────────────────────────────────
if not args.mask_only:
    print('Loading raw stack...', flush=True)
    raw = tifffile.imread(args.stack)
    if args.ds > 1:
        raw = raw[::args.ds, ::args.ds, ::args.ds]
    NZ, NY, NX = raw.shape
    print(f'  shape={raw.shape}  dtype={raw.dtype}')
    # normalize to uint8
    p_lo = float(np.percentile(raw, 0.5))
    p_hi = float(np.percentile(raw, 99.5))
    raw_u8 = np.clip((raw.astype(np.float32) - p_lo) / (p_hi - p_lo + 1e-8) * 255,
                     0, 255).astype(np.uint8)
    del raw
else:
    # infer shape from file without loading
    with tifffile.TiffFile(args.stack) as tf:
        s = tf.series[0].shape
    # s might be (Z,Y,X)
    NZ, NY, NX = s[0]//args.ds, s[1]//args.ds, s[2]//args.ds
    raw_u8 = None
    print(f'  mask-only mode  inferred shape=({NZ},{NY},{NX})')

shape = (NZ, NY, NX)

# ── Allocate label mask ───────────────────────────────────────────────────────
print('Allocating mask...', flush=True)
mask = np.zeros(shape, dtype=np.uint8)   # 0=background

# ── Core rasterizer ───────────────────────────────────────────────────────────
def rasterize_capsule(mask, xc1, yc1, zc1, r1, xc2, yc2, zc2, r2, label):
    """
    Fill a truncated cone from (xc1,yc1,zc1) to (xc2,yc2,zc2) into mask.
    All coords are in voxels.  xc→X(axis2), yc→Y(axis1), zc→Z(axis0).
    """
    NZ, NY, NX = mask.shape
    margin = max(r1, r2) + 1.5

    xs0 = max(0, int(np.floor(min(xc1, xc2) - margin)))
    xs1 = min(NX-1, int(np.ceil (max(xc1, xc2) + margin)))
    ys0 = max(0, int(np.floor(min(yc1, yc2) - margin)))
    ys1 = min(NY-1, int(np.ceil (max(yc1, yc2) + margin)))
    zs0 = max(0, int(np.floor(min(zc1, zc2) - margin)))
    zs1 = min(NZ-1, int(np.ceil (max(zc1, zc2) + margin)))

    if xs0 > xs1 or ys0 > ys1 or zs0 > zs1:
        return

    xi = np.arange(xs0, xs1+1, dtype=np.float32)
    yi = np.arange(ys0, ys1+1, dtype=np.float32)
    zi = np.arange(zs0, zs1+1, dtype=np.float32)

    # meshgrid with ZYX order to match mask indexing
    ZZ, YY, XX = np.meshgrid(zi, yi, xi, indexing='ij')

    dx, dy, dz = xc2-xc1, yc2-yc1, zc2-zc1
    seg_sq = dx*dx + dy*dy + dz*dz

    if seg_sq < 1e-9:
        dist_sq = (XX-xc1)**2 + (YY-yc1)**2 + (ZZ-zc1)**2
        inside  = dist_sq <= r1*r1
    else:
        t  = ((XX-xc1)*dx + (YY-yc1)*dy + (ZZ-zc1)*dz) / seg_sq
        t  = np.clip(t, 0.0, 1.0)
        rt = (r1 + t*(r2-r1))              # interpolated radius
        cx = xc1 + t*dx
        cy = yc1 + t*dy
        cz = zc1 + t*dz
        dist_sq = (XX-cx)**2 + (YY-cy)**2 + (ZZ-cz)**2
        inside  = dist_sq <= rt*rt

    sub = mask[zs0:zs1+1, ys0:ys1+1, xs0:xs1+1]
    sub[inside] = label

# ── Rasterize all segments ────────────────────────────────────────────────────
print('Rasterizing tubes...', flush=True)
t0 = time.time()

type_counts = {t: 0 for t in TYPE_COLOR}

for nid, (ntype, x, y, z, r, par) in nodes.items():
    # soma: use actual biological radius
    if ntype == 1:
        r_use = soma_r_actual
    else:
        r_use = r

    # convert to voxels
    xv = x / voxel_ds
    yv = y / voxel_ds
    zv = z / voxel_ds
    rv = r_use / voxel_ds

    if par == -1:
        # root node: draw sphere only
        rasterize_capsule(mask, xv, yv, zv, rv, xv, yv, zv, rv, ntype)
        type_counts[ntype] = type_counts.get(ntype, 0) + 1
        continue

    if par not in nodes:
        continue

    ptype, px, py, pz, pr, _ = nodes[par]
    pr_use = soma_r_actual if ptype == 1 else pr

    pxv = px / voxel_ds
    pyv = py / voxel_ds
    pzv = pz / voxel_ds
    prv = pr_use / voxel_ds

    rasterize_capsule(mask, xv, yv, zv, rv, pxv, pyv, pzv, prv, ntype)
    type_counts[ntype] = type_counts.get(ntype, 0) + 1

elapsed = time.time() - t0
print(f'  Done in {elapsed:.1f}s')
for t, cnt in type_counts.items():
    vox = int((mask == t).sum())
    print(f'  type {t} ({TYPE_NAME.get(t,t):6s}): {cnt:4d} segments  {vox:,} filled voxels')

# ── Save label mask ───────────────────────────────────────────────────────────
mask_path = os.path.join(args.out, 'tube_mask.tif')
print(f'\nSaving mask → {mask_path}')
tifffile.imwrite(
    mask_path,
    mask,
    compression='zlib',
    compressionargs={'level': 6},
    metadata={
        'axes': 'ZYX',
        'PhysicalSizeX': voxel_ds, 'PhysicalSizeXUnit': 'µm',
        'PhysicalSizeY': voxel_ds, 'PhysicalSizeYUnit': 'µm',
        'PhysicalSizeZ': voxel_ds, 'PhysicalSizeZUnit': 'µm',
    },
)
print(f'  {os.path.getsize(mask_path)/1e6:.1f} MB')

# ── Composite RGB (raw + tubes at 50% alpha) ─────────────────────────────────
if not args.mask_only:
    print('\nCompositing RGB overlay...', flush=True)
    alpha = args.alpha

    # Build tube RGB layer
    tube_r = np.zeros(shape, np.uint8)
    tube_g = np.zeros(shape, np.uint8)
    tube_b = np.zeros(shape, np.uint8)
    for t, (cr, cg, cb) in TYPE_COLOR.items():
        sel = mask == t
        tube_r[sel] = cr
        tube_g[sel] = cg
        tube_b[sel] = cb

    has_tube = mask > 0

    # Start from grayscale raw
    out_r = raw_u8.copy()
    out_g = raw_u8.copy()
    out_b = raw_u8.copy()

    # Alpha blend only where tube exists
    rf = raw_u8[has_tube].astype(np.float32)
    out_r[has_tube] = np.clip(rf*(1-alpha) + tube_r[has_tube]*alpha, 0, 255).astype(np.uint8)
    out_g[has_tube] = np.clip(rf*(1-alpha) + tube_g[has_tube]*alpha, 0, 255).astype(np.uint8)
    out_b[has_tube] = np.clip(rf*(1-alpha) + tube_b[has_tube]*alpha, 0, 255).astype(np.uint8)

    composite = np.stack([out_r, out_g, out_b], axis=-1)   # (Z,Y,X,3)
    del raw_u8, tube_r, tube_g, tube_b, out_r, out_g, out_b

    ovr_path = os.path.join(args.out, 'tube_overlay.tif')
    print(f'Saving overlay → {ovr_path}')
    tifffile.imwrite(
        ovr_path,
        composite,
        photometric='rgb',
        compression='zlib',
        compressionargs={'level': 6},
        metadata={
            'axes': 'ZYXS',
            'PhysicalSizeX': voxel_ds, 'PhysicalSizeXUnit': 'µm',
            'PhysicalSizeY': voxel_ds, 'PhysicalSizeYUnit': 'µm',
            'PhysicalSizeZ': voxel_ds, 'PhysicalSizeZUnit': 'µm',
        },
    )
    mb = os.path.getsize(ovr_path)/1e6
    print(f'  {mb:.0f} MB  →  {ovr_path}')

print('\nDone.')
print()
print('── FIJI 사용법 ──────────────────────────────────────────')
print('  [바로 보기]  File → Open → tube_overlay.tif')
print('              스크롤로 Z slice 탐색, 색상: red=axon / green=basal / blue=apical')
print()
print('  [채널 조절]  File → Open → tube_mask.tif')
print('              Image → Lookup Tables → 원하는 LUT 선택')
print('              raw와 병합: Image → Color → Merge Channels')
print('────────────────────────────────────────────────────────')
