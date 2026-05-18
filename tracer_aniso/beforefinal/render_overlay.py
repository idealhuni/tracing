#!/usr/bin/env python3
"""
Pre-rendered RGB overlay TIFF for FIJI.

Usage:
    python render_overlay.py
    python render_overlay.py --ds 2   # 2x downsample

Output: output/overlay_swc.tif
    Open in FIJI: File → Open
    Channels: Ch1=raw, Ch2=SWC+soma
"""

import argparse, os, time
import numpy as np
import tifffile
from collections import defaultdict
from skimage.draw import line_nd
from scipy.ndimage import binary_dilation

HERE = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--swc',   default=os.path.join(HERE, 'output/neurons_riem.swc'))
parser.add_argument('--stack', default=os.path.join(HERE, 'output/stack_preprocessed.tif'))
parser.add_argument('--prep',  default=os.path.join(HERE, 'output/prep_riem.npz'))
parser.add_argument('--soma',  default=os.path.join(HERE, 'output/tubularity_anchored.npz'))
parser.add_argument('--out',   default=os.path.join(HERE, 'output/overlay_swc.tif'))
parser.add_argument('--ds',    type=int, default=1)
parser.add_argument('--thick', type=int, default=2,  help='line dilation radius (vox)')
args = parser.parse_args()

# ── Load ────────────────────────────────────────────────────
print('Loading stack...', flush=True)
stack     = tifffile.imread(args.stack)
prep      = np.load(args.prep)
voxel_iso = float(prep['voxel_iso'])
ds        = args.ds
if ds > 1:
    stack = stack[::ds, ::ds, ::ds]
voxel_ds = voxel_iso * ds
NZ, NY, NX = stack.shape
print(f'  {stack.shape}  voxel={voxel_ds:.3f} µm')

# Normalize raw → uint8
p_lo = float(np.percentile(stack, 0.5))
p_hi = float(np.percentile(stack, 99.5))
raw_u8 = np.clip((stack.astype(np.float32) - p_lo) / (p_hi - p_lo + 1e-8) * 255,
                 0, 255).astype(np.uint8)
del stack

# Load SWC
swc_nodes = {}
with open(args.swc) as f:
    for line in f:
        if line.startswith('#') or not line.strip(): continue
        p = line.split()
        nid = int(p[0])
        swc_nodes[nid] = dict(x=float(p[2]), y=float(p[3]),
                               z=float(p[4]), parent=int(p[6]))
print(f'  SWC: {len(swc_nodes):,} nodes')

# Load soma mask
soma_mask = None
try:
    soma_data = np.load(args.soma)
    if 'soma_mask' in soma_data:
        sm = soma_data['soma_mask']
        if ds > 1:
            sm = sm[::ds, ::ds, ::ds]
        soma_mask = sm
        print(f'  Soma mask: {soma_mask.sum():,} voxels')
except Exception as e:
    print(f'  Soma mask skipped: {e}')

# ── Tree / Colors ────────────────────────────────────────────
children_map = defaultdict(list)
for nid, n in swc_nodes.items():
    if n['parent'] != -1:
        children_map[n['parent']].append(nid)

primary_ids = sorted(children_map.get(1, []))

def get_primary(nid):
    cur = nid
    while cur in swc_nodes and swc_nodes[cur]['parent'] not in (-1, 1):
        cur = swc_nodes[cur]['parent']
    return cur

import matplotlib.pyplot as plt
cmap = plt.cm.tab20
PRIMARY_RGB = {
    pid: tuple(int(c*255) for c in cmap(i/max(len(primary_ids),1))[:3])
    for i, pid in enumerate(primary_ids)
}

# ── Rasterize SWC ────────────────────────────────────────────
print('Rasterizing SWC...', flush=True)
t0 = time.time()

# Separate R/G/B volumes for color
ovr_r = np.zeros((NZ, NY, NX), np.uint8)
ovr_g = np.zeros((NZ, NY, NX), np.uint8)
ovr_b = np.zeros((NZ, NY, NX), np.uint8)

for nid, n in swc_nodes.items():
    par = n['parent']
    if par == -1 or par not in swc_nodes: continue
    p = swc_nodes[par]

    z1 = int(round(n['z']/voxel_ds)); y1 = int(round(n['y']/voxel_ds)); x1 = int(round(n['x']/voxel_ds))
    z2 = int(round(p['z']/voxel_ds)); y2 = int(round(p['y']/voxel_ds)); x2 = int(round(p['x']/voxel_ds))
    z1,y1,x1 = np.clip([z1,y1,x1],[0,0,0],[NZ-1,NY-1,NX-1])
    z2,y2,x2 = np.clip([z2,y2,x2],[0,0,0],[NZ-1,NY-1,NX-1])

    pid = get_primary(nid)
    r, g, b = PRIMARY_RGB.get(pid, (200,200,200))

    try:
        zz, yy, xx = line_nd((z1,y1,x1), (z2,y2,x2), endpoint=True)
        ovr_r[zz, yy, xx] = r
        ovr_g[zz, yy, xx] = g
        ovr_b[zz, yy, xx] = b
    except Exception:
        pass

# Dilation: thicken lines
if args.thick > 0:
    print(f'  Dilating lines (radius={args.thick})...', flush=True)
    struct = np.ones((1, args.thick*2+1, args.thick*2+1), bool)  # XY only
    mask_skel = ovr_r > 0
    mask_dil  = binary_dilation(mask_skel, structure=struct)
    ring      = mask_dil & ~mask_skel
    # fill ring with nearest color (simple: copy from nearest skeleton voxel)
    from scipy.ndimage import distance_transform_edt
    _, idx = distance_transform_edt(~mask_skel, return_indices=True)
    ovr_r[ring] = ovr_r[tuple(idx[:, ring])]
    ovr_g[ring] = ovr_g[tuple(idx[:, ring])]
    ovr_b[ring] = ovr_b[tuple(idx[:, ring])]

print(f'  Done in {time.time()-t0:.1f}s')

# Soma mask: white overlay
if soma_mask is not None:
    sm = soma_mask[:NZ, :NY, :NX]
    ovr_r[sm] = 255
    ovr_g[sm] = 255
    ovr_b[sm] = 255

# ── Composite ────────────────────────────────────────────────
print('Compositing...', flush=True)

mask = (ovr_r > 0) | (ovr_g > 0) | (ovr_b > 0)

# RGB TIFF: blend raw (gray) with colored overlay
raw_rgb = np.stack([raw_u8, raw_u8, raw_u8], axis=-1)
composite = raw_rgb.copy()
alpha = 0.75
composite[mask, 0] = np.clip(raw_rgb[mask,0]*(1-alpha) + ovr_r[mask]*alpha, 0, 255).astype(np.uint8)
composite[mask, 1] = np.clip(raw_rgb[mask,1]*(1-alpha) + ovr_g[mask]*alpha, 0, 255).astype(np.uint8)
composite[mask, 2] = np.clip(raw_rgb[mask,2]*(1-alpha) + ovr_b[mask]*alpha, 0, 255).astype(np.uint8)
del raw_rgb, ovr_r, ovr_g, ovr_b, raw_u8

# ── Save ────────────────────────────────────────────────────
print(f'Saving {args.out} ...', flush=True)
tifffile.imwrite(
    args.out,
    composite,
    photometric='rgb',
    compression='zlib',
    metadata={'axes':'ZYXS',
              'PhysicalSizeX': voxel_ds,
              'PhysicalSizeY': voxel_ds,
              'PhysicalSizeZ': voxel_ds,
              'PhysicalSizeXUnit':'µm'},
)
mb = os.path.getsize(args.out)/1e6
print(f'  {mb:.0f} MB  →  {args.out}')
print()
print('Open in FIJI:')
print('  File → Open → overlay_swc.tif')
print('  Image → Adjust → Brightness/Contrast')
print('  Scroll Z slider to inspect slices')
