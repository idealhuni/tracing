#!/usr/bin/env python3
"""
Napari neuron inspection viewer — improved.

Usage:
    python inspect_napari.py
    python inspect_napari.py --swc output/neurons_riem.swc

Controls:
    2 / 3       : 2D ↔ 3D view
    scroll      : Z slice (2D)
    Ctrl+drag   : rotate (3D)
    V           : toggle layer visibility
    Left panel  : toggle individual primary branches
"""

import argparse, os
import numpy as np
import tifffile
import napari
from collections import defaultdict

# ── Config ──────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--swc',   default=os.path.join(HERE, 'output/neurons_riem.swc'))
parser.add_argument('--stack', default=os.path.join(HERE, 'output/stack_preprocessed.tif'))
parser.add_argument('--soma',  default=os.path.join(HERE, 'output/soma.npz'))
parser.add_argument('--prep',  default=os.path.join(HERE, 'output/prep_riem.npz'))
args = parser.parse_args()

# ── Load ────────────────────────────────────────────────────
print('Loading stack...', flush=True)
stack     = tifffile.imread(args.stack)
voxel_iso = float(np.load(args.prep)['voxel_iso'])
NZ, NY, NX = stack.shape
print(f'  {stack.shape}  voxel={voxel_iso:.4f} µm')

print('Loading SWC...', flush=True)
swc_nodes = {}
with open(args.swc) as f:
    for line in f:
        if line.startswith('#') or not line.strip(): continue
        p = line.split()
        nid = int(p[0])
        swc_nodes[nid] = dict(
            type=int(p[1]), x=float(p[2]), y=float(p[3]),
            z=float(p[4]), r=float(p[5]), parent=int(p[6]))
print(f'  {len(swc_nodes):,} nodes')

print('Loading soma mesh...', flush=True)
soma_data   = np.load(args.soma)
sv_local    = soma_data['mesh_verts'].astype(np.float64)
sf          = soma_data['mesh_faces'].astype(np.int32)
soma_cv     = soma_data['soma_centroid_vox']
voxel_iso_s = float(soma_data['voxel_iso'])
sv_global   = sv_local + (soma_cv * voxel_iso_s - sv_local.mean(axis=0))
sv_vox      = sv_global / voxel_iso_s   # (V,3) [z,y,x] voxels

# ── Tree structure ───────────────────────────────────────────
children_map = defaultdict(list)
for nid, n in swc_nodes.items():
    if n['parent'] != -1:
        children_map[n['parent']].append(nid)

primary_ids = sorted(children_map.get(1, []))

# Primary ancestor lookup
def get_primary_ancestor(nid):
    cur = nid
    while cur in swc_nodes and swc_nodes[cur]['parent'] not in (-1, 1):
        cur = swc_nodes[cur]['parent']
    return cur

# ── Branch segment collection ────────────────────────────────
# Collect continuous path segments (branch point → branch point/tip)
# shape_type='path' renders as connected polyline → no gaps

def px(n):
    """Node → voxel coords [z, y, x]"""
    nd = swc_nodes[n]
    return [nd['z']/voxel_iso, nd['y']/voxel_iso, nd['x']/voxel_iso]

def collect_branch_segments(root, panc):
    """Recursively collect branch segments for this subtree."""
    segments = []
    def _trace(node, current_seg):
        current_seg.append(node)
        kids = children_map.get(node, [])
        if not kids:   # tip
            segments.append(list(current_seg))
        elif len(kids) == 1:
            _trace(kids[0], current_seg)
        else:          # branch point — close current, start new per child
            segments.append(list(current_seg))
            for kid in kids:
                _trace(kid, [node])   # new segment starts at branch point
    _trace(root, [])
    return segments

# Per-primary: (segments, color)
import matplotlib.pyplot as plt
cmap   = plt.cm.tab20
N_prim = max(len(primary_ids), 1)
PRIMARY_COLORS = {
    pid: tuple(int(c*255) for c in cmap(i/N_prim)[:3])
    for i, pid in enumerate(primary_ids)
}

# ── Napari viewer ────────────────────────────────────────────
print('Building viewer...', flush=True)
viewer = napari.Viewer(title='Neuron Inspection')

# 1. Raw image
p_lo = float(np.percentile(stack,  1))
p_hi = float(np.percentile(stack, 99.5))
viewer.add_image(
    stack,
    name='Raw',
    colormap='gray',
    contrast_limits=[p_lo, p_hi],
    blending='additive',
    gamma=0.85,
    visible=True,
)

# 2. Soma mesh
viewer.add_surface(
    (sv_vox, sf),
    name='Soma',
    colormap='bop orange',
    opacity=0.65,
    shading='smooth',
    blending='additive',
)

# 3. Per-primary path layers (toggleable)
for pid in primary_ids:
    segs  = collect_branch_segments(pid, pid)
    if not segs: continue

    paths_px = [np.array([px(n) for n in seg]) for seg in segs if len(seg) >= 2]
    if not paths_px: continue

    rgb   = PRIMARY_COLORS[pid]
    col   = [f'#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}'] * len(paths_px)
    widths = [2.5] * len(paths_px)

    viewer.add_shapes(
        paths_px,
        shape_type='path',
        name=f'Primary {pid}',
        edge_color=col,
        edge_width=widths,
        face_color='transparent',
        opacity=0.95,
        blending='additive',
    )

# 4. Soma center marker
soma_n = swc_nodes[1]
viewer.add_points(
    [[soma_n['z']/voxel_iso, soma_n['y']/voxel_iso, soma_n['x']/voxel_iso]],
    name='Soma center',
    size=12,
    face_color='white',
    symbol='star',
    opacity=1.0,
    blending='additive',
)

# 5. Camera: center on soma
viewer.camera.center = (
    soma_n['z'] / voxel_iso,
    soma_n['y'] / voxel_iso,
    soma_n['x'] / voxel_iso,
)

# ── Status bar: show node info on hover ─────────────────────
# Build spatial index for fast nearest-node lookup
from scipy.spatial import cKDTree

node_ids  = list(swc_nodes.keys())
node_coords = np.array([
    [swc_nodes[n]['z']/voxel_iso,
     swc_nodes[n]['y']/voxel_iso,
     swc_nodes[n]['x']/voxel_iso]
    for n in node_ids
])
node_tree = cKDTree(node_coords)

@viewer.mouse_move_callbacks.append
def on_mouse_move(viewer, event):
    try:
        pos = np.array(event.position[:3])
        dist, idx = node_tree.query(pos)
        if dist < 8:   # within 8 voxels
            nid = node_ids[idx]
            n   = swc_nodes[nid]
            panc = get_primary_ancestor(nid)
            bo   = 0
            cur  = nid
            while cur in swc_nodes and swc_nodes[cur]['parent'] != -1:
                kids = children_map.get(swc_nodes[cur]['parent'], [])
                if len(kids) >= 2: bo += 1
                cur = swc_nodes[cur]['parent']
            viewer.status = (
                f'node={nid}  primary={panc}  order={bo}  '
                f'r={n["r"]:.3f} µm  '
                f'pos=({n["x"]:.1f},{n["y"]:.1f},{n["z"]:.1f}) µm'
            )
    except Exception:
        pass

print()
print('=' * 52)
print(f'  Layers: Raw + Soma + {len(primary_ids)} primaries')
print()
print('  Controls:')
print('    2 / 3          : 2D ↔ 3D view')
print('    Scroll         : Z slice')
print('    Ctrl+drag      : rotate 3D')
print('    Left panel     : toggle primary layers')
print('    Hover over SWC : node info in status bar')
print('=' * 52)

napari.run()
