#!/usr/bin/env python3
"""
SWC MIP overlay visualizer.

Usage:
  python viz_mip.py                          # all samples with gold
  python viz_mip.py neuron2 neuron4          # specific samples
  python viz_mip.py neuron2 --mode prep      # preprocessed image
  python viz_mip.py neuron2 --out /tmp       # custom output dir
  python viz_mip.py neuron2 --show           # open window after saving
  python viz_mip.py neuron2 --lw 1.2 --dpi 200
"""
import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib.collections import LineCollection

# ── Paths ────────────────────────────────────────────────────────────────────
BENCH_DIR   = Path(__file__).parent.resolve()
GOLD_DIR    = BENCH_DIR / 'data' / 'gold_standard'
RESULTS_DIR = BENCH_DIR / 'results'
OUT_DIR     = BENCH_DIR / 'methods' / 'ours' / 'output'
IMAGE_DIR   = BENCH_DIR / 'data' / 'images'

COLORS = {
    'Gold' : '#f5c518',
    'Ours' : '#4c9be8',
    'Vaa3D': '#e87b4c',
}


# ── SWC helpers ──────────────────────────────────────────────────────────────

def read_pixelsize(stem):
    txt = (GOLD_DIR / f'{stem}.pixelsize.txt').read_text()
    m = re.search(r'Voxel size:\s*([\d.]+)x[\d.]+x([\d.]+)', txt, re.I)
    if not m:
        raise ValueError(f'Cannot parse pixelsize from {stem}.pixelsize.txt')
    return float(m.group(1)), float(m.group(2))   # vxy, vz (µm)


def load_swc(path, scale_xy=1.0, scale_z=1.0):
    """Return list of (x1,y1,z1, x2,y2,z2) µm segments."""
    nodes = {}
    for line in Path(path).read_text().splitlines():
        if line.startswith('#') or not line.strip():
            continue
        p = line.split()
        if len(p) < 7:
            continue
        nid = int(p[0])
        nodes[nid] = (float(p[2]) * scale_xy,
                      float(p[3]) * scale_xy,
                      float(p[4]) * scale_z,
                      int(p[6]))
    segs = []
    for nid, (x, y, z, pid) in nodes.items():
        if pid == -1 or pid not in nodes:
            continue
        px, py, pz, _ = nodes[pid]
        segs.append((x, y, z, px, py, pz))
    return segs


def _lc(segs, ax0, ax1, vox0, vox1, color, lw, alpha):
    lines = [[(s[ax0]/vox0, s[ax1]/vox1),
               (s[ax0+3]/vox0, s[ax1+3]/vox1)] for s in segs]
    return LineCollection(lines, colors=color, linewidths=lw, alpha=alpha)


# ── MIP figure ───────────────────────────────────────────────────────────────

def build_mip_figure(stem, mode='orig', lw=0.8, dpi=150, out_dir=None, show=False):
    """
    mode='orig'  — raw image from data/images/
    mode='prep'  — stack_preprocessed.tif from methods/ours/output/
    """
    px_txt = GOLD_DIR / f'{stem}.pixelsize.txt'
    has_pixelsize = px_txt.exists()
    vxy, vz = read_pixelsize(stem) if has_pixelsize else (1.0, 1.0)

    # ── Image ───────────────────────────────────────────────────
    if mode == 'prep':
        img_path = OUT_DIR / stem / 'stack_preprocessed.tif'
        meta_npz = OUT_DIR / stem / 'preprocess_meta.npz'
        if not img_path.exists():
            sys.exit(f'[{stem}] stack_preprocessed.tif not found at {img_path}')
        viso   = float(np.load(str(meta_npz))['voxel_iso']) if meta_npz.exists() else 1.0
        px_xy  = viso
        px_z   = viso
        suffix = 'prep'
    else:
        img_path = IMAGE_DIR / f'{stem}.tif'
        if not img_path.exists():
            sys.exit(f'[{stem}] image not found at {img_path}')
        px_xy  = 1.0   # image is in native pixels; SWC is scaled to µm then divided
        px_z   = 1.0
        suffix = 'orig'

    print(f'[{stem}] loading {img_path.name} ...', flush=True)
    stack  = tifffile.imread(str(img_path)).astype(np.float32)
    if stack.ndim == 4:          # TCZYX or similar — take first channel
        stack = stack[0] if stack.shape[0] < stack.shape[1] else stack[:, 0]
    mip_xy = stack.max(axis=0)   # Z-MIP  → shape (Y, X)
    mip_xz = stack.max(axis=1)   # Y-MIP  → shape (Z, X)
    mip_yz = stack.max(axis=2)   # X-MIP  → shape (Z, Y)

    # ── SWC sources ─────────────────────────────────────────────
    # Each entry: (label, path, color, sxy, sz)
    # After load_swc applies scale, coords are divided by (px_xy, px_z) → image pixels.
    #
    # orig mode: image is native pixels.
    #   Gold SWC = pixel coords → keep as-is (scale=1.0)
    #   Ours SWC = µm coords   → convert to pixels via 1/vxy
    #   Vaa3D SWC= pixel coords → keep as-is (scale=1.0)
    #   px_xy=1.0, px_z=1.0
    #
    # prep mode: image is viso µm/pixel isotropic.
    #   Gold SWC = pixel coords → scale to µm via vxy/vz
    #   Ours SWC = µm coords   → keep as-is (scale=1.0)
    #   Vaa3D SWC= pixel coords → scale to µm via vxy/vz
    #   px_xy=viso, px_z=viso
    if mode == 'prep':
        swc_sources = [
            ('Gold',  GOLD_DIR / f'{stem}.swc',               '#f5c518', vxy, vz ),
            ('Ours',  RESULTS_DIR / 'ours'  / f'{stem}.swc',  '#4c9be8', 1.0, 1.0),
            ('Vaa3D', RESULTS_DIR / 'vaa3d' / f'{stem}.swc',  '#e87b4c', vxy, vz ),
        ]
    else:
        swc_sources = [
            ('Gold',  GOLD_DIR / f'{stem}.swc',               '#f5c518', 1.0,              1.0           ),
            ('Ours',  RESULTS_DIR / 'ours'  / f'{stem}.swc',  '#4c9be8', 1/vxy if vxy else 1.0, 1/vz if vz else 1.0),
            ('Vaa3D', RESULTS_DIR / 'vaa3d' / f'{stem}.swc',  '#e87b4c', 1.0,              1.0           ),
        ]

    # Load SWC (skip missing)
    loaded = []
    for label, path, color, sxy, sz in swc_sources:
        if not Path(path).exists():
            print(f'  [{stem}] {label}: not found ({Path(path).name}), skipped')
            loaded.append((label, None, color))
        else:
            segs = load_swc(path, sxy, sz)
            print(f'  [{stem}] {label}: {len(segs):,} segments')
            loaded.append((label, segs, color))

    # ── Layout ──────────────────────────────────────────────────
    n_rows   = 1 + len(loaded)
    fig, axes = plt.subplots(n_rows, 3,
                             figsize=(15, 4 * n_rows),
                             gridspec_kw={'hspace': 0.04, 'wspace': 0.04})

    proj_titles = ['XY (Z-MIP)', 'XZ (Y-MIP)', 'YZ (X-MIP)']
    mips        = [mip_xy, mip_xz, mip_yz]

    def _vmax(mip):
        pos = mip[mip > 0]
        return float(np.percentile(pos, 99)) if pos.size else 1.0

    # Row 0: raw image
    for col, (mip, title) in enumerate(zip(mips, proj_titles)):
        ax = axes[0, col]
        ax.imshow(mip, cmap='gray', origin='upper',
                  aspect='auto' if col > 0 else 'equal', vmax=_vmax(mip))
        ax.set_title(title, fontsize=11)
        if col == 0:
            ax.set_ylabel('Image', fontsize=11, color='white')
        ax.axis('off')

    # Remaining rows: SWC overlays
    for row, (label, segs, color) in enumerate(loaded, start=1):
        for col, mip in enumerate(mips):
            ax = axes[row, col]
            ax.imshow(mip, cmap='gray', origin='upper',
                      aspect='auto' if col > 0 else 'equal', vmax=_vmax(mip))
            if segs:
                if col == 0:
                    # XY: axis-0=x→X pixel, axis-1=y→Y pixel (segments: x,y,z,px,py,pz)
                    lc = _lc(segs, 0, 1, px_xy, px_xy, color, lw, 0.9)
                elif col == 1:
                    # XZ: horizontal=x, vertical=z
                    lc = _lc(segs, 0, 2, px_xy, px_z, color, lw, 0.9)
                else:
                    # YZ: horizontal=y, vertical=z
                    lc = _lc(segs, 1, 2, px_xy, px_z, color, lw, 0.9)
                ax.add_collection(lc)
            if col == 0:
                ax.set_ylabel(label, fontsize=11, color=color)
            ax.axis('off')

    title_str = f'{stem}  [{suffix}]  Gold(yellow) / Ours(blue) / Vaa3D(orange)'
    plt.suptitle(title_str, fontsize=12, y=1.005)

    # ── Save ────────────────────────────────────────────────────
    if out_dir is None:
        out_dir = BENCH_DIR / 'evaluation'
    out_path = Path(out_dir) / f'mip_{stem}_{suffix}.png'
    plt.savefig(str(out_path), dpi=dpi, bbox_inches='tight')
    print(f'  Saved: {out_path}')

    if show:
        plt.show()
    plt.close(fig)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='SWC MIP overlay visualizer')
    ap.add_argument('samples', nargs='*',
                    help='Sample stems. Default: all samples with gold SWC.')
    ap.add_argument('--mode', choices=['orig', 'prep'], default='orig',
                    help='Image source: orig=raw TIFF, prep=preprocessed (default: orig)')
    ap.add_argument('--out', default=None,
                    help='Output directory (default: benchmark/evaluation/)')
    ap.add_argument('--lw', type=float, default=0.8,
                    help='SWC line width (default: 0.8)')
    ap.add_argument('--dpi', type=int, default=150,
                    help='Output DPI (default: 150)')
    ap.add_argument('--show', action='store_true',
                    help='Show figure window after saving')
    args = ap.parse_args()

    if args.samples:
        stems = args.samples
    else:
        # Auto-discover: samples present in results/ours/ AND data/gold_standard/
        ours_stems = {p.stem for p in (RESULTS_DIR / 'ours').glob('*.swc')}
        gold_stems = {p.stem for p in GOLD_DIR.glob('*.swc')}
        stems = sorted(ours_stems & gold_stems)
        print(f'Auto-discovered samples: {stems}')

    for stem in stems:
        build_mip_figure(
            stem,
            mode   = args.mode,
            lw     = args.lw,
            dpi    = args.dpi,
            out_dir= args.out,
            show   = args.show,
        )


if __name__ == '__main__':
    main()
