#!/usr/bin/env python
"""
local_bg_norm.py — Local background normalization experiment

stack_preprocessed.tif 원본을 건드리지 않고,
Gaussian rolling-ball 배경 차감 결과를 stack_preprocessed_lbn.tif 로 저장.

사용법:
    python local_bg_norm.py neuron4
    python local_bg_norm.py 1201_01_s10mm_ch2.tif --sigmas 5 10 20
    python local_bg_norm.py neuron4 --save --save-sigma 10
    python local_bg_norm.py neuron4 --out-dir ../tracer_aniso/output
"""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter

# benchmark output 기준 기본 경로
DEFAULT_OUT_DIR = Path(__file__).parent.parent / 'benchmark' / 'methods' / 'ours' / 'output'
DEFAULT_SIGMAS_UM = [5.0, 10.0, 20.0]


# ── Core: Gaussian rolling-ball background subtraction ──────────────────────

def rolling_ball(stack: np.ndarray, sigma_vox: float) -> np.ndarray:
    """
    배경 = 큰 Gaussian blur (sigma >> 최대 neurite 직경)
    결과 = max(0, 원본 - 배경)  → 음수 clip
    """
    bg = gaussian_filter(stack.astype(np.float32), sigma=float(sigma_vox))
    return np.maximum(stack - bg, 0.0, dtype=np.float32)


def normalize_p99(vol: np.ndarray) -> np.ndarray:
    """p99 clip → [0, 1] float32"""
    pos = vol[vol > 0]
    p99 = float(np.percentile(pos, 99)) if pos.size > 0 else 1.0
    return np.clip(vol / (p99 + 1e-10), 0.0, 1.0).astype(np.float32)


# ── Visualization ────────────────────────────────────────────────────────────

def mip_row(ax_row, vol: np.ndarray, label: str):
    projs = [vol.max(axis=0), vol.max(axis=1), vol.max(axis=2)]
    for col, mip in enumerate(projs):
        ax = ax_row[col]
        pos = mip[mip > 0]
        vmax = float(np.percentile(pos, 99)) if pos.size > 0 else 1.0
        ax.imshow(mip, cmap='gray', vmin=0, vmax=vmax,
                  aspect='auto', interpolation='nearest')
        ax.axis('off')
    ax_row[0].set_ylabel(label, fontsize=9)


def compare_mips(vols_labels, title='', out_path=None):
    """vols_labels: list of (label, vol_float32) tuples"""
    n = len(vols_labels)
    fig, axes = plt.subplots(n, 3, figsize=(15, 3.5 * n),
                             facecolor='#111', subplot_kw={'facecolor': '#111'})
    if n == 1:
        axes = axes[np.newaxis, :]

    plt.rcParams.update({'text.color': 'white', 'axes.labelcolor': 'white'})

    col_titles = ['XY (Z-MIP)', 'XZ (Y-MIP)', 'YZ (X-MIP)']
    for col, ct in enumerate(col_titles):
        axes[0, col].set_title(ct, color='white', fontsize=10)

    for row, (label, vol) in enumerate(vols_labels):
        mip_row(axes[row], vol, label)

    fig.suptitle(title, color='white', fontsize=12)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#111')
        print(f'  plot → {out_path}')
    plt.show()
    plt.close(fig)


# ── Stats helper ─────────────────────────────────────────────────────────────

def stats(vol: np.ndarray, label: str):
    pos = vol[vol > 0]
    pct = 100.0 * pos.size / vol.size
    print(f'  {label:<22s}  max={vol.max():.4f}  '
          f'mean={vol.mean():.5f}  '
          f'fg>0: {pct:.1f}%  '
          f'p99={float(np.percentile(pos, 99)) if pos.size else 0:.4f}')


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Local background normalization')
    ap.add_argument('stem', help='Sample stem (e.g. neuron4)')
    ap.add_argument('--out-dir', type=Path, default=DEFAULT_OUT_DIR,
                    help='Root output dir containing {stem}/ subdirectories')
    ap.add_argument('--sigmas', nargs='+', type=float, default=DEFAULT_SIGMAS_UM,
                    metavar='UM', help='Background σ values in µm (default: 5 10 20)')
    ap.add_argument('--save', action='store_true',
                    help='Save result as stack_preprocessed_lbn.tif')
    ap.add_argument('--save-sigma', type=float, default=None,
                    help='Which σ to save (default: first in --sigmas)')
    args = ap.parse_args()

    sample_dir = Path(args.out_dir) / args.stem
    stack_path = sample_dir / 'stack_preprocessed.tif'
    meta_path  = sample_dir / 'preprocess_meta.npz'

    for p in (stack_path, meta_path):
        if not p.exists():
            raise FileNotFoundError(p)

    meta      = np.load(str(meta_path))
    voxel_iso = float(meta['voxel_iso'])

    print(f'=== {args.stem}  voxel_iso={voxel_iso:.4f} µm ===')
    print(f'Loading {stack_path} ...')
    t0    = time.time()
    stack = tifffile.imread(str(stack_path)).astype(np.float32)
    print(f'  shape={stack.shape}  [{time.time()-t0:.1f}s]')

    # 원본 normalizing (display용)
    stack_n = normalize_p99(stack)
    stats(stack_n, 'original')

    vols = [('original', stack_n)]

    results = {}
    for sigma_um in args.sigmas:
        sigma_vox = sigma_um / voxel_iso
        print(f'  rolling-ball σ={sigma_um}µm = {sigma_vox:.1f}vox ...', end=' ', flush=True)
        t1  = time.time()
        out = rolling_ball(stack, sigma_vox)
        print(f'{time.time()-t1:.1f}s')
        out_n = normalize_p99(out)
        stats(out_n, f'LBN σ={sigma_um}µm')
        vols.append((f'LBN σ={sigma_um}µm', out_n))
        results[sigma_um] = out

    out_png = Path(f'lbn_{args.stem}.png')
    compare_mips(vols,
                 title=f'{args.stem} — Local Background Normalization comparison',
                 out_path=out_png)

    if args.save:
        save_sigma = args.save_sigma if args.save_sigma is not None else args.sigmas[0]
        if save_sigma not in results:
            sigma_vox = save_sigma / voxel_iso
            print(f'  Computing σ={save_sigma}µm for save ...', end=' ', flush=True)
            t1 = time.time()
            results[save_sigma] = rolling_ball(stack, sigma_vox)
            print(f'{time.time()-t1:.1f}s')
        out      = results[save_sigma]
        p999     = float(np.percentile(out[out > 0], 99.9)) if out.max() > 0 else 1.0
        out_norm = np.clip(out / p999, 0.0, 1.0).astype(np.float32)
        save_path = sample_dir / 'stack_preprocessed_lbn.tif'
        tifffile.imwrite(str(save_path), out_norm)
        print(f'  saved → {save_path}')
        print(f'  원본 stack_preprocessed.tif 는 그대로 유지됨')


if __name__ == '__main__':
    main()
