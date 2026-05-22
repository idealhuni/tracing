#!/usr/bin/env python3
"""
Z unsharp mask 효과 미리보기
stack_preprocessed.tif에 적용 후 XY/XZ/YZ MIP 비교
"""
import argparse
import numpy as np
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from pathlib import Path

def z_unsharp(stack, sigma, alpha):
    z_blur = gaussian_filter1d(stack.astype(np.float32), sigma=sigma, axis=0)
    return np.clip(stack + alpha * (stack - z_blur), 0, None)

def mip_row(axes, vol, label, vmax=None):
    mips = [vol.max(axis=0), vol.max(axis=1), vol.max(axis=2)]
    titles = ['XY (Z-MIP)', 'XZ (Y-MIP)', 'YZ (X-MIP)']
    for ax, mip, title in zip(axes, mips, titles):
        vm = vmax or float(np.percentile(mip[mip > 0], 99)) if mip.max() > 0 else 1.0
        ax.imshow(mip, cmap='hot', vmin=0, vmax=vm, aspect='auto', interpolation='nearest')
        ax.set_title(title, fontsize=8, color='white')
        ax.axis('off')
    # 행 레이블을 이미지 위에 크게 표시
    axes[0].text(0.02, 0.97, label, transform=axes[0].transAxes,
                 fontsize=13, fontweight='bold', color='cyan',
                 va='top', ha='left',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stem', default='1201_01_s10mm_ch2.tif')
    ap.add_argument('--sigma', type=float, default=1.5)
    ap.add_argument('--alphas', nargs='+', type=float, default=[0.3, 0.5, 0.8])
    args = ap.parse_args()

    out_dir = Path('benchmark/methods/ours/output') / args.stem
    stack_path = out_dir / 'stack_preprocessed.tif'
    meta = np.load(str(out_dir / 'preprocess_meta.npz'))
    voxel_iso = float(meta['voxel_iso'])

    print(f'Loading {stack_path} ...')
    stack = tifffile.imread(str(stack_path)).astype(np.float32)
    print(f'  shape={stack.shape}  voxel_iso={voxel_iso:.4f}µm')

    # sigma를 voxel 단위로 (voxel_z / voxel_iso 기반)
    sigma_vox = args.sigma
    print(f'  sigma={sigma_vox:.2f} vox ({sigma_vox*voxel_iso:.3f}µm)')

    rows = [('original', stack)] + [(f'α={a}', z_unsharp(stack, sigma_vox, a)) for a in args.alphas]

    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(15, 3.5*n), facecolor='#111',
                             subplot_kw={'facecolor': '#111'})
    plt.rcParams.update({'text.color': 'white'})
    if n == 1:
        axes = axes[np.newaxis, :]

    # 각 row 독립적으로 p99.5 정규화 (밝기 차이 제거, 구조 비교용)
    for row_axes, (label, vol) in zip(axes, rows):
        mip_row(row_axes, vol, label, vmax=None)

    fig.suptitle(f'{args.stem} — Z unsharp mask (σ={sigma_vox:.1f}vox)', color='white', fontsize=11)
    plt.tight_layout()

    out_png = Path(f'benchmark/z_sharp_preview_{args.stem}.png')
    plt.savefig(str(out_png), dpi=150, bbox_inches='tight', facecolor='#111')
    print(f'저장: {out_png}')
    plt.close()

if __name__ == '__main__':
    main()
