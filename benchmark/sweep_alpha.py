#!/usr/bin/env python3
"""
ALPHA sweep for noisy in-vivo samples (conservative mode).

Usage:
    python sweep_alpha.py                        # 기본: 4개 샘플, alpha 5~11
    python sweep_alpha.py --alphas 6 7 8 9      # alpha 값 직접 지정
    python sweep_alpha.py --samples neuron2 neuron4
    python sweep_alpha.py --step 0.5            # 5.0~11.0 0.5 간격

결과: benchmark/results/alpha_sweep/sweep_results.csv + 콘솔 테이블
"""
import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT       = Path(__file__).parent
STEP3_PY   = ROOT / 'methods' / 'ours' / 'step3_auto.py'
OUT_BASE   = ROOT / 'methods' / 'ours' / 'output'
GOLD_DIR   = ROOT / 'data' / 'gold_standard'
SWEEP_DIR  = ROOT / 'results' / 'alpha_sweep'
SWEEP_DIR.mkdir(parents=True, exist_ok=True)

NOISY_SAMPLES = [
    'neuron2',
    'neuron4',
    '1201_01_s06b_L36_Sum_ch2.tif',
    '1201_01_s10mm_ch2.tif',
]


# ── 평가 헬퍼 ────────────────────────────────────────────────────────────────

def read_pixelsize(stem):
    for name in [f'{stem}.pixelsize.txt', f'{stem}.tif.pixelsize.txt']:
        p = GOLD_DIR / name
        if p.exists():
            m = re.search(r'Voxel size:\s*([\d.]+)x[\d.]+x([\d.]+)', p.read_text(), re.I)
            if m:
                return float(m.group(1)), float(m.group(2))
    return 1.0, 1.0


def load_swc_um(path, scale_xy=1.0, scale_z=1.0):
    nodes = []
    for line in Path(path).read_text().splitlines():
        if line.startswith('#') or not line.strip():
            continue
        p = line.split()
        if len(p) < 7:
            continue
        nodes.append([float(p[2]) * scale_xy,
                      float(p[3]) * scale_xy,
                      float(p[4]) * scale_z])
    return np.array(nodes, dtype=np.float32) if nodes else np.empty((0, 3), dtype=np.float32)


def compute_f1(auto_pts, gold_pts, thr=2.0):
    if len(auto_pts) == 0:
        return dict(f1=0.0, precision=0.0, recall=0.0, esa=999.0, n_auto=0)
    d_a2g, _ = cKDTree(gold_pts).query(auto_pts)
    d_g2a, _ = cKDTree(auto_pts).query(gold_pts)
    p   = float((d_a2g <= thr).mean())
    r   = float((d_g2a <= thr).mean())
    f1  = 2 * p * r / (p + r + 1e-8)
    esa = float((d_a2g.mean() + d_g2a.mean()) / 2)
    return dict(f1=f1, precision=p, recall=r, esa=esa, n_auto=len(auto_pts))


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', nargs='+', default=NOISY_SAMPLES)
    ap.add_argument('--alphas',  nargs='+', type=float, default=None,
                    help='ALPHA 값 직접 지정 (예: --alphas 6 7 8 9)')
    ap.add_argument('--step',    type=float, default=1.0,
                    help='--alphas 미지정 시 5~11 범위에서 간격 (기본 1.0)')
    ap.add_argument('--min-alpha', type=float, default=5.0)
    ap.add_argument('--max-alpha', type=float, default=11.0)
    args = ap.parse_args()

    if args.alphas:
        alphas = sorted(args.alphas)
    else:
        alphas = list(np.arange(args.min_alpha, args.max_alpha + 1e-9, args.step))
        alphas = [round(a, 2) for a in alphas]

    samples = args.samples
    print(f'ALPHA sweep: {alphas}')
    print(f'Samples   : {samples}')
    print(f'총 실행 수 : {len(alphas) * len(samples)}\n')

    # gold standard 로드 (반복 방지)
    gold_cache = {}
    for stem in samples:
        gp = GOLD_DIR / f'{stem}.swc'
        if not gp.exists():
            gp = GOLD_DIR / f'{stem}.tif.swc' if not stem.endswith('.tif') else None
        if gp is None or not gp.exists():
            print(f'[WARN] gold SWC 없음: {stem}')
            gold_cache[stem] = None
            continue
        vxy, vz = read_pixelsize(stem)
        gold_cache[stem] = (load_swc_um(gp, vxy, vz), vxy, vz)

    rows = []
    t_total = time.time()

    for alpha in alphas:
        for stem in samples:
            out_dir = OUT_BASE / stem
            if not (out_dir / 'prep_riem.npz').exists():
                print(f'[SKIP] prep_riem.npz 없음: {stem}  (step0~2 먼저 실행 필요)')
                continue

            # sentinel 삭제
            sentinel = out_dir / 'neurons_auto.swc'
            sentinel.unlink(missing_ok=True)

            # step3 실행
            t0 = time.time()
            try:
                subprocess.run(
                    [sys.executable, str(STEP3_PY),
                     '--out-dir', str(out_dir),
                     '--alpha',   str(alpha)],
                    check=True, capture_output=True
                )
            except subprocess.CalledProcessError as e:
                print(f'[ERR] alpha={alpha} {stem}: {e.stderr.decode()[-200:]}')
                continue
            elapsed = time.time() - t0

            # 결과 SWC 저장 (alpha별 디렉터리)
            alpha_dir = SWEEP_DIR / f'alpha_{alpha:.1f}'
            alpha_dir.mkdir(exist_ok=True)
            result_swc = alpha_dir / f'{stem}.swc'
            if sentinel.exists():
                shutil.copy(sentinel, result_swc)

            # 평가
            if gold_cache.get(stem) is None or not result_swc.exists():
                print(f'  alpha={alpha:5.1f}  {stem:35s}  gold 없음 또는 SWC 생성 실패')
                continue

            gold_pts, vxy, vz = gold_cache[stem]
            auto_pts = load_swc_um(result_swc)   # ours는 이미 µm
            m = compute_f1(auto_pts, gold_pts)

            rows.append(dict(alpha=alpha, sample=stem, **m, elapsed=elapsed))
            print(f'  alpha={alpha:5.1f}  {stem:35s}'
                  f'  F1={m["f1"]:.3f}  P={m["precision"]:.3f}'
                  f'  R={m["recall"]:.3f}  ESA={m["esa"]:6.2f}'
                  f'  n={m["n_auto"]:5d}  ({elapsed:.0f}s)')

    # ── 결과 저장 및 요약 ────────────────────────────────────────────────────
    if not rows:
        print('\n결과 없음 (prep_riem.npz가 있는 샘플 없음)')
        return

    df = pd.DataFrame(rows)
    csv_path = SWEEP_DIR / 'sweep_results.csv'
    df.to_csv(csv_path, index=False)
    print(f'\n결과 저장: {csv_path}  (총 {time.time()-t_total:.0f}s)')

    # 샘플별 최적 alpha
    print('\n' + '='*70)
    print('샘플별 최적 ALPHA (F1 기준)')
    print('='*70)
    for stem in samples:
        sub = df[df['sample'] == stem]
        if sub.empty:
            continue
        best = sub.loc[sub['f1'].idxmax()]
        print(f'  {stem:35s}  best_alpha={best["alpha"]:5.1f}'
              f'  F1={best["f1"]:.3f}  P={best["precision"]:.3f}'
              f'  R={best["recall"]:.3f}  ESA={best["esa"]:.2f}')

    # alpha별 전체 평균
    print('\nalpha별 전체 평균 F1:')
    summary = df.groupby('alpha')[['f1', 'precision', 'recall', 'esa']].mean().round(3)
    print(summary.to_string())

    # 최적 pivot 테이블
    print('\nF1 pivot (sample × alpha):')
    pivot = df.pivot_table(index='sample', columns='alpha', values='f1').round(3)
    print(pivot.to_string())


if __name__ == '__main__':
    main()
