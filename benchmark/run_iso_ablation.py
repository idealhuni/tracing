#!/usr/bin/env python3
"""
Isotropic FMM ablation 배치 실행.

step3_iso_ablation.py만 재실행 (prep_riem.npz 재사용).
결과: benchmark/results/iso_ablation/{stem}.swc

Usage:
    cd /Users/lee/Tracer
    conda run -n tracing python benchmark/run_iso_ablation.py
    conda run -n tracing python benchmark/run_iso_ablation.py --samples neuron2,neuron4
"""
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT       = Path(__file__).parent.resolve()
OUT_DIR    = ROOT / 'methods' / 'ours' / 'output'
RESULTS    = ROOT / 'results' / 'iso_ablation'
STEP3_ISO  = ROOT / 'methods' / 'ours' / 'step3_iso_ablation.py'

SAMPLES = [
    'neuron2',
    'neuron4',
    '1201_01_s06b_L36_Sum_ch2.tif',
    '1201_01_s10mm_ch2.tif',
]


def run_sample(sample: str, force: bool = False) -> bool:
    # output 폴더명은 sample 그대로 (예: '1201_01_s06b_L36_Sum_ch2.tif')
    # results SWC는 .tif 확장자 제거한 stem 사용
    work_dir = OUT_DIR / sample
    src_swc  = work_dir / 'neurons_iso_ablation.swc'
    dst_swc  = RESULTS / f'{sample}.swc'   # ours와 동일 명명: .tif 포함

    RESULTS.mkdir(parents=True, exist_ok=True)

    if not (work_dir / 'prep_riem.npz').exists():
        print(f'[SKIP] {stem}: prep_riem.npz 없음 (step2 먼저 실행 필요)', flush=True)
        return False

    if dst_swc.exists() and not force:
        print(f'[SKIP] {stem}: 이미 존재 ({dst_swc})', flush=True)
        return True

    if src_swc.exists() and not force:
        shutil.copy(src_swc, dst_swc)
        print(f'[COPY] {stem}: {dst_swc}', flush=True)
        return True

    print(f'\n{"="*60}', flush=True)
    print(f'[RUN ] {stem}', flush=True)
    print(f'{"="*60}', flush=True)

    t0 = time.time()
    ret = subprocess.run(
        [sys.executable, str(STEP3_ISO), '--out-dir', str(work_dir)],
        check=False,
    )
    elapsed = time.time() - t0

    if ret.returncode != 0:
        print(f'[FAIL] {stem}  ({elapsed:.0f}s)', flush=True)
        return False

    if not src_swc.exists():
        print(f'[FAIL] {stem}: neurons_iso_ablation.swc 생성 안 됨', flush=True)
        return False

    shutil.copy(src_swc, dst_swc)
    print(f'[DONE] {stem}  ({elapsed:.0f}s)  → {dst_swc}', flush=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', default=','.join(SAMPLES),
                    help='쉼표 구분 샘플 목록')
    ap.add_argument('--force', action='store_true',
                    help='기존 결과 있어도 재실행')
    args = ap.parse_args()

    samples = [s.strip() for s in args.samples.split(',') if s.strip()]
    print(f'Ablation targets: {samples}')
    print(f'Force: {args.force}\n')

    ok = err = 0
    for s in samples:
        if run_sample(s, force=args.force):
            ok += 1
        else:
            err += 1

    print(f'\n완료: {ok}개 성공, {err}개 실패')
    print(f'결과 위치: {RESULTS}')


if __name__ == '__main__':
    main()
