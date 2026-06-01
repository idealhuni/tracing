# Tracer — Neuron Tracing Pipeline

## 폴더 구조

```
tracer_portable/
├── run_batch.py              # 메인 실행 스크립트
├── environment.yml           # conda 환경 정의
├── methods/ours/             # 파이프라인 스크립트 (step0~3)
├── bin/
│   ├── macos_arm64/          # macOS Apple Silicon 바이너리 (포함됨)
│   └── windows_x64/          # Windows 바이너리 (빌드 필요, 아래 참고)
├── data/
│   ├── images/               # 입력 TIF 이미지를 여기에
│   └── gold_standard/        # 평가용 gold SWC (선택사항)
└── results/ours/             # 출력 SWC 저장 위치
```

## 설치

```bash
conda env create -f environment.yml
conda activate tracer
```

## 실행

```bash
# data/images/ 에 이미지 넣은 후
python run_batch.py --method ours --samples "파일명.tif"
```

## FileHFM 바이너리 (플랫폼별)

### macOS (Apple Silicon) — 포함됨
`bin/macos_arm64/FileHFM_Riemann3` 이미 포함. 바로 실행 가능.

### Windows (WSL2 사용 권장)
1. WSL2 Ubuntu 설치
2. 아래 명령으로 Linux 바이너리 빌드:
```bash
sudo apt install cmake build-essential git
git clone https://github.com/Mirebeau/HamiltonFastMarching
cd HamiltonFastMarching && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make FileHFM_Riemann3 FileHFM_Isotropic3
```
3. 빌드된 바이너리를 `bin/windows_x64/` 에 복사
4. `methods/ours/FileHFM_binary_dir.txt` 를 아래로 수정:
```
../../bin/windows_x64
```

### Linux
WSL2와 동일한 빌드 방법 사용. 바이너리를 `bin/linux_x64/` 에 복사.

## 파이프라인 단계

| Step | 스크립트 | 입력 → 출력 |
|------|---------|------------|
| step0 | step0_preprocess.py | TIF → stack_preprocessed.tif |
| step1 | step1_tubularity_oof.py | stack_preprocessed → tubularity.npz |
| step1b | step1b_soma.py | tubularity → soma.npz |
| step2 | step2_prep_aniso.py | soma + tubularity → prep_riem.npz |
| step3 | step3_auto.py | prep_riem → neurons_auto.swc |
