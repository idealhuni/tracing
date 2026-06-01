# 현재 진행 중인 작업

## Windows 실행 + 배포 가능화

Tracer 파이프라인(neuron tracing)을 Windows에서도 돌리고, 다른 사람에게 배포할 수 있게 만드는 작업 중.

### 상황
- 현재 파이프라인은 macOS arm64 전용 바이너리(FileHFM_Riemann3)에 의존함
- 이 폴더(`tracer_portable/`)가 이식 가능한 패키지로 준비 중
- Windows 환경: RTX 3060 12GB, CUDA 설치됨

### 다음 할 일 (Windows PC에서)
1. `nvidia-smi` / `wsl --version` / `docker --version` 으로 환경 확인
2. WSL2 or Docker 중 방향 결정
3. FileHFM을 Linux/Windows용으로 빌드
   - 소스: https://github.com/Mirebeau/HamiltonFastMarching
   - `cmake .. && make FileHFM_Riemann3 FileHFM_Isotropic3`
4. 빌드된 바이너리를 `bin/windows_x64/` 또는 `bin/linux_x64/` 에 복사
5. `methods/ours/FileHFM_binary_dir.txt` 경로 수정

### 패키지 구조
```
tracer_portable/
├── run_batch.py              # 메인 실행
├── environment.yml           # conda env (conda env create -f environment.yml)
├── methods/ours/             # step0~3 파이프라인 스크립트
├── bin/
│   ├── macos_arm64/          # Mac 바이너리 (포함됨)
│   ├── windows_x64/          # Windows 빌드 후 여기에
│   └── linux_x64/            # WSL2/Linux 빌드 후 여기에
└── data/images/              # 입력 TIF 이미지 여기에
```

### 참고
- FileHFM 자체는 CPU 전용 (CUDA 가속 없음)
- CUDA GPU는 향후 step1 OOF 연산 가속화에 활용 가능
