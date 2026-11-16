# 실행과 재현 조건
20222038 이진호 · 2026-11-16

## 환경

기존 실험은 Windows / Python 3.12에서 수행했다. 정확한 학습 패키지는 `requirements-training.txt`에 보존했다. CUDA용 torch와 당시 버전 핀은 과거 실행 환경 기록이며 모든 OS에서 설치된다는 보장은 없다. CPU 검사는 해당 환경의 torch를 사용하되 GPU를 비활성화한다. 보고서 재생성에는 별도로 reportlab, pypdf, pdfplumber, Pillow, Poppler와 한글 글꼴이 필요하다.

## 데이터 없는 가벼운 검사

이 폴더를 현재 작업 디렉터리로 지정하고 Python 환경을 활성화한다. 아래 검사는 학습이나 네트워크 접속을 하지 않는다.

```powershell
python -B tools/verify_package.py
$env:CUDA_VISIBLE_DEVICES = '-1'
python -B -m unittest discover -s continual -p test_protocol.py -v
python -B -m unittest discover -s continual/drift_v2 -p test_protocol.py -v
python -B -m unittest discover -s continual/research_ABCDE -p test_policies.py -v
python -B -c "import torch,unittest; s=unittest.defaultTestLoader.discover('continual/head_audit',pattern='test*.py'); r=unittest.TextTestRunner().run(s); raise SystemExit(not r.wasSuccessful())"
python -B -c "import torch,unittest; s=unittest.defaultTestLoader.discover('continual/followup_heads',pattern='test*.py'); r=unittest.TextTestRunner().run(s); raise SystemExit(not r.wasSuccessful())"
```

## 전체 학습은 추가 자료가 필요함

코드가 사용하는 ROOT 상대 경로를 보존했다. 학습 실행 전에 다음 자료를 별도로 준비해야 한다. 이 커밋의 요약 JSON만으로 전체 성능을 재검산할 수 있다는 뜻이 아니다.

1. `Data/N-BaIoT/IID-50-Client_Data/Client-1..50/{normal,test_normal,abnormal}/data.csv`: 원래 순서의 헤더 없는 115열 숫자 CSV. 저장소의 `../FedMSE/Data`를 사용할 수 있으나 원래 실험의 SHA·라이브러리 파싱 정의와 일치하는지 확인한다.
2. `original_output/status.json`과 원본 `src/Checkpoint/Results`: 원본 30개 완료를 확인하는 실제 실행 근거. V1/V2는 이 선행 조건을 검사한다. `evidence/`는 역사 요약 보관 위치로서 실행 상태 파일을 대신하도록 연결하지 않았다.
3. V2 결과의 `split_manifest.json.gz`, 공통 초기 모델·scaler, 단계별 번들: A~E가 의존한다. A~E의 저장 모델과 분할은 후속 탐지 헤드 실험에도 필요하다.
4. 독립 검증기에는 원래 실행의 예측 CSV·학습 ID·모델이 필요하다. 대형 자료는 로컬 실험 경로에 별도로 보존돼 있다.

기존 `FedMSE/Data`를 복사하지 않고 연결하려면 Windows에서 이 폴더의 비어 있는 `Data` 경로에만 로컬 디렉터리 junction을 만들 수 있다. 이 연결은 커밋하지 않는다. 이번 패키징에서는 생성하지 않았다.

V2가 준비된 별도 실행 환경에서의 예:

```powershell
python continual/drift_v2/run.py --profile smoke --device cpu --run-id cpu_smoke_NEW
python continual/drift_v2/verify.py --run-id cpu_smoke_NEW
# A~E는 먼저 대응하는 V2 초기 학습·분할 결과가 있어야 한다.
python continual/research_ABCDE/run_campaign.py --profile full --device cuda --run-id campaign_NEW
```

일부 runner는 선행 원본·V2 경로 이름을 코드에서 고정한다. 따라서 새로운 run-id만 바꾸면 모든 후속 단계가 자동으로 새 결과를 참조하는 범용 CLI가 아니다. 기존 출력 경로가 있으면 덮어쓰기를 거부하며 미완료 실행을 자동 재시작하지 않는다. runner 소스와 각 시점 README의 선행 조건을 확인한다.

## 이번 커밋의 검증 범위

`evidence/package_validation.json`은 파일·소스 구문·복사 해시 검사를, `evidence/unit_test_results.json`은 이 패키지에서 재실행한 소규모 규칙 검사를 기록한다. 과거 성능 실험에 대한 독립 검증 결과는 `evidence/continual/.../validation.json`에 별도로 보존했다. 원시 특징·예측·모델을 이 커밋에 넣지 않았으므로 이번 정리 작업에서 과거 전체 학습과 AUC 재계산을 반복하지 않았다.

## 제외한 것

원시 데이터셋, 가상환경, GPU 체크포인트, pickle 번들, 대형 예측 CSV, 분할 행별 전체 manifest, 학습 로그, 렌더링 이미지 캐시, 중복 ZIP. 보고서의 절대 경로와 일부 검증 SHA는 과거 로컬 실행의 출처로 남겼다. 논문 수준 확증을 위해서는 새 분할과 독립 평가 자료가 추가로 필요하다.

## 이 패키지에서 확인한 검사 결과

2026-11-16 CPU 단위 검사 42개 통과: V1 프로토콜 6개, V2 7개, A~E 정책 8개, 헤드·정상 기억 13개, 후속 거리 8개. 복사한 183개 파일의 해시, 확장·도구 Python 61개 구문, 새 안내 문서 링크 20개를 검사했다. 원본 src는 역사적 소스를 그대로 보존했으며 확장 구문 검사 수에 포함하지 않았다. 전체 학습 성능 검증의 재실행 결과는 아니다.
