# 코드 구조와 출력
20222038 이진호

모든 경로는 `retraining_improvement/` 기준이다. 각 하위 README는 당시 실행 상태와 경로를 보존한 역사 자료다. 현재 배치 및 실행 조건은 이 문서와 REPRODUCIBILITY_KO.md를 우선 참고한다.

| 파일 | 입력 → 처리 → 출력 |
|---|---|
| `continual/protocol.py` | 행 특징·ID → 중복 검사 / 검증자 라우팅 / 동일 표본 예산 혼합 → 정상·공격·보류 목록과 리플레이 혼합 목록 |
| `continual/experiment.py` | 50클라이언트 CSV → 별도 초기 학습, 3방식 갱신, CEN·임계값 평가 → 결과 JSON, 분할·라우팅, 가중치·scaler·CEN 번들 |
| `continual/infer.py` | 저장 번들 + 정답 없는 115열 CSV → CPU 추론 → 이상 점수·판정 CSV |
| `continual/drift_v2/scenario.py` | 정상 개발 자료 → signed-log/PCA 기반 모의 A/B 분할, 정상 검증·보관 혼합 → 단계별 정상 후보와 고정 입력 전처리 |
| `continual/drift_v2/backend.py` | 원본 SAE·학습기 + 정상 학습 목록 → 원본 동작을 보존한 학습·평가 → 전역 가중치, 비용, 지표 |
| `continual/drift_v2/run.py` | 동결된 V2 계획 → 공통 초기 모델과 7방식×3단계×5시드 → 105개 평가 기록 |
| `continual/research_ABCDE/data.py` | V2 분할 근거 → 조기 종료·임계값·방향 선택의 역할 분리 → 공통 개발 자료, 잠긴 최종 자료 |
| `continual/research_ABCDE/policies.py` | 개발 지표·정상 보관 목록 → 비율·다양성·적응·채택 게이트 → 다음 구성과 배포 여부 |
| `continual/research_ABCDE/engine.py` | 공통 초기 모델 + 정책 → 실제 재학습·개발 분석·평가 → 안별 결과, 버전·채택/복구 기록 |
| `continual/research_ABCDE/run_campaign.py` | V2 모델·분할 번들 → A→분석→B…E→선택 동결→최종 평가 → 안별 보고서와 최종 요약 |
| `continual/head_audit/head_models.py` | 고정 A 인코더·정상 참조 → 중심·거리·정상 보정 → 이상 점수 |
| `continual/head_audit/residual_models.py` | 입력 115차원 + SAE 잠재 11차원 → 블록 표준화·가중 결합 kNN → 이상 점수 |
| `continual/head_audit/stream_memory.py` | 검증 정상 ID → 고정 용량 우선순위 기억 갱신 → 과거·현재 정상 참조 |
| `continual/followup_heads/models.py`, `metric_models.py` | 검증 정상 특징 → 거리·밀도·공분산·입력 전용 기준선 → 이상 점수 |
| `continual/**/verify*.py` | 원래 로컬 실행의 분할·번들·예측 → AUC·혼동행렬·누수·예산·불변성 독립 재계산 → validation.json |
| `continual/**/stress*.py` | 후보 공격을 정상으로 잘못 승인한 모의 상황 → 정상 기억 오염 / 부분 가드 → 강건성 실패와 부분 회복 기록 |
| `continual/**/test*.py` | 작은 가상 입력 → 정책·분할·기억·거리 검사 → 단위 검사 성공/실패 |

## 원본 모듈의 연결

`src/DataLoader`가 표준화된 115차원 배치를 만들고, `src/Model`의 SAE가 `115 → 50 → 11 → 50 → 115`를 계산한다. `src/Trainer`가 로컬 학습과 MSEAvg 집계를 수행한다. CEN은 정상 잠재 표현의 중심과 거리를 사용한다. 원본 로컬 `model.cpt`는 최종 전역 모델 파일이 아니므로 확장은 별도의 공통 초기 전역 모델을 확보했다.

## 정책을 머릿속으로 따라가기

신규 정상 100개, 리플레이 상한 25%라면 총 학습 길이는 100개를 유지하면서 신규 75개 + 기존 25개로 바꾼다. 무조건 100+25개로 늘리는 방식이 아니다. 그래서 최대 표본 예산을 맞춰 비교할 수 있다. 실제 최적화 스텝 수는 조기 종료 때문에 달라질 수 있어 따로 기록한다.

E안에서는 새 버전의 개발 점수와 기존 성능 유지 조건을 검사한다. 통과하면 새 가중치를 채택하고 실패하면 이전 버전을 유지한다. 새 버전을 학습했다는 사실과 배포에 채택했다는 사실을 구분해야 한다.

자세한 실제 코드와 배열·상태 예시는 [전체 코드 읽기 PDF](../reports/FedMSE_Complete_Execution_Reader.pdf)를 참고한다. 이 PDF는 전체 로컬 프로젝트 코드 범위를 설명하므로, 이 커밋에서 제외한 데이터 감사·원본 실행 도우미도 부록에서 언급할 수 있다.
