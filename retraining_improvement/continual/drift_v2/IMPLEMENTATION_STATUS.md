# 개선 구현 V2 진행 현황

20222038 이진호 / 2026-11-13

## 완료

- 개발 자료만으로 정상 A/B 특징 분포를 정하고 A 초기 학습 -> B 유입 -> 검증 -> 재학습 -> A/B 평가를 구현
- 보관량0/32/128, 불확실20% 보류 민감도 비교
- 고정 초기 입력 스케일러와 모든 방식 공통 [-20,20] 범위 제한, 정상 복원 MSE·초기 임계값 고정 지표·실제 스텝·학습 시간 기록
- 프로토콜 테스트7개, CPU14개 평가와 GPU 개발42개 평가의 독립 재검산 통과
- 저장 번들 정답 없는 CPU 추론664행 실행, 보류58행 점수·판정 완전 일치
- 기존 코드·결과54개, 원본CSV150개 해시 보존 확인

## 실행 중

전체50클라이언트·5시드·7방식·3단계=105개 평가를 GPU로 실행 중이다. 개발 실행은 최종 성능 결과가 아니다.

- 전체 작업 상태: full_job_status.json
- 학습 상세 상태: output/full_v1/status.json
- 학습 기록: output/full_v1/training.log
- 완료 후 독립 검증: output/full_v1/validation.json
- 완료 후 최종 표: output/full_v1/summary.csv
- 완료 후 한국어 결과: output/full_v1/FINAL_RESULTS_KO.md

run_full.ps1은 학습 정상 종료 뒤 verify.py와 finalize.py를 순서대로 실행한다. 실패 시 기록을 보존하고 멈추며, 처음부터 자동 재시작하지 않는다. 전체 결과가 검증되기 전에는 성능 향상이나 리플레이 우월성을 주장하지 않는다. 모든 작업은 OneDrive 밖의 C:/Users/dlwls/LocalProjects/fedmse/continual/drift_v2에서 수행한다.

설정·해석 범위는 PROTOCOL_FROZEN.json과 README_KO.md를 참고한다. 최종 자료는 시나리오/학습/임계값 선정에서 제외하지만, 이 실험은 기존과 같은 데이터셋을 별도 분할한 모의 특징 변화이며 독립 실시간 드리프트 검증은 아니다.
