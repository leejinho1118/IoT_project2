# IoT_project2

# 사물인터넷 Project 2 수행 계획서

주제를 정하고, 서베이 논문과 구현체가 공개된 논문을 한 편씩 골라 적습니다. 팀장이 대표로 한 부를 제출합니다.

## 1. 기본 정보

| 항목 | 내용 | 항목 | 내용 |
|---|---|---|---|
| 분반 | 오후반 | 조 | 8조 |
| 제출일 | 2026. 10. 31. | 발표 순번 | 2 |

## 2. 팀원

| 구분 | 학번 | 이름 |
|---|---|---|
| 팀장 | 20221997 | 안민준 |
| 팀원 | 20221993 | 손동한 |
| 팀원 | 20222038 | 이진호 |

## 3. 선정 주제

연합학습

## 4. 선정한 서베이 논문

IF가 높은 서베이·리뷰 논문 한 편 (2024년 이후 출판)

| 항목 | 내용 |
|---|---|
| 논문 제목 | Survey: “Intrusion Detection Based on Federated Learning: A Systematic Review” (ACM Computing Surveys, 2025) |
| 저널명 | ACM Computing Surveys |
| 출판 연도 | 2025년 7월 12일 |
| IF | 30.4 (2025) |
| Quartile | Q1 - 상위 0.3% |
| 논문 URL | https://dl.acm.org/doi/10.1145/3731596 |

## 5. 선정한 구현체 논문

구현체가 GitHub 등에 공개된 논문 한 편 (2024년 이후 출판)

| 항목 | 내용 |
|---|---|
| 논문 제목 | FedMSE: Semi-supervised federated learning approach for IoT network intrusion detection |
| 게재처 | COMPUTERS & SECURITY |
| 출판 연도 | 2025년 4월 |
| IF / 등급 | IF: 6.8 (2025) |
| 논문 URL | 논문 링크 (원본에 URL 주소 미기재) |
| GitHub 주소 | https://github.com/dino-chiio/fedmse |

## 6. 수행 계획

**목표 수준**

- [ ] 문제 분석과 발표까지
- [ ] 구현체 구동까지
- [x] 기법 적용과 확장까지

**수행 계획**

- **3~8주차 (9월 15일 ~ 10월 20일)**
  - ACM Computing Surveys 서베이 논문을 통해 연합학습 기반 침입 탐지(IDS)의 전반적인 트렌드를 읽고, FedMSE 논문의 서론·초록을 읽으며 IoT 환경의 프라이버시 및 데이터 부족 문제 파악.
- **9주차 (10월 27일)**
  - Project 2 계획서 점검.
  - FedMSE 논문 집중 분석: Shrink Autoencoder(SAE)를 통한 정상 네트워크 데이터의 잠재 공간(Latent space) 특징 표현 방식과 Centroid(CEN) 기반 이상 탐지 원리 분석. 기존 중앙집중형 IDS의 한계와 FedMSE가 제안하는 MSEAvg(Mean Squared Error-based aggregation) 알고리즘의 구조 정리.
- **10주차 (11월 3일)**
  - 논문 분석 내용을 바탕으로 발표 자료 구성 (문제 정의, SAE-CEN 하이브리드 모델 구조, 연합학습 아키텍처 다이어그램 설계).
  - GitHub(dino-chiio/fedmse) 저장소의 구조(`DataLoader/`, `Model/`, `Trainer/` 등)를 파악하고, 실험에 사용되는 N-BaIoT 등의 데이터셋 준비 사항 점검.
- **11주차 (11월 10일)**
  - '자율형 IoT와 인공지능 - 연합학습' 정규 수업 진도와 연계하여 학습 진행.
  - 구현체 구동 시작: 로컬 환경에 Python 가상 환경을 구축하고 `requirements.txt`를 통해 라이브러리 설치. 설정 파일(`config_file`)을 바탕으로 데이터 전처리 및 클라이언트별 Non-IID 환경 분할 코드 실행.
- **12주차 (11월 17일)**
  - `main.py`를 구동하여 다수의 게이트웨이(Client) 간 로컬 학습 및 중앙 서버의 MSEAvg 모델 집계(Aggregation) 과정 재현.
  - 학습 로그(Checkpoint)를 분석하여 논문에서 제시한 탐지 성능(Accuracy, AUC 등) 지표가 재현되는지 확인하고 최종 발표용 PPT 제작 마감.
