from pathlib import Path
import json,datetime,html,sys
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Preformatted
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
ROOT=Path(__file__).resolve().parents[1]; C=ROOT/'continual'; OUT=C/'output/reports'
base=json.loads((C/'original_baseline_summary.json').read_text(encoding='utf-8'))
original=json.loads((ROOT/'original_output/status.json').read_text(encoding='utf-8'))
full=json.loads((C/'output/full_v1/status.json').read_text(encoding='utf-8'))
smoke=json.loads((C/'output/cuda_smoke_v1/results.json').read_text(encoding='utf-8'))
verified=json.loads((C/'output/cuda_smoke_v1/validation.json').read_text(encoding='utf-8'))
plan=json.loads((C/'output/full_prepare_v1/split_audit.json').read_text(encoding='utf-8'))
now=datetime.datetime.now().astimezone().isoformat(timespec='seconds')
full_rows=json.loads((C/'output/full_v1/results.json').read_text(encoding='utf-8')) if (C/'output/full_v1/results.json').exists() else []
expected={(arm,stage) for arm in ['no_update','new_only','replay'] for stage in [1,2,3]}
done_seeds=[seed for seed in sorted({x['seed'] for x in full_rows}) if {(x['arm'],x['stage']) for x in full_rows if x['seed']==seed}==expected]
labels={'avg':'FedAvg','fedprox':'FedProx','mse_avg':'MSEAvg'}; models={'hybrid':'SAE-CEN','autoencoder':'AE'}
pdfmetrics.registerFont(TTFont('Body','C:/Windows/Fonts/malgun.ttf')); pdfmetrics.registerFont(TTFont('Bold','C:/Windows/Fonts/malgunbd.ttf')); pdfmetrics.registerFont(TTFont('Code','C:/Windows/Fonts/consola.ttf'))
pdfmetrics.registerFontFamily('Body',normal='Body',bold='Bold',italic='Body',boldItalic='Bold')
W,H=A4; M=47; WIDTH=W-2*M; NAVY=colors.HexColor('#1c3956'); MUTED=colors.HexColor('#60758a'); LIGHT=colors.HexColor('#edf4f8'); GRID=colors.HexColor('#d9e0e7')
S={k:ParagraphStyle(k,fontName=font,fontSize=size,leading=lead,spaceAfter=8,wordWrap='CJK',textColor=col) for k,font,size,lead,col in [('body','Body',10,15.7,colors.HexColor('#18232d')),('title','Bold',18,25,NAVY),('h','Bold',11.5,17,NAVY),('small','Body',8.4,13,MUTED),('cell','Body',9,13.5,colors.HexColor('#18232d')),('head','Bold',9,13.5,NAVY)]}
S['h'].keepWithNext=True; S['h'].spaceBefore=8
story=[]; md=[]; pages=0

def p(t,style='body'): story.append(Paragraph(t,S[style])); md.append(t.replace('<br/>','\n'))
def page(t):
 global pages
 if pages: story.append(PageBreak())
 pages+=1; p(t,'title'); md.append('\n')
def h(t): p(t,'h')
def table(headers,rows,widths):
 data=[[Paragraph(html.escape(str(x)),S['head' if i==0 else 'cell']) for x in r] for i,r in enumerate([headers]+rows)]
 widths=[WIDTH*x/sum(widths) for x in widths]; t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),LIGHT),('LINEABOVE',(0,0),(-1,0),.8,NAVY),('LINEBELOW',(0,0),(-1,-1),.4,GRID),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
 story.extend([t,Spacer(1,10)]); md.extend(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])

v=json.loads((C/'output/full_v1/validation.json').read_text()); summary=v['summary_stage_3']
page('01 FedMSE 원본 재현과 지속학습 최종 결과')
p('<b>20222038 이진호</b> | 최종 보고 '+now)
p('원본 기본 IID-50 실행 30개와 지속학습 확장 5시드 × 3방식 × 3단계의 <b>45개 평가가 모두 완료</b>됐다. 확장 종료는 2026년 11월 10일 오전 9시 56분이다. 코드·데이터 분할·저장 예측을 별도 도구로 검증했다.')
h('확인한 핵심 결과')
p('신규 정상 재학습 두 방식 모두 무갱신보다 신규 영역의 평균 AUC와 공격 탐지율이 높았다. 그러나 <b>리플레이가 신규 정상만 재학습보다 우수하다는 근거는 확보하지 못했다.</b> 신규 정상만 방식의 신규 AUC 평균은 0.99312, 리플레이는 0.99263이다. 두 방식 차이에 대한 통계적 유의성을 주장하지 않는다.')
h('원본 구현의 완료 범위')
rows=[]
for x in base['summary']:
 rows.append([labels[x['method']]+' + '+models[x['model']],f"{x['final_mean_client_auc_mean']:.5f} ± {x['std_across_runs']:.5f}"])
table(['원본 조합 / 각 5회','마지막 라운드 평균 AUC ± 표준편차'],rows,[220,281])
p('공개 기본 설정 50클라이언트·참여율 50%·최대 100에포크·20라운드로 원본 main.py를 수정 없이 실행했다. 30개 결과의 연속 라운드·50개 AUC·25개 참여를 확인했고 원본 추적 파일 70개가 불변이었다. 기준 커밋: d1268850696ffa7e9ca5469c0fc9276e966eafd7. 모든 논문 시나리오 재현을 의미하지 않는다.','small')
p('원본 표는 각 반복 마지막 라운드의 클라이언트 평균 AUC를 5회 요약했다. 확장은 별도 초기 학습과 분할을 사용하므로 다음 페이지 수치와 직접 합산하거나 개선 폭을 계산하지 않는다.','small')
page('02 지속학습 비교 결과: 신규 영역')
p('아래는 사전에 고정한 마지막 유입 단계의 5시드 평균 ± 표본 표준편차다. AUC는 클라이언트 50개의 AUC를 단순 평균하며, FPR·TPR은 클라이언트 혼동행렬을 합산해 계산한다. 평가에 유리한 단계나 시드를 골라내지 않았다.')
table(['방식','ROC-AUC','FPR','TPR'],[[x['arm']]+[f"{x['new'][k]['mean']:.5f} ± {x['new'][k]['std']:.5f}" for k in ['mean_client_auc','fpr','tpr']] for x in summary],[95,140,133,133])
h('공통 초기 모델에서 무엇이 달라졌는가')
p('신규 정상만 재학습은 신규 TPR이 91.15%에서 99.45%로 약 8.30%p 높아졌다. 리플레이는 99.13%로 약 7.98%p 높았다. 반면 FPR은 세 방식 모두 약 5.7%로, 오탐률 개선은 확인하지 못했다.')
h('리플레이에 대한 해석')
p('현재 평균에서는 신규 정상만 방식이 리플레이보다 신규 AUC·TPR이 조금 높다. 리플레이의 일관된 우월성이나 망각 완화 효과를 결론으로 제시하면 안 된다. 5회는 같은 분할에서 초기화·참여 순서를 바꾼 반복이며 서로 독립인 5개 실제 환경이 아니다.')
h('평가 범위와 제약')
p('원본 CSV에 실제 시간·물리 장치·공격 유형 매핑이 없어 모의 유입 시나리오로 구현했다. 초기·유입·평가의 역할은 구분되지만, 실제 시간 드리프트나 미지 공격 유형 검증은 수행하지 않았다. 단계마다 동일한 보류 평가 세트를 사용해 갱신 추이를 측정했으며 최종 평가 라벨로 학습·임계값·설정을 조정하지 않았다.')
p('FPR: 정상 중 공격으로 잘못 판단한 비율. TPR: 실제 공격 중 탐지한 비율. 이 실험의 AUC는 정확도가 아니다.','small')
page('03 기존 성능 유지와 학습 비용')
table(['방식','기존 AUC','기존 FPR','기존 TPR'],[[x['arm']]+[f"{x['old'][k]['mean']:.5f} ± {x['old'][k]['std']:.5f}" for k in ['mean_client_auc','fpr','tpr']] for x in summary],[95,140,133,133])
p('초기 대비 기존 영역 AUC 변화는 신규 정상만 +0.00445 ± 0.00484, 리플레이 +0.00403 ± 0.00395다. 평균에서 기존 성능 저하가 나타나지 않았으므로, 이 결과만으로 망각이 발생했거나 리플레이가 이를 해결했다고 주장할 수 없다. 개별 시드의 변화는 근거 JSON에 보존했다.')
h('추가 학습 비용: 3단계 누적 / 시드별')
table(['방식','추가 학습 시간(초)','실제 optimizer 스텝'],[[x['arm'],f"{x['cumulative_additional_seconds']['mean']:.1f} ± {x['cumulative_additional_seconds']['std']:.1f}",f"{x['update_optimizer_steps']['mean']:.0f} ± {x['update_optimizer_steps']['std']:.0f}"] for x in summary],[130,185,186])
p('공통 초기 전역 모델을 별도로 확보하는 학습 시간은 시드당 789.7 ± 253.4초, 5시드 합계 3,948.3초다. 각 방식마다 중복 비용으로 더하지 않는다. 위 추가 시간은 fit 수행 시간이며 평가·파일 저장을 포함한 전체 경과 시간이 아니다. 전체 비교의 벽시계 경과 시간은 약 84분이다.')
p('리플레이는 클라이언트당 정상 128개, 총 6,400개를 고정 보관했다. 특징 페이로드는 5,888,000바이트(약 5.62 MiB, 115열 float64)다. Python 객체·모델·GPU 메모리·프로세스 전체 RSS는 이 수치에 포함하지 않았고 총 피크 메모리 계측은 수행하지 않았다.')
p('두 갱신 방식은 같은 최대 라운드·에포크·참여 순서와 클라이언트별 표본 수를 사용한다. 원본 조기 종료를 유지해 실제 실행 라운드와 스텝 수는 다를 수 있다.','small')
page('04 구현 흐름과 데이터 누수 통제')
table(['단계','구현 및 통제'],[['초기 학습','원본 SAE·MSEAvg로 별도 공통 전역 모델 학습. 로컬 model.cpt를 전역 모델로 사용하지 않음'],['탐지 및 검증','현재 모델로 후보 점수 기록. 정답 라벨을 검증자 대체 수단으로 사용하며 후보 전체 검증'],['선별','검증 정상만 학습. 실제 공격은 격리, 불확실은 보류. 전체 실험 불확실 비율 0%, 소규모 검증에서 보류 경로 확인'],['재학습','무갱신 / 신규 정상만 / 신규 정상 중 최대 25%를 기존 정상 리플레이로 대체'],['평가 및 배포 상태','기존·신규 보류 자료 평가. 가중치·입력 스케일러·CEN·임계값·기준 ID 저장, 독립 CPU 추론 검증']], [100,401])
p('<b>정상 여부를 확인할 수 있는 검증자 보조 환경</b>이다. 데이터셋 라벨을 접근할 수 없는 실제 운영 환경의 자동 검증 능력을 구현한 것은 아니다.')
p('148,254행에서 전역 중복 제거 후 145,401개 고유 수치 벡터를 분할했다. 모든 역할의 행 ID·특징 SHA-256 교차 중복은 0개다. 학습 런타임 pandas 3.1.0·numpy 2.5.4 기준이며, 다른 CSV 파서의 exact duplicate 집계와 혼합하지 않았다.')
p('normal은 초기 학습·정상 검증·서버 개발·기존 평가로, test_normal은 유입 후보·갱신 검증·이후 정상 평가로, abnormal은 유입 공격 후보·기존 공격 평가·이후 공격 평가로 분리했다. 상세 행 식별자와 특징 해시는 split_manifest.json.gz에 기록했다.')
p('초기 입력 스케일러는 고정했다. 모든 방식이 초기 정상 서버 개발 자료와 임계값 검증 자료를 공통 사용한다. 각 방식의 CEN 기준은 해당 방식 정상 학습 자료로 재계산하고, 임계값은 고정된 초기 정상 검증 점수의 95백분위수로 재설정한다. 신규 정상만이라는 표현은 로컬 학습 및 CEN 기준을 뜻한다.')
p('원본의 모델 구조·MSEAvg·공유 모델 순차 학습·조기 종료를 유지했다. 이 공유 모델 동작은 보존한 원본 구현의 특성이며, 독립 로컬 모델 방식으로 교정한 실험을 함께 섞지 않았다.','small')
page('05 검증 근거와 연구 결론')
table(['항목','검증 결과'],[['전체 평가','5시드 × 3방식 × 3단계 = 45개, 조합 중복·누락 없음'],['저장 예측 재검산','45개 예측 파일의 역할·정답·혼동행렬·FPR·TPR 확인. 기존/신규 총 4,500개 클라이언트 AUC 재계산'],['비교 조건','무갱신 모델 및 지표 불변, 갱신 모델 변화, 참여 순서 공통 접두·클라이언트별 표본 예산 및 실제 스텝 확인'],['코드 및 분할','실행 전후 원본·확장 소스 해시 동일, 전역 ID·특징 중복 없음, 후보 정상 검증 라우팅 확인'],['별도 실행 검증','프로토콜 테스트 6개, CPU smoke_v4·GPU cuda_smoke_v1 및 독립 검증, 라벨 없는 번들 CPU 추론 일치']], [130,371])
h('보고서에 제시할 수 있는 결론')
p('검증된 신규 정상 트래픽을 활용해 FedMSE를 갱신하는 지속학습 파이프라인을 구현했고, 고정된 모의 유입 분할에서 신규 영역 AUC·TPR의 평균 상승과 기존 영역 평균 성능 유지를 관찰했다. 오탐률 감소와 리플레이의 우월성은 확인하지 못했다.')
h('다음 연구에서 검증해야 할 것')
p('실제 시간 순서 또는 장치별 분포 변화가 있는 자료를 확보하고, 이전 정상 패턴을 잊기 쉬운 시나리오에서 리플레이의 효과를 평가해야 한다. 검증자의 오판·불확실성과 메모리 용량 변화도 추가 과제다. 이번 최종 보류 결과에 맞춰 설정을 다시 고른 뒤 동일 자료를 최종 평가로 재사용해서는 안 된다.')
h('재현 및 확인 경로')
p('코드: continual/experiment.py · protocol.py · infer.py<br/>독립 검증: continual/verify_full.py<br/>결과·검증: continual/output/full_v1/results.json · validation.json<br/>설정·분할: config.json · split_manifest.json.gz · split_audit.json<br/>원본 요약: continual/original_baseline_summary.json<br/>실행 로그·번들·시드별 학습 기록은 full_v1에 보존했다.','small')
p('원본: https://github.com/dino-chiio/fedmse<br/>관련 논문: https://arxiv.org/abs/2410.14121<br/>작업 및 대용량 결과 위치: C:/Users/dlwls/LocalProjects/fedmse<br/>현재 결과는 실제 운영 환경 일반화 또는 개인정보 보호·오염 공격 강건성을 검증하지 않는다.','small')
def footer(canvas,doc):
 canvas.setFont('Body',8);canvas.setFillColor(MUTED);canvas.drawString(M, H-27,'FedMSE 원본 재현 및 지속학습 최종 보고 / 20222038 이진호');canvas.drawString(M,26,'원본 결과와 확장 비교를 구분 · 모의 유입 시나리오');canvas.drawRightString(W-M,26,f'{doc.page} / 5')
OUT.mkdir(exist_ok=True,parents=True)
f=OUT/'FedMSE_Final_Continual_Report.pdf';SimpleDocTemplate(str(f),pagesize=A4,rightMargin=M,leftMargin=M,topMargin=48,bottomMargin=43).build(story,onFirstPage=footer,onLaterPages=footer)
(OUT/'FedMSE_Final_Continual_Report.md').write_text('\n\n'.join(md),encoding='utf-8')
print(f)
