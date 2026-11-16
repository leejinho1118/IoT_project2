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
page('01 원본 재현 완료와 개선 구현 상태')
p('<b>20222038 이진호</b> | 보고 기준 '+now)
p('원본 GitHub의 기본 IID-50 설정으로 수행한 6개 모델·집계 조합 × 5회 반복, 총 <b>30개 실험이 정상 종료</b>됐다. 종료 시각은 '+original['updated']+'이고 프로세스 종료 코드는 0이다. 학습 후 원본 추적 파일 70개의 내용도 다시 비교해 차이가 없음을 확인했다.')
rows=[]
for x in base['summary']:
 rounds=[r['rounds'][-1]['round'] for r in x['runs']]
 rows.append([labels[x['method']]+' + '+models[x['model']],'5/5',f"{x['final_mean_client_auc_mean']:.5f} ± {x['std_across_runs']:.5f}",str(min(rounds))+'~'+str(max(rounds))])
table(['조합','완료','마지막 ROC-AUC 평균 ± 표준편차','종료 라운드'],rows,[155,55,205,86])
p('각 반복의 실제 마지막 라운드에서 클라이언트 50개의 ROC-AUC를 단순 평균한 뒤, 5회 반복의 평균과 표본 표준편차를 계산했다. 시험 AUC가 가장 높은 라운드를 골라 보고하지 않았다.')
h('이 결과가 의미하는 재현 범위')
p('공개 코드와 배포 데이터에 대한 기본 IID-50 실행 완료이다. 원본의 모델 공유·조기 종료 동작을 유지했다. 모든 논문 시나리오나 논문의 정확도 수치까지 동일하게 재현했다는 뜻은 아니다. ROC-AUC와 정확도는 다른 지표다.')
p('SAE-CEN 평균 AUC는 세 방식이 모두 약 0.983으로 비슷하고, 이번 기본 설정에서는 FedProx 평균이 MSEAvg보다 약간 높다. 이를 근거로 모든 환경에서 특정 집계가 우수하다고 결론 내리지 않는다.','small')

page('02 지속학습 확장에서 구현한 것')
p('목표는 검증·선별한 신규 정상 트래픽으로 SAE-CEN을 갱신하면서 기존 탐지 성능을 유지하는 것이다. 원본 학습 코드와 데이터는 보존하고 continual 폴더에 독립 실행기를 구현했다.')
table(['단계','현재 구현'],[['초기 학습','원본 모듈로 공통 초기 전역 모델을 별도 학습하고 가중치와 초기 탐지 상태를 저장'],['신규 트래픽 탐지','각 방식의 현재 모델로 유입 후보의 점수·이상 판정을 기록'],['검증과 선별','데이터셋 라벨로 검증자를 대신함. 정상·실제 공격·불확실을 분리'],['재학습','재학습 없음 / 신규 정상만 / 신규 정상과 기존 정상 리플레이를 비교'],['갱신 이후 평가','고정한 기존·신규 보류 평가 자료에서 FPR·TPR·AUC와 추가 학습 비용을 기록']], [110,391])
h('정답을 사용하는 정확한 범위')
p('<b>정상 여부를 확인할 수 있는 검증자 보조 환경</b>이다. 유입 후보는 전체를 검증하며 예측을 정답으로 쓰지 않는다. 검증된 정상만 학습에 넣고, 공격과 불확실 표본은 학습에서 제외해 라우팅 기록에 남긴다. 후보 라벨 접근과 최종 평가 라벨 접근을 분리한다.')
h('원본과 공통으로 유지한 조건')
p('SAE 구조, λ 10, 원본 MSEAvg, 로컬 ClientTrainer, 공유 모델의 순차 학습과 전역 조기 종료 동작을 유지했다. 세 방식은 같은 초기 모델, 유입 검증 결과, 참여 순서와 최대 학습 예산을 공유한다.')
h('독립 추론에 필요한 상태도 저장')
p('가중치뿐 아니라 초기 입력 스케일러, CEN, 검증 기반 임계값, 기준 자료 ID와 설정을 detector_bundle.pt에 저장했다. infer.py는 이 번들로 정답 없는 115열 CSV를 처리한다. 원본의 로컬 model.cpt를 최종 전역 모델로 잘못 사용하는 방식은 아니다.')

page('03 데이터 분할과 비교의 통제 조건')
p('이 확장 실험은 원본 결과와 분할이 다른 독립 비교이다. 기존 CSV에는 실제 시간·장치·공격 유형 매핑이 없어 <b>모의 유입 시나리오</b>로 표시한다. 무작위 분할을 실제 시간 드리프트 또는 미지 공격 유형 검증으로 부르지 않는다.')
table(['원본 폴더','확장 실험의 역할'],[['normal','초기 학습 40% / 초기 정상 검증 10% / 서버 개발 후보 40% / 나머지 기존 정상 평가'],['test_normal','신규 후보 40% / 업데이트 정상 검증 20% / 나머지 신규 정상 평가'],['abnormal','신규 후보 20% / 기존 공격 평가 40% / 나머지 신규 공격 평가']], [110,391])
p('분할 전에 115개 특징 전체의 SHA-256으로 전역 중복을 제거하고 행 ID와 교차 중복을 확인했다. 학습용 pandas 3.1.0·numpy 2.5.4 기준 원본 148,254행 중 고유 수치 벡터 145,401개를 보존했다. 정상과 공격 사이 수치 벡터 충돌은 0개다.')
p('문서용 런타임의 기존 감사에서는 146,017개로 계산됐다. 같은 CSV라도 파서의 부동소수점 변환 차이로 정확 중복 수가 달라질 수 있어 두 결과를 같은 정의로 섞지 않는다. 실제 확장 모델이 읽는 런타임의 값으로 분할 누수를 검사했다.','small')
h('리플레이와 학습 예산')
p('초기 정상은 클라이언트별 최대 128개를 보관한다. 재학습 표본 수는 신규 정상 수와 같게 맞추고 최대 25%를 리플레이로 대체한다. 실제 조기 종료에 따른 최적화 스텝 수는 별도로 기록한다. 메모리 지표는 리플레이 특징 페이로드이며 총 프로세스 RSS와 같지 않다.')
h('정규화와 임계값')
p('입력 스케일러는 초기 정상 학습으로 적합한 값을 고정한다. CEN은 각 방식의 정상 기준 자료로 다시 적합하고, 고정된 초기 정상 검증 자료의 점수 95백분위수 규칙으로 임계값을 계산한다. 신규 정상만 방식도 서버 개발 자료와 임계값용 초기 검증 자료는 공통으로 유지한다.')
p('평가 자료의 특징·정답은 후보 선별, 학습, 임계값 튜닝에 쓰지 않는다. 단계마다 같은 보류 평가 세트를 사용하는 적응 추적이므로, 반복 관측 결과를 새 독립 일반화 평가로 설명하지 않는다.','small')

page('04 CPU와 GPU 실행 검증 결과')
p('CPU 프로토콜 검사 6개와 실제 자료 소규모 통합 실행을 통과했다. 원본 종료 후 GPU에서도 같은 소규모 흐름을 실행해 검증했다. 범위는 <b>4개 클라이언트, 역할별 최대 120행, 최대 2에포크·2라운드, 유입 2단계, 1시드</b>이다. 전체 성능 결론용 실험이 아니다.')
table(['검사','확인 결과'],[['분할과 선별','행 ID·특징 벡터 교차 0, 정상만 학습, 공격·불확실 라우팅 분리'],['예산과 실제 갱신','신규 정상과 리플레이의 참여 순서·표본 예산 일치. 두 갱신 방식의 가중치 실제 변경'],['무갱신 기준','초기 가중치와 평가 지표가 유지되고 단계 간에도 변하지 않음'],['저장 결과 재검증','예측 CSV로 혼동행렬을 독립 재계산해 지표와 일치'],['CPU 독립 추론','정답 없이 664행 처리. 저장된 미래 정상 48행과 점수·판정 일치, 최대 점수 차이 0']], [130,371])
h('GPU 소규모 검증의 마지막 단계')
rows=[]
for x in smoke:
 if x['stage']!=2: continue
 a=x['metrics']['old']; b=x['metrics']['new']
 rows.append([x['arm'],f"{a['fpr']:.4f}",f"{b['fpr']:.4f}",f"{b['tpr']:.4f}",f"{b['mean_client_auc']:.4f}"])
table(['방식','기존 FPR','신규 FPR','신규 TPR','신규 AUC'],rows,[145,85,85,85,101])
p('이 수치는 정상 검증 점수의 95백분위수 임계값에서 얻은 동작 확인값이다. 학습량과 자료가 작고 실제 환경 변화를 입증하지 않았으므로, 리플레이의 효과나 운영 탐지 성능을 이 표만으로 주장하지 않는다.')
p('검증 파일: output/smoke_v4/validation.json, output/cuda_smoke_v1/validation.json, output/smoke_v3/standalone_inference_validation.json. 각각의 범위와 실행 장치를 구분한다.','small')

page('05 첫 번째 전체 비교의 중간 결과')
p('오전 9시 기준 전체 비교는 5시드 중 '+str(len(done_seeds))+'시드가 세 방식·유입 3단계까지 완료됐다. 아래는 첫 번째 완료 시드의 마지막 유입 단계 결과다. 초기 모델과 평가 분할은 세 방식에서 동일하다. 원본 표와 분할이 다르므로 원본 최종 AUC와 직접 증감률을 계산하지 않는다.')
partial=[x for x in full_rows if done_seeds and x['seed']==done_seeds[0] and x['stage']==3]
rows=[]
for x in partial:
 for domain,label in [('old','기존'),('new','신규')]:
  v=x['metrics'][domain]
  rows.append([x['arm'],label,f"{v['mean_client_auc']:.5f}",f"{v['fpr']:.5f}",f"{v['tpr']:.5f}"])
table(['방식','평가 영역','평균 AUC','FPR','TPR'],rows,[145,70,100,90,96])
p('현재 한 시드에서는 신규 정상만 재학습한 방식이 무갱신보다 높은 AUC·TPR을 보였고, 리플레이도 무갱신보다 높은 AUC·TPR을 보였다. 리플레이가 신규 정상만 방식보다 우수하다는 결과는 아니다. 5회 평균이나 통계적 우월성으로 해석하지 않는다.')
h('망각 완화가 입증됐는가')
p('이 첫 시드에서는 신규 정상만 방식의 기존 영역 성능도 하락하지 않았다. 따라서 이 결과로 망각이 발생했다거나 리플레이가 망각을 해결했다고 주장할 수 없다. 현재 자료는 실제 시간·장치 변화가 확인되지 않은 모의 유입이므로, 망각을 평가하려면 분포 변화의 근거도 추가로 확보해야 한다.')
h('설정을 바꾸지 않고 반복 결과를 확인')
p('중간 평가 수치를 보고 유리한 시드·라운드·임계값을 골라 바꾸지 않는다. 사전에 정한 5시드·3단계 비교를 유지하며 이후 전체 평균, 변동성, 기존·신규 성능 변화와 학습 비용을 함께 확인한다.')
p('진행 중인 2회차 결과는 이 표에 섞지 않았다. 실제 완료한 단계별 기록은 continual/output/full_v1/results.json과 각 evaluation.json에 저장된다.','small')

page('06 전체 비교 진행과 다음 확인 사항')
p('원본 완료 확인과 30개 결과 검증, GPU 소규모 검증을 통과한 뒤 <b>2026년 11월 10일 오전 8시 32분</b>에 full_v1 전체 비교를 시작했다. 보고서 스냅샷 상태는 '+html.escape(full['state'])+'이며 5회 중 '+str(len(done_seeds))+'회가 완료됐다. 현재 시드는 '+str(full.get('seed'))+'이다. 아직 개선 전체 최종 성능은 산출되지 않았다.')
table(['항목','전체 비교 설정'],[['자료와 클라이언트','IID-50 전체 자료를 읽고 전역 중복 제거, 50개 클라이언트'],['초기 학습','별도 공통 초기 전역 모델, 최대 100에포크·20라운드'],['갱신 비교','세 방식, 신규 유입 3단계, 시드 5개'],['갱신 학습','단계별 최대 100에포크·20라운드. 원본 조기 종료 적용'],['출력 위치','continual/output/full_v1. 원본 결과와 분리']], [125,376])
h('오전 목표의 완료와 미완료를 구분')
p('<b>완료:</b> 원본 기본 실행 30개, 원본 파일 불변 검사, 개선 코드, 분할 감사, CPU·GPU 소규모 검증, 정답 없는 저장 번들 추론 검증.<br/><b>진행 중:</b> 50개 클라이언트·5시드의 전체 지속학습 비교. 완료 전에 성능 향상이나 망각 완화 효과를 확정하지 않는다.')
h('최종 분석에서 확인할 것')
p('새 환경 FPR·TPR와 기존 환경 AUC·FPR 변화, 추가 학습 시간, 실제 최적화 스텝 수와 리플레이 크기를 함께 비교한다. 초기 모델을 별도 학습한 비용도 보고한다. 모델 공유나 조기 종료를 수정한 과거 결과는 원본 기준에 합산하지 않는다.')
h('근거와 이어서 볼 자료')
p('원본: <link href="https://github.com/dino-chiio/fedmse" color="#1265a8">github.com/dino-chiio/fedmse</link><br/>기준 커밋: d1268850696ffa7e9ca5469c0fc9276e966eafd7<br/>원 논문: <link href="https://arxiv.org/abs/2410.14121" color="#1265a8">arxiv.org/abs/2410.14121</link><br/>코드·검증: continual/README_KO.md 및 NIGHT_PROGRESS.md<br/>원본 요약: continual/original_baseline_summary.json 및 CSV<br/>학습 후 원본 검사: continual/original_source_postrun_verification.json<br/>기존 16쪽 인계서는 이전 시각의 구현·계획 설명이며 본 보고서를 후속 기록으로 사용한다.','small')

assert pages==6
snapshot=dict(report_at=now,original=original,baseline_summary=base,full_status=full,gpu_smoke_final=[x for x in smoke if x['stage']==2])
(C/'output/reports/morning_snapshot.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2),encoding='utf-8')
(C/'output/reports/FedMSE_Baseline_and_Continual_Status.md').write_text('\n\n'.join(md),encoding='utf-8')
def footer(c,d):
 c.saveState(); c.setFillColor(MUTED); c.setFont('Body',8); c.drawString(M,H-28,'FedMSE 원본 완료와 개선 구현 검증 / 2026.11.10 / 20222038 이진호'); c.drawString(M,23,'최종 개선 성능과 구현 검증 결과를 구분'); c.drawRightString(W-M,23,str(d.page)+' / 6'); c.restoreState()
pdf=OUT/'FedMSE_Baseline_and_Continual_Status.pdf'
SimpleDocTemplate(str(pdf),pagesize=A4,leftMargin=M,rightMargin=M,topMargin=48,bottomMargin=43,title='FedMSE 원본 재현 완료와 지속학습 구현 검증',author='20222038 이진호').build(story,onFirstPage=footer,onLaterPages=footer)
from pypdf import PdfReader
r=PdfReader(pdf); assert len(r.pages)==6; assert all('20222038' in p.extract_text() for p in r.pages)
print(str(pdf)); print('6 pages verified; report snapshot',now)

