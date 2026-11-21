"""Generate a tablet-readable evidence report from independently validated finals."""
from pathlib import Path
import json,datetime,html,collections,hashlib
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
H=Path(__file__).resolve().parent;A=json.loads((H/'analysis_v1/analysis.json').read_text(encoding='utf-8'));OUT=H/'output/pdf';OUT.mkdir(parents=True,exist_ok=True);PDF=OUT/'FedMSE_Overnight_Retraining_Study.pdf';# Explicit report rebuild; model outputs and evaluation records remain unchanged.
pdfmetrics.registerFont(TTFont('KR','C:/Windows/Fonts/malgun.ttf'));pdfmetrics.registerFont(TTFont('KRB','C:/Windows/Fonts/malgunbd.ttf'))
styles=getSampleStyleSheet();styles.add(ParagraphStyle(name='BodyKR',fontName='KR',fontSize=10.3,leading=16,spaceAfter=9,wordWrap='CJK'));styles.add(ParagraphStyle(name='TitleKR',fontName='KRB',fontSize=21,leading=29,spaceAfter=17,wordWrap='CJK'));styles.add(ParagraphStyle(name='HeadKR',fontName='KRB',fontSize=15,leading=22,spaceAfter=13));styles.add(ParagraphStyle(name='SmallKR',fontName='KR',fontSize=8,leading=12,wordWrap='CJK'));story=[]
def p(text,style='BodyKR'):return Paragraph(html.escape(str(text)),styles[style])
def para(text):story.append(p(text))
def title(text):story.append(p(text,'HeadKR'))
def page(text):story.append(PageBreak());title(text)
def table(headers,rows,widths=None):
 data=[[p(x,'SmallKR') for x in headers]]+[[p(x,'SmallKR') for x in row] for row in rows];t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT');t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8eef5')),('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),5),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#627b94')),('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#d8dfe6')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f8fafc')])]));story.extend([t,Spacer(1,11)])
def f(x,n=4):return '-' if x is None else f'{x:.{n}f}'
def pct(x):return '-' if x is None else f'{100*x:.1f}%'
names={'no_update':'고정','calibration_only':'임계값만','reference_only':'정상 기준만','new_only':'신규만','replay':'리플레이','safe_replay':'검증 리플레이','trigger_safe':'조건부 검증','safe_new':'검증 신규','normal_gate_new':'정상 선택 신규','normal_gate_replay':'정상 선택 리플레이','reference_only_new':'신규 기준만','distill_replay':'증류 리플레이','align_distill_replay':'정렬+증류','safe_align_distill':'검증 정렬+증류'}
primary=A['selection']['selected'][1];datasets=[('raw_blocks_v2','N-BaIoT'),('rt_thing_to_mqtt_v1','RT: Thing → MQTT'),('rt_mqtt_to_thing_v1','RT: MQTT → Thing')]
def rows(data,kind='main30',mode=None):return sorted([r for r in A['summary'] if r['data']==data and r['kind']==kind and r['mode']==(mode or primary['mode']) and r['loss']==primary['loss'] and r['stage']==3],key=lambda r:list(names).index(r['method']))
def result_table(rr,show_data=False):
 hdr=['방식','시드','기존 AUC','신규 AUC','신규 FPR','신규 TPR'];body=[[names[r['method']],str(r['n_seeds']),f(r['old_mean_client_auc_mean']),f(r['new_mean_client_auc_mean']),pct(r['new_fpr_mean']),pct(r['new_tpr_mean'])] for r in rr];table(hdr,body,[98,35,85,85,80,80])
def matched_comparison(data,kind,controls):
 target=rows(data,kind);seeds={x['seed'] for x in A['records'] if x['data']==data and x['kind']==kind and x['mode']==primary['mode'] and x['stage']==3};result=[]
 for method in controls:
  values=[x for x in A['records'] if x['data']==data and x['kind']=='main30' and x['mode']==primary['mode'] and x['stage']==3 and x['method']==method and x['seed'] in seeds]
  if not values:continue
  row=dict(next(r for r in rows(data) if r['method']==method));row['n_seeds']=len(values);row['seeds']=sorted(x['seed'] for x in values)
  for key in ['old_mean_client_auc','new_mean_client_auc','new_fpr','new_tpr']:row[key+'_mean']=sum(x[key] for x in values)/len(values)
  result.append(row)
 return result+target
now=datetime.datetime.now().astimezone().isoformat();story.append(p('FedMSE 재학습\n야간 추가 실험 보고서','TitleKR'));para('20222038 이진호 · '+now);para('원본 방식 고정과 정상 검증 후 재학습을 분리하고, 전처리·검증 예산·승인 오류·학습 예산을 추가로 비교했다. 이 보고서는 독립 검산을 마친 최종 평가만 수록한다.')
para(f"실제 완료: {len(A['runs'])}개 실행, {len(A['records'])}개 방식·시드·단계 평가. 개발 조건 4개 중 추가 검증 대상으로 {primary['mode']} / {primary['loss']}를 개발 자료에서 고정했다. 사전 주 방법은 ‘검증 신규’이며 최종 평가 성능으로 바꾸지 않았다.")
for data,label in datasets:
 rr=rows(data);b=next((r for r in rr if r['method']=='no_update'),None);s=next((r for r in rr if r['method']=='safe_new'),None)
 if b and s:para(f"{label}: 고정 대비 검증 신규의 신규 AUC 변화 {s['new_mean_client_auc_mean']-b['new_mean_client_auc_mean']:+.4f}, 신규 FPR 변화 {(s['new_fpr_mean']-b['new_fpr_mean'])*100:+.2f}%p, TPR 변화 {(s['new_tpr_mean']-b['new_tpr_mean'])*100:+.2f}%p. 기존 AUC 변화 {s['old_mean_client_auc_mean']-b['old_mean_client_auc_mean']:+.4f}.")
para('AUC 증가만으로 개선을 선언하지 않는다. 공격 탐지율과 오탐률, 기존 성능 유지, 실제 재학습 여부와 비용을 함께 확인해야 한다. 로그 전처리의 효과와 재학습의 효과도 따로 비교했다.')
page('결론 · 재학습의 이득과 실패를 함께 해석')
para('사전 주 방법인 검증 신규는 외부 데이터에서 단순 신규/리플레이보다 일관되게 좋지 않았다. 따라서 검증 게이트의 우월성 가설은 이번 실험에서 지지되지 않았다. 추가 증류·정렬 손실과 정상 전용 선택도 보편적인 이득을 보여주지 않았다.')
para('사후 분석에서 실용적인 후속 후보는 단순 리플레이다. 이는 최종 결과를 보고 내린 탐색적 판단이며, 사전 주 방법을 바꾸거나 새 보류 자료의 확인 검증을 마친 것으로 표현하지 않는다.')
for data,label in datasets:
 rr=rows(data);base=next(r for r in rr if r['method']=='no_update');replay=next(r for r in rr if r['method']=='replay');para(f"{label}: 고정 → 리플레이의 신규 AUC {base['new_mean_client_auc_mean']:.4f} → {replay['new_mean_client_auc_mean']:.4f}, 신규 FPR {100*base['new_fpr_mean']:.2f}% → {100*replay['new_fpr_mean']:.2f}%, 신규 TPR {100*base['new_tpr_mean']:.2f}% → {100*replay['new_tpr_mean']:.2f}%. 기존 AUC {base['old_mean_client_auc_mean']:.4f} → {replay['old_mean_client_auc_mean']:.4f}.")
para('개발 자료에서 선택한 로그 조건도 최종 원자료에서는 더 좋지 않았다. 표준화 고정의 신규 AUC는 0.9953이고 로그 고정은 0.9836이다. 전처리 선택의 외부 이전 실패까지 포함해 해석해야 한다.')
para('외부 정상 패턴 전환에서는 재학습이 고정 모델보다 좋아질 근거가 있으나, 원자료에서는 초기 성능이 높아 추가 이득이 작거나 악화됐다. 또한 정상 기준만 바꿔도 신규 AUC가 크게 오르므로 전체 향상을 신경망 재학습 덕분으로 돌릴 수 없다. 동일 신규 정상 기준 대조군과의 짝지은 차이를 따로 제공했다.')
para('중요한 실패: Thing→MQTT에서 리플레이의 Metasploit SSH 탐지율은 1.18%에 그쳤다(34개 보류 공격, 5시드 평균). 평균 AUC가 높아도 특정 공격을 거의 놓칠 수 있다. 현 상태를 실사용 보안 성능이나 투고 준비 완료로 판단하지 않는다.')
para('원자료 오승인 5% 조건에서는 검증 신규가 신규 AUC 0.9805를 유지한 반면 단순 신규 0.9308, 리플레이 0.9342였다(각 3시드). 이는 깨끗한 보호 검증 자료가 있는 오류 실험에서의 방어 효과이며, 정상 상태의 탐지 성능 향상이나 모든 외부 환경의 우월성을 뜻하지 않는다.')
page('1. 데이터와 평가 분리')
raw=json.loads((H/'raw_blocks_v2/preparation_audit.json').read_text());prep=json.loads((H/'preparation_validation.json').read_text());table(['자료','실험 단위','특징 / 검증'],[['N-BaIoT 공식 원자료','9개 실제 장치; Mirai는 7개 장치','115개 수치 특징; 89개 CSV 해시 확인'],['RT-IoT2022 공식 원자료','5개 가상 클라이언트; 정상 패턴 순서 2종','81개 수치 특징; 각 스키마로 초기 모델 새 학습']],[140,180,183]);para(f"N-BaIoT는 이전 가공 CSV와 같은 공개 수집 자료다. 이전 정확 일치 {raw['audit']['prior_exact_excluded']:,}행과 보수적 로그 반올림 일치 {raw['audit']['prior_coarse_excluded']:,}행을 제외했지만 독립적인 새 수집 자료는 아니다.")
para('N-BaIoT: 정상 파일의 서로 떨어진 행 블록을 초기 학습·검증·서버 개발·임계값·의사결정·후보·최종 평가로 나눴다. Gafgyt 일부만 개발/후보에 사용하고 Mirai 특징은 최종 평가에만 두었다. 9개 장치 정상 FPR과 7개 장치 Mirai AUC의 분모가 다르다.')
para('RT: Thing_Speak와 MQTT_Publish를 기존/신규 정상으로 서로 바꾼 두 시나리오다. Wipro_bulb는 최종 신규 정상에만 포함했다. DOS_SYN_Hping·ARP_poisioning만 개발 공격에 사용하고 나머지 7종은 최종 평가에만 사용했다. 두 시나리오는 같은 원자료를 공유하므로 독립 데이터셋 2개로 세지 않는다.')
para('RT의 동일 특징·상충 라벨 4개 그룹 107행은 적합·선택에서 제외하고 최종 평가에는 전부 남겼다. 어려운 사례를 최종 결과에서 제거하지 않았다. proto/service 범주형과 행 인덱스·정답은 모델 입력에서 제외했다.')
para('CSV 행 순서를 사용한 블록 시나리오다. 실제 시각·물리 호스트·미래 실환경 드리프트를 검증했다는 주장은 하지 않는다. RT는 N-BaIoT와 별도 데이터셋이지만 입력 스키마가 달라 가중치를 직접 전이하지 않았다.')
page('보조 분석 · 정상 분포 차이')
shift=json.loads((H/'development_shift_diagnostic.json').read_text());short={'raw_blocks_v2':'N-BaIoT','rt_thing_to_mqtt_v1':'RT Thing→MQTT','rt_mqtt_to_thing_v1':'RT MQTT→Thing'}
table(['자료','단계','클라이언트','최소 MMD²','중앙 MMD²','최대 MMD²'],[[short[r['dataset']],r['stage'],r['clients'],f(r['MMD_squared_min']),f(r['MMD_squared_median']),f(r['MMD_squared_max'])] for r in shift['summary']],[130,40,65,90,90,90])
para('초기 정상으로만 적합한 로그 스케일러를 고정하고 초기 정상과 해당 단계의 검증 정상 후보를 각 최대 128개씩 뽑아 RBF 커널의 편향 MMD²를 계산했다. 이는 모의 유입 설정을 설명하는 개발 자료 진단이며 학습·라운드 선택·최종 임계값에는 사용하지 않았다.')
para('자료마다 커널 폭이 적응적으로 달라지므로 수치를 공통의 절대 크기나 배율로 해석하지 않는다. 같은 방식의 진단에서 두 시나리오군의 분리가 다르게 나타났다. 정상 분포 차이가 존재해도 초기 탐지기가 이미 충분히 구분할 수 있으며, 반드시 재학습이 좋아진다는 뜻은 아니다.')
para('행 간 상관 때문에 교환 가능성이 보장되지 않는다. 199회 순열의 명목 p값은 원본 JSON에 탐색용으로만 보관했고, 다중 비교를 보정한 확증 검정이나 실제 시간 변화의 증거로 제시하지 않는다.')
para('방법 참고: Gretton et al., A Kernel Two-Sample Test, JMLR 13, 723-773, 2012. https://www.jmlr.org/papers/v13/gretton12a.html')
page('2. 무엇을 바꿨고 무엇을 고정했나')
table(['방식','가중치 / 정상 기준 / 임계값'],[['고정','초기 모델·정상 CEN 기준·초기 정상 q95 모두 유지'],['임계값만','가중치·CEN 고정, max(기존 정상 q95, 현재 신규 정상 q95)'],['정상 기준만','가중치 고정, 신규/기존 정상으로 CEN과 임계값 갱신'],['신규만','검증 신규 정상만 가중치 학습, 기준·임계값 갱신'],['리플레이','같은 최대 표본 예산 중 25%를 초기 정상으로 대체'],['검증 신규 / 리플레이','개발 지표·기존 성능 제약으로 전역 라운드 선택; 거절 시 가중치와 기준 함께 복구'],['조건부 검증','이전 배포 모델의 검증 신규 정상 오탐률이 10% 이하이면 가중치·기준 학습 생략. 임계값 보정은 수행']],[130,373])
para('원본 구현처럼 한 라운드 안에서 다음 클라이언트는 앞 클라이언트가 갱신한 같은 모델 객체를 이어받는다. 모든 클라이언트를 독립 초기화한 표준 병렬 FL 실험으로 해석하지 않는다. 원본의 두 if 전역 종료 조건도 보존했으므로 실제 종료 라운드를 함께 제공한다.')
para('원본 SAE 은닉 50·잠재 11, MSEAvg 수식, 클라이언트 순차 공유 모델, 원본의 두 if 전역 종료 조건을 유지했다. 입력/출력 차원은 데이터 스키마에 맞춘다. 원본 추적 코드는 수정하지 않았다. 논문 수식의 제곱 노름도 별도 개발 조건으로 실험했으며 원본 비제곱 노름과 혼합하지 않았다.')
para('이번 새 분할 실험은 표준화 조건과 로그+표준화 조건 모두 입력을 ±20으로 제한한다. 초기 스케일러는 초기 정상만으로 적합한다. 이 공통 clip과 제한 예산·확장 검증을 포함한 비교이며, 원본 GitHub의 전체 전처리/실행과 완전히 동일하다고 부르지 않는다. 원본 30개 실행은 별도로 완료·보존돼 있다.')
para('주 실험 상한: 초기 30 epoch / 8 round, 갱신 20 epoch / 6 round, 최대 실제 학습 정상 256행/클라이언트/단계, 배치 12, Adam 1e-5, clipping 5. 초기 100 epoch / 20 round와 갱신 100 epoch / 20 round는 별도 민감도다. 원본 전체 학습과 동일 예산이라고 쓰지 않는다.')
para('세 방식 공통 초기 모델은 데이터·전처리·손실·시드별로 별도 학습한 실제 전역 모델이다. 초기 비용은 각 고유 모델당 한 번만 계산한다. 원본 로컬 model.cpt를 전역 가중치로 재사용하지 않았다. 새 스케일러는 초기 정상으로 한 번만 적합하고 고정했다.')
page('3. N-BaIoT 최종 단계 결과')
para('AUC·PR-AUC는 정상과 공격이 모두 있는 클라이언트의 매크로 평균이다. FPR·TPR은 해당 영역의 모든 행 혼동행렬을 합산한 비율이다. 이 수치는 분류 정확도(accuracy)와 다르다. 표준편차와 개별 시드 값은 CSV로 제공한다.')
para('같은 최종 평가 자료에서 각 조건의 완료 시드 평균이다. ‘고정’은 이 장치별 분할에 원본 학습 방식을 적용해 만든 공통 초기 모델이다. 원 논문 표의 숫자를 옮겨 비교한 결과가 아니다.')
for mode,label in [('standard','표준화 조건 · 공통 clip 20'),(primary['mode'],'개발 선택 로그 조건 · 공통 clip 20')]:title(label);result_table(rows('raw_blocks_v2',mode=mode))
para('신규 AUC는 최종 보류한 Mirai 평가다. 기존 AUC는 최종 보류한 Gafgyt 평가다. 기존/신규 정상 FPR도 서로 다른 보류 블록에서 계산했다. 세 시점 중 가장 좋은 값을 골라 보고하지 않고 사전에 고정한 마지막 3단계를 주 결과로 썼다.')
page('보조 분석 · 기존 영역 성능 유지')
for data,label in datasets:
 title(label);rr=[r for r in rows(data) if r['method'] in ['no_update','new_only','replay','safe_new']];baseline=next((r for r in rr if r['method']=='no_update'),None)
 if baseline:table(['방식','시드','기존 AUC','AUC 변화','기존 FPR','기존 TPR'],[[names[r['method']],r['n_seeds'],f(r['old_mean_client_auc_mean']),f(r['old_mean_client_auc_mean']-baseline['old_mean_client_auc_mean']),pct(r['old_fpr_mean']),pct(r['old_tpr_mean'])] for r in rr],[110,40,85,85,85,85])
para('현재 제약은 개발 자료의 평균 AUC와 합산 혼동률을 사용하므로 특정 장치의 악화를 숨길 수 있다. 또한 이전 모델도 현재 정상 자료로 임계값을 보정한 뒤 후보 가중치와 비교한다. 임계값 보정 자체가 초기 고정 임계값보다 TPR을 낮추는 경우까지 막는 절대 TPR 제약은 아니다. 따라서 최종 기존 영역 수치를 별도로 확인해야 한다.')
page('4. 외부 데이터 최종 단계 결과')
for data,label in datasets[1:]:title(label);result_table(rows(data))
para('외부 공격은 개발에서 사용하지 않은 유형을 평가한다. 신규 정상에는 유입 패턴 외에도 최종 전용 Wipro가 있으므로 정상 패턴별 FPR과 공격 유형별 TPR을 함께 보관했다. 개별 기기/호스트 드리프트 실험으로 해석하지 않는다.')
page('5. 유입 단계별 변화')
story.append(Image(str(H/'analysis_v1/FedMSE_Overnight_Curves.png'),width=505,height=459));para('모든 유입 시점 모델을 동일한 마지막 보류 평가 자료에 적용했다. 최종 지표를 다음 시점 학습에 돌려주지 않았다. 시점별 서로 다른 실시간 평가 자료를 쓰는 prequential 실험으로 해석하지 않는다.')
para('가로축은 전역 학습 라운드가 아니라 신규 트래픽 유입 1·2·3단계다. 선은 완료 시드 평균, 음영은 시드 간 표준편차다. 모든 방식은 조건별 공통 초기 가중치로 시작한다. 정밀 수치·분모·시드별 결과는 CSV와 JSON에 남겼다.')
page('6. 전처리와 학습 예산 민감도')
story.append(Image(str(H/'analysis_v1/Preprocessing_Ablation.png'),width=505,height=170));para('전처리 변경은 새 초기 모델을 필요로 한다. 따라서 로그 전처리+재학습과 표준화 조건+고정의 차이 전체를 재학습 효과로 주장할 수 없다. 같은 전처리 안의 고정/재학습 차이와 전처리 간 차이를 분리한다.')
for data,label in datasets:
 rr=[r for r in rows(data,'budget100') if r['method'] in ['no_update','new_only','replay','safe_new']]
 if rr:title(label+' · 100/20 상한');result_table(rr)
para('100/20 상한 민감도는 RT 두 방향×3시드=6개 실행만 완료했다. N-BaIoT 원자료 3시드는 실행 전 시간 추정으로 시작하지 않았으며, 실패하거나 끝난 실험으로 합산하지 않았다.')
para('상한에 도달하기 전 원본 조기 종료가 발생할 수 있다. 실제 optimizer step과 종료 라운드를 함께 기록했다. 시드가 부족한 조건은 완료 수를 그대로 표시하며 주 실험 5시드와 합치지 않는다.')
page('7. 검증 비용과 오검증 민감도')
for kind,label in [('query25','후보 무작위 25%만 검증'),('poison05','정상 승인 중 공격 목표 5%')]:
 if kind=='poison05':page('7. 검증 비용과 오검증 민감도 · 오승인')
 title(label);body=[]
 for data,dlabel in datasets:
  for r in rows(data,kind):body.append([dlabel,names[r['method']],r['n_seeds'],f(r['new_mean_client_auc_mean']),pct(r['new_fpr_mean']),pct(r['new_tpr_mean'])])
 if body:table(['자료','방식','시드','신규 AUC','FPR','TPR'],body,[125,100,35,82,82,82])
 else:para('시간 한도 내 최종 검증 완료 결과가 없어 성능 결론을 제시하지 않는다.')
 para('보호된 초기 정상·현재 단계 검증/임계값/개발 자료는 깨끗하게 유지했다. 후보 승인만 교란한 실험이며 악의적 FL 업데이트 방어 검증은 아니다. 25%는 후보 검증 비율이지 전체 라벨 사용량 25%가 아니다.')
page('보조 분석 · 실제 승인 오류와 학습 오염')
ac=json.loads((H/'analysis_v1/approval_cost_analysis.json').read_text());body=[]
for data,label in datasets:
 for r in ac['summary']:
  if r['data']==data and r['setting']=='poison05':body.append([label,names[r['method']],pct(r['approval_contamination_mean']),pct(r['training_contamination_mean']),pct(r['CEN_reference_contamination_mean'])])
if body:table(['자료','방식','승인 오염','학습 표본 오염','배포 CEN 오염'],body,[125,105,85,95,95])
para('공격 목표 5%는 후보 정상 승인 집합의 목표 비율이다. 표본 상한·리플레이 대체·갱신 거절에 따라 실제 학습/배포 기준의 공격 비율은 달라진다. 원자료 정답과 저장된 입력 ID를 대조해 실제 비율을 계산했다. 거절된 학습 제안의 오염과 실제 배포 기준 오염도 구분했다.')
para('후보 25% 검증은 클라이언트별 올림 때문에 정확히 25.000%가 아닐 수 있다. 보호된 현재 정상 검증·임계값·의사결정 자료는 별도로 추가 제공된다. 한 번 확인한 후보 라벨을 여러 방법이 공유하므로 방법 수만큼 검증 비용을 부풀리지 않았다. 전체 쿼리·행 수는 approval_cost CSV에 제공한다.')
page('8. 가중치 학습과 정상 기준 변화 분리')
para('이 정상 기준 대조군은 CPU에서 CEN을 재계산했고, 주 학습은 GPU에서 수행한 후 CPU로 평가했다. 공통 가중치 해시와 입력 ID를 확인했으며 수치 허용오차를 검증했지만 비트 단위로 같은 연산 환경은 아니다.')
para('추가 대조군은 초기 가중치를 고정하고, 신규만 재학습과 정확히 같은 현재 단계 신규 정상 표본으로 CEN 기준과 임계값만 갱신한다. 신규만 재학습과 비교하면 가중치 학습의 추가 효과를 분리할 수 있다. 검증 방식의 롤백은 이전 기준을 유지하므로 전체 시스템 효과로 해석한다.')
for data,label in datasets:
 rr=matched_comparison(data,'reference_only_new',['new_only','safe_new'])
 if rr:title(label);result_table(rr)
page('9. 공격 라벨 없는 갱신 선택')
para('공격 개발 라벨을 선택 기준에서 제거한 추가 비교다. 기존/신규 정상 재구성 MSE의 로그 평균을 줄이되, 기존 정상 MSE가 초기의 1.05배를 넘거나 기존/신규 FPR이 이전보다 2%p 넘게 증가하면 거절한다. 후보 정상 검증자와 보호 정상 자료는 여전히 필요하다. 공격 AUC·TPR은 선택에 쓰지 않고 관찰용으로만 기록했다.')
for data,label in datasets:
 rr=rows(data,'normal_only')
 if rr:title(label);result_table(rr)
para('공통 로그 전처리 조합의 사전 선택에는 N-BaIoT의 알려진 공격 개발 라벨이 사용됐다. 따라서 정상 전용 방식도 전체 연구 파이프라인이 공격 라벨 없이 선택됐다고 쓰지 않는다. 이번 제거 범위는 온라인 전역 라운드 선택 기준이다.')
para('깨끗한 정상 자료만으로 갱신을 선택하면 공격 탐지 성능을 보장할 수 있는지 시험한 민감도다. 원래 공격 개발 라벨을 사용하는 검증 방식과 동일한 강건성 주장을 하지 않는다.')
page('추가 비교 · 기존 정상 표현 보존 손실')
para('초기 전역 모델의 기존 정상 잠재 표현을 고정 teacher로 두고, 기존 리플레이 표본의 표현 차이에 가중치 0.1을 주었다. 정렬 방식은 신규 정상 표현을 초기 정상 중심으로 유도하는 항을 가중치 0.05로 추가했다. 계수는 과거 별도 V3 탐색에서 사용한 값이며 이번 최종 자료로 선택하지 않았다.')
for data,label in datasets:
 title(label);rr=matched_comparison(data,'gpu_objectives',['replay','safe_replay']);result_table(rr)
para('같은 GPU·초기 모델·참여 순서·실제 입력 ID·학습 상한을 쓰는 기존 주 실험의 리플레이/검증 리플레이를 대조군으로 재사용했다. 추가 손실 외의 집계·모델 구조는 유지했다. 신경망 손실을 바꾼 연구 조건이며 원본 재현 결과로 합산하지 않는다. 기존 표현 보존이나 정상 정렬이 공격 구분을 해칠 수도 있어 실패 결과를 함께 남겼다.')
page('10. 추가 학습 비용과 갱신 수락')
initial_unique={}
for run in A['runs']:
 for source in run['initial_sources'].values():initial_unique[source['path']]=source['cached_initial_training_seconds']
para(f'보고서 포함 조건의 고유 공통 초기 모델은 {len(initial_unique)}개이며, 캐시에 기록된 초기 학습 시간 합은 {sum(initial_unique.values())/60:.1f}분이다. 같은 초기 모델을 재사용한 방식·민감도마다 중복 합산하지 않았다. 선택하지 않은 개발 조건과 소규모 검증 비용은 이 합계에 포함하지 않았다.')
for data,label in datasets:
 title(label);rr=rows(data);body=[]
 for r in rr:
  if r['method'] not in ['new_only','replay','safe_new','trigger_safe']:continue
  trials=[x for x in A['records'] if x['data']==data and x['mode']==primary['mode'] and x['kind']=='main30' and x['method']==r['method']]
  attempted=sum(x['trained'] for x in trials);accepted=sum(x['trained'] and x['accepted'] is True for x in trials);skipped=sum(x['trigger_skipped'] for x in trials)
  body.append([names[r['method']],r['n_seeds'],f(r['seconds_mean'],1),f(r['steps_mean'],0),f'{accepted}/{attempted}',skipped])
 table(['방식','시드','추가 초','실제 step','채택/시도','사전 생략'],body,[110,40,90,90,90,70])
para('고정·임계값만 방식의 optimizer 학습 step은 0이다. 시간과 step은 마지막 단계까지 누적한 시드 평균이다. 채택/시도와 생략 횟수는 세 유입 단계 전체 합이다. 갱신 후 거절은 이미 학습 비용을 지불한 것이며 사전 생략과 다르다. 임계값/CEN 재계산·추론·모델 직렬화는 이 optimizer 학습 시간에 모두 포함되는 총 운영 비용이 아니다.')
page('11. 근거·비용·통계 해석')
para('GPU 저장값과 CPU 독립 재계산의 CEN·q95 비교는 rtol=2e-4, atol=2e-6을 사용했다. 스케일러 재적합 파라미터·가중치/파일 해시는 정확 일치, 예측 CSV 지표는 1e-12 이내로 확인했다. 부동소수점 근사 일치와 바이트 일치를 구분한다.')
para('분할 ID·정확/보수적 특징 중복, 원자료/실행 코드 해시, 정상 학습, 현재 단계 공개, 공통 초기 모델·참여 순서·표본 예산, 무갱신 불변, 라운드 선택·기준 복구, CEN·임계값 재계산을 독립 검증했다. 최종 CSV에서도 모든 기대 ID·정답·혼동행렬·AUC/PR-AUC를 다시 계산했다.')
para('‘정상 여부를 확인할 수 있는 검증자 보조 환경’이다. 깨끗한 주 실험은 실제 공격을 정상 적합에서 제외한다. 오승인 민감도는 계획대로 일부 공격을 정상 승인해 넣는다. 주 갱신 방식은, 갱신 선택에는 알려진 공격 개발 라벨을 사용한다. 완전한 무라벨·미지 공격만으로 가능한 선택이라고 주장하지 않는다. 초기 서버 정상 개발·임계값 자료도 공통으로 필요하다.')
mem=H/'memory_observer_status.json'
if mem.exists():
 M=json.loads(mem.read_text());v=list(M['processes'].values())
 if v:para(f"실제 메모리 관측: {len(v)}개 학습 프로세스에서 Windows peak working set 최대 {max(x['peak_working_set_bytes'] for x in v)/2**30:.2f} GiB, 관측 private usage 최대 {max(x['max_sampled_private_bytes'] for x in v)/2**30:.2f} GiB. 관측 시작 전 완료 실행은 소급 측정하지 않았다. 이 값은 GPU VRAM이나 시스템 전체 사용량이 아니다.")
para('메모리 관측은 초기 학습과 여러 방식이 함께 실행된 프로세스 단위다. 공통 teacher 복사·은행·캐시 등 분석용 오버헤드가 있어 방식별 순수 배포 메모리나 갱신 전용 peak로 해석하지 않는다. 벽시계 학습 시간도 CPU 검산과 병행한 이 노트북 실행 비용이며 전용 장비의 성능 벤치마크는 아니다.')
para('리플레이 특징 바이트는 보관 행×특징 수×dtype 크기로 따로 집계했다. 총 RSS·통신량으로 바꾸어 부르지 않는다. 구현은 공통 분석을 위해 초기 정상 은행을 모든 방식에서 메모리에 올리므로 논리적 리플레이 용량과 실제 메모리 차이가 곧바로 같지는 않다.')
para('시드 반복은 고정된 평가 행에서 학습 변동을 측정한다. 새로운 5개 수집 환경을 검증한 것이 아니다. 평균·표준편차·짝지은 시드 차이·최솟값/최댓값을 제공했다. 5개 비영 차이의 정확 양측 부호 반전 검정은 최소 p=0.0625이므로 강한 유의성 표현을 피한다.')
page('12. 평균 뒤에 숨은 실패 조건')
subgroups=json.loads((H/'analysis_v1/subgroup_summary.json').read_text())
for data,label in datasets:
 title(label);rr=[x for x in subgroups if x['data']==data and x['kind']=='main30' and x['method']=='safe_new' and x['domain']=='new']
 normal=sorted([x for x in rr if x['type']=='normal'],key=lambda x:x['mean_rate'],reverse=True)[:2];attack=sorted([x for x in rr if x['type']=='attack'],key=lambda x:x['mean_rate'])[:2]
 table(['구분','최악 2개 정상 패턴 / 공격 유형','평가 행/시드','비율'],[[('오탐' if x['type']=='normal' else '탐지'),x['group'],x['test_rows_per_seed'],pct(x['mean_rate'])] for x in normal+attack],[45,295,85,75])
para('주 방법의 신규 영역에서 오탐률이 가장 높은 정상 패턴과 탐지율이 가장 낮은 공격 유형을 표시했다. 이것으로 모델을 다시 선택하지 않았다. 각 시드는 같은 평가 행을 반복하므로 평가 행 수를 시드 수만큼 부풀리지 않았다. 나머지 모든 장치·유형도 subgroup CSV에 제공한다.')
page('13. 재현 구조와 남은 연구 과제')
table(['파일','역할'],[['prepare_raw_blocks_v2.py / prepare_rt_blocks.py','출처·분할·중복·모순 라벨 처리'],['run_study.py / federated_core.py','원본 학습 구조를 보존한 순차 재학습·라운드 선택'],['study_data.py / transforms.py','미래 단계 지연 공개, 초기 정상 스케일러 고정'],['study_metrics.py','정상 CEN·q95 임계값·타입별 평가·모델 번들'],['run_budget_study.py / run_approval_study.py','예산 확대·후보 검증 비용·오승인 민감도'],['verify_*.py / audit_costs.py','독립 검산, 라벨 요구량·리플레이 페이로드'],['family_selection_frozen.json / analysis_v1','사전 선택 근거, 완료 시드 전체 최종 결과']],[230,273])
para('개선의 근거가 충분한 조건과 악화/동률인 조건을 분리해야 한다. 다른 공개 자료로 효과가 이전되는지, 최종 전용 정상/공격 유형에서 실패하는지, 보호 검증 자료가 없을 때도 동작하는지는 별도 연구 문제다. 이 결과만으로 KCI 투고 가능 수준이나 실환경 일반화를 보장하지 않는다.')
para('출처: FedMSE arXiv:2410.14121v2, https://arxiv.org/html/2410.14121v2 · UCI N-BaIoT, DOI 10.24432/C5RC8J · UCI RT-IoT2022, DOI 10.24432/C5P338. 두 UCI 자료는 CC BY 4.0이며 다운로드 원본·해시·공식 출처를 로컬 manifest에 보관했다.')
para('이전 V3의 가공/PCA 유입 실험은 별도 탐색 결과다. 이번 원자료/외부 데이터의 수치와 합산하지 않는다. 원본 30개 재현 실험, 지속학습 확장, 이번 추가 검증은 서로 다른 분할과 목적을 갖는다.')
page('14. 선행연구와 투고 전 보강점')
para('재학습·정상 패턴 변화 적응 자체는 이미 연구된 주제다. 이번 실험의 내부 대조군이 좋아도 기존 적응형 탐지 연구 전체보다 우월하다는 뜻은 아니다. 아래 연구를 검토했지만 이번 야간 실험에서 동일 조건의 원 구현 벤치마크를 재현한 것은 아니다.')
table(['선행연구','이번 연구와의 관계'],[['OWAD · NDSS 2023','정상 패턴 변화의 탐지·설명·적응과 수동 라벨 비용을 다룬다. 단순 재학습 추가를 신규성으로 주장하기 어렵다.'],['When Model Meets New Normals · AAAI 2024','변하는 정상 패턴에 대한 테스트 시점 적응을 연구한다. 우리 행 블록 실험은 그 시계열 평가를 그대로 재현하지 않는다.'],['Unsupervised Model Selection · ICLR 2023','예측 오차·모델 중심성·합성 이상치 등 불완전한 대리 지표를 결합한다. 정상 MSE 하나로 공격 탐지 성능을 보장하지 않는다.']],[190,313])
para('투고 전 우선순위: ① 공격 라벨 선택/정상 전용 선택을 동일 예산에서 비교 ② 원본 고정뿐 아니라 기존 적응형 방법의 재현 대조군 추가 ③ 실제 시각 또는 장치 전이 근거가 있는 새 수집 자료 검증 ④ 라벨 비용·거절된 갱신·실패 조건까지 포함한 반복 결과 보고. 이번 결과가 이를 모두 충족했다고 표현하지 않는다.')
para('원문: https://www.ndss-symposium.org/ndss-paper/anomaly-detection-in-the-open-world-normality-shift-detection-explanation-and-adaptation/ · https://arxiv.org/abs/2312.11976 · https://arxiv.org/abs/2210.01078')
def footer(c,doc):
 c.saveState();c.setFont('KR',8);c.setFillColor(colors.HexColor('#647080'));c.drawString(43,24,'20222038 이진호 | FedMSE 추가 재학습 연구');c.drawRightString(A4[0]-43,24,str(doc.page));c.restoreState()
doc=SimpleDocTemplate(str(PDF),pagesize=A4,rightMargin=43,leftMargin=43,topMargin=42,bottomMargin=43,title='FedMSE Overnight Retraining Study',author='20222038 이진호');doc.build(story,onFirstPage=footer,onLaterPages=footer);print(str(PDF))
