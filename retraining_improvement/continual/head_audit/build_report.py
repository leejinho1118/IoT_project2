"""검증된 수치와 출처로 A안 한계·새 후보·남은 위험을 정리하는 8쪽 PDF."""
from pathlib import Path
import json,html,datetime,subprocess
from reportlab.platypus import SimpleDocTemplate,Paragraph,Table,TableStyle,Spacer,PageBreak,Preformatted
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
D=Path(__file__).resolve().parent;O=D/'output/reports';O.mkdir(parents=True,exist_ok=True)
e=json.loads((D/'evidence_summary.json').read_text(encoding='utf-8'));risk=json.loads((D/'risk_audit.json').read_text(encoding='utf-8'));sources=json.loads((D/'sources.json').read_text(encoding='utf-8'))
base=e['baseline'];new=e['candidate'];by={r['method']:r for r in e['residual_summary']};phase1={r['method']:r for r in e['phase1_summary']}
pdfmetrics.registerFont(TTFont('Body','C:/Windows/Fonts/malgun.ttf'));pdfmetrics.registerFont(TTFont('Bold','C:/Windows/Fonts/malgunbd.ttf'));pdfmetrics.registerFont(TTFont('Code','C:/Windows/Fonts/consola.ttf'));pdfmetrics.registerFontFamily('Body',normal='Body',bold='Bold',italic='Body',boldItalic='Bold')
W,H=A4;M=47;WIDTH=W-2*M;NAVY=colors.HexColor('#203c57');MUTED=colors.HexColor('#60768c');LIGHT=colors.HexColor('#edf4f8');GRID=colors.HexColor('#d6e1e8')
styles={k:ParagraphStyle(k,fontName=f,fontSize=s,leading=l,spaceAfter=8,wordWrap='CJK',textColor=c) for k,f,s,l,c in [('title','Bold',18,25,NAVY),('h','Bold',11.5,17,NAVY),('body','Body',10,15.5,colors.HexColor('#182430')),('small','Body',8.2,12.5,MUTED),('cell','Body',8.5,12.8,colors.HexColor('#182430')),('head','Bold',8.5,12.8,NAVY)]};styles['h'].keepWithNext=True
story=[];md=[];pages=0

def p(t,style='body'):story.append(Paragraph(t,styles[style]));md.append(t.replace('<br/>','\n'))
def h(t):p(t,'h')
def page(t):
 global pages
 if pages:story.append(PageBreak())
 pages+=1;p(t,'title')
def table(headers,rows,widths):
 data=[[Paragraph(html.escape(str(v)),styles['head' if i==0 else 'cell']) for v in row] for i,row in enumerate([headers]+rows)]
 t=Table(data,colWidths=[WIDTH*w/sum(widths) for w in widths],repeatRows=1)
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),LIGHT),('LINEABOVE',(0,0),(-1,0),.8,NAVY),('LINEBELOW',(0,0),(-1,-1),.4,GRID),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]));story.extend([t,Spacer(1,10)]);md.extend(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])
def percent(x):return f'{x*100:.2f}%'
def metric(r,d,k):return r['metrics'][d][k]['mean']
def name_row(label,r):return [label,f"{metric(r,'new','mean_client_auc'):.5f}",percent(metric(r,'new','fpr')),percent(metric(r,'new','tpr'))]
page('01 A안이 최선인가: 감사와 개선 결과')
p('<b>20222038 이진호</b> / '+datetime.datetime.now().astimezone().isoformat(timespec='seconds'))
p('<b>A안은 전역 최선이 아니다.</b> 이전 다섯 안에서 정한 개발 점수로 선택됐을 뿐이며, 점수 가중치를 바꾸면 B·E도 선택된다. 새 원인 분해 실험을 수행했고, 입력 특징을 보존한 정상 기억 후보가 A안보다 좋은 결과를 보였다.')
table(['구성 / 동일 A 가중치','신규 AUC','신규 FPR','신규 TPR'],[name_row('A 원래 CEN',base),name_row('임계값만 보정',by['threshold_updated']),name_row('입력+SAE 잔차 기억',new),name_row('입력 kNN 진단 기준',by['input_knn5']),name_row('log 입력 kNN 진단',by['log_input_knn5'])],[190,100,105,106])
p('50클라이언트·5시드의 마지막 단계에서 평가한 평균이다. 임계값만 높이면 공격도 놓치므로 오탐 감소만으로 성공이라고 하지 않았다. 진단 기준은 원시 입력을 쓰며 FedMSE 개선으로 둔갑시키지 않는다.')
h('이번에 실제로 수행한 범위')
p('SAE/MSEAvg 가중치는 동결했다. 정상 참조만으로 탐지 헤드를 새로 적합하고 정상 보정 자료로 임계값을 정했다. 인코더를 다시 신경망 학습한 실험이 아니라, 탐지층 및 입력 우회 표현을 바꾼 통제 비교다. 1차 9구성, 잔차 후속 7구성을 검증했다.')
p('새 후보의 신규 AUC는 5시드 모두 A보다 높았다. 하지만 같은 보류 풀을 이전 연구에서도 사용했으므로 탐색적 후속 결과이며, 새 외부 데이터에서 확인한 일반화 성능으로 주장하지 않는다.','small')
page('02 A안 선택은 얼마나 확실했는가')
table(['기존 AUC 비중','FPR 벌점 계수','개발 점수 선택'],[[f"{r['old_weight']:.2f}",f"{r['fpr_penalty']:.2f}",r['winner']] for r in risk['objective_sensitivity'] if (r['old_weight'],r['fpr_penalty']) in [(.5,.1),(.5,0),(.75,.1)]],[150,150,201])
p('이전 기준은 기존·신규 AUC를 반씩 평균하고 과도한 FPR에 0.1의 벌점을 주는 방식이었다. 정상 오탐 수를 실제 운영 비용으로 환산한 기준이 아니므로 58% 오탐도 상대적 최고가 될 수 있었다. A를 절대 최적이라고 부르는 근거가 되지 않는다.')
h('이번 후보 선택 기준을 먼저 고정')
p('개발 자료에서 두 영역 평균 FPR 각각 10% 이하, 가장 나쁜 시드의 FPR 15% 이하, 두 영역 평균 TPR 각각 80% 이상을 동시에 요구했다. 통과 후보 중 낮은 쪽 TPR이 높은 것을 선택했다. 이 값은 연구용 선별 기준이며 운영 서비스의 허용 오탐 기준이 아니다.')
p('입력만 쓰는 진단 기준은 주 후보 선택에서 제외했다. 잔차 블록 가중치 0.25·0.5·1.0 중 개발 자료로 0.25를 고정한 뒤 보류 평가를 열었다. 최종 결과를 보고 가중치를 다시 고르지 않았다.')
h('남는 선택 편향')
p('다섯 시드는 같은 분할을 공유한다. 여러 후보를 개발 자료에서 고른 탐색 과정, 앞선 V2와 A~E에서 이미 본 보류 풀, 초기 모델의 과거 검증 자료 사용 이력이 있다. 작은 시드 표준편차를 외부 강건성으로 해석하거나 통계적 유의성을 주장하지 않는다.')
p('후보의 파라미터 적합은 정상만 사용했지만, 선택에는 라벨 있는 개발 공격의 TPR·AUC를 사용했다. 공격 라벨 없이 선택한 설정이라는 뜻은 아니다. 향후 확인 실험에서는 현재 설정을 미리 고정해야 한다.','small')
page('03 논문과 코드: 정확히 무엇을 재현했나')
table(['항목','확인한 차이와 근거'],[['축소 손실','논문 Eq4·Algorithm1은 잠재 벡터 제곱 L2를 표기. 공개 코드는 제곱하지 않은 L2 norm 평균을 사용'],['수치 직접 검사','latent=(3,4), lambda10, 복원 오차0에서 원본 shrink_loss=50. 논문 표기 제곱 항은250'],['로컬 모델 전달','논문 GatewayUpdate는 초기 가중치 수신으로 설명. 원본 main.py는 같은 전역 객체를 클라이언트에 순차 전달'],['전역 조기 종료','원본 두 if가 개선 시에도 카운터를 증가. 원본 재현 및 후속 비교에서 그대로 유지']], [115,386])
p('공개 코드 기준 재현과 논문 수식·알고리즘 기준 재현은 구분해야 한다. 지금까지 원본 src는 수정하지 않고 공개 저장소와 동일하게 실행했다. 이번 감사 결과를 원본을 고쳐 실행했다고 해석하지 않는다. [1]')
h('원인과 의심을 구분')
p('손실 식 차이는 직접 확인된 사실이다. 그러나 그 차이가 현재 성능 저하의 단일 원인이라는 것은 아직 증명하지 않았다. 제곱 손실, 공유 모델 전달, 전역 조기 종료를 동시에 바꾸면 원인을 분리할 수 없으므로 각각 별도 통제 실험이 필요하다.')
h('자료의 재현 경계')
p('현재의 A/B는 개발 특징 PCA로 만든 모의 정상 분포 변화다. 실제 시간 순서나 물리 기기 매핑이 아니다. UCI 원자료에는 9개 물리 기기 정보가 있지만, 현재 준비된 50클라이언트 CSV에 그 매핑이 복원된 것은 아니다. [8]')
p('원본 코드·기존 결과 65개와 현재 원본 CSV 150개의 해시가 그대로임을 다시 확인했다. 새 코드는 continual/head_audit에만 추가했다.','small')
page('04 원인 분해: 임계값만의 문제가 아니다')
labels={'centroid_original':'원래 CEN','threshold_updated':'임계값 갱신','reference_balanced':'균형 정상 CEN','centroid_two':'두 정상 중심','latent_knn5':'잠재 kNN5','latent_lof20':'잠재 LOF20','latent_mahalanobis':'잠재 공분산 거리','input_knn5':'입력 kNN 진단','log_input_knn5':'log 입력 진단'}
table(['탐지 구성','신규 AUC','신규 FPR','신규 TPR'],[name_row(labels[m],phase1[m]) for m in labels],[190,100,105,106])
p('임계값만 바꾼 경우 원래 점수와 AUC는 동일하다. 오탐은 줄지만 TPR이 크게 감소해 정상과 공격의 점수 순위·표현도 문제라는 근거다. 다중 중심과 잠재 거리 변경도 저오탐·고탐지를 동시에 달성하지 못했다.')
clip=e['input_saturation'];p(f"표준화 뒤 범위 제한에 걸린 특징 비율은 개발 기존 정상 평균 {clip['old_normal']['mean_feature_clip_fraction']*100:.2f}%, 신규 정상 {clip['new_normal']['mean_feature_clip_fraction']*100:.2f}%, 공격 {clip['attack']['mean_feature_clip_fraction']*100:.2f}%였다. 전처리 손실 가능성도 남는다.")
p('입력 kNN과 잠재 kNN은 같은 정상 참조와 같은 k=5를 썼다. 입력 기준이 더 좋다는 것은 압축 표현/거리 기하가 병목일 수 있다는 강한 진단이다. 다만 원인이 손실 항 하나라고 확정한 것은 아니다.','small')
page('05 새 후보: 입력 우회와 정상 기억')
p('원래 입력을 잃지 않게 115차원 입력과 11차원 SAE 잠재 벡터를 결합한다. 여기서 잔차는 입력 우회 특징이라는 뜻이며 ResNet 블록을 새로 훈련했다는 뜻이 아니다.')
cs=ParagraphStyle('code',fontName='Code',fontSize=9,leading=13,spaceAfter=12)
story.append(Preformatted('r(x) = [ standardize(x) / sqrt(115),\n         0.25 * standardize(SAE(x)) / sqrt(11) ]\nscore(x) = mean distance to 5 nearest normal memories',cs))
p('블록별 표준화는 두 정상 참조만으로 적합한다. 기존 정상 최대128개와 현재 검증 신규 정상 최대128개를 균형 보관하고, kNN 평균 거리를 이상 점수로 쓴다. 예측 때 실제 A/B 구분이나 정답 라벨을 읽지 않는다.')
p('임계값은 별도의 기존·신규 정상 보정 풀 점수의 각각95백분위수 중 큰 값이다. 정상 분포 두 개를 보정에 반영하되, 교환가능성·독립성 조건을 만족한다고 확인하지 않았으므로 분포무관 오탐 보장을 주장하지 않는다. [3,6]')
table(['영역','후보 AUC 평균 ± SD','후보 FPR','후보 TPR'],[[d,f"{new['metrics'][k]['mean_client_auc']['mean']:.5f} ± {new['metrics'][k]['mean_client_auc']['std']:.5f}",percent(metric(new,k,'fpr')),percent(metric(new,k,'tpr'))] for d,k in [('기존','old'),('신규','new')]],[65,205,115,116])
p('정상 특징 기억과 최근접 거리라는 아이디어는 PatchCore 등에도 사용된다. 해당 연구는 이미지 이상 탐지이며 그 성능이 IoT에 그대로 적용된다는 근거는 아니다. 여기서는 위 표의 자체 통제 실험으로 판단한다. [2]')
p('입력만 쓰는 kNN도 거의 같은 결과를 보였다. SAE의 추가 이득이나 이 복합 모델의 필요성은 아직 입증되지 않았다. 더 단순한 방법과 비용을 함께 비교해야 한다.','small')
page('06 불안한 부분: 검증자가 틀리면')
table(['실제 공격 참조 혼입','가드 없음: 신규 TPR','가드 적용: 신규 TPR'],[[percent(r['actual_contamination']),percent(r['unprotected_new_tpr']),percent(r['guarded_new_tpr'])] for r in e['verifier_error_stress']],[145,178,178])
p('개발 평가 전용 스트레스다. 정상으로 잘못 검증된 후보 공격을 참조 기억에 주입했고, 평가 공격과 ID를 분리했다. 약4.79% 혼입만으로 TPR이 89.3%에서19.3%로 감소했다. 오탐률은 약7%로 유지돼 낮은 FPR만으로 오염을 알아낼 수 없었다.')
h('실제로 구현·시험한 보완')
p('초기 정상과 로컬 신규 정상 검증 풀의 최대32개 신뢰 앵커로 입력 kNN 지지 영역을 만들고, 별도 정상 보정 점수99백분위수 밖의 후보를 격리했다. 가드는 후보의 실제 정답을 읽어 거르는 것이 아니라 특징 거리로 승인한다. 같은 혼입에서 TPR을 약65%로 회복했지만 충분하지 않았다.')
h('가드의 대가와 남은 한계')
p('깨끗한 조건에서도 가드 TPR은 약87%로 약89%보다 낮았다. 신뢰 앵커와 보정 자료 자체가 정상이라고 믿을 수 있어야 하며, 그 라벨까지 틀리면 보호를 기대할 수 없다. 미지 정상 패턴을 거부하는 적응 지연도 가능하다.')
p('이 스트레스는 참조 기억의 오판 민감도다. SAE 가중치 자체의 오염, 적응적 공격자, 악의적 연합 업데이트 방어를 검증한 것은 아니다. 주요 깨끗한 모델·결과는 덮어쓰지 않았다.','small')
page('07 검증·자원 비용·실사용 한계')
table(['검증','실제 완료'],[['원인 분해','90개 개발/보류 평가, 9,000개 클라이언트 AUC 독립 재계산'],['잔차 후보','70개 개발/보류 평가, 7,000개 AUC 독립 재계산'],['보정·참조','정상 참조·보정 ID 검사, 원본 CSV에서 임계값 재계산, 고정 A 인코더 확인'],['단독 추론',f"115열664행 CPU 추론, 보류{e['independent_inference']['matched_heldout_rows']}행 판정 일치, 최대 점수 차이 {e['independent_inference']['max_absolute_score_difference']:.2g}"],['안전 규칙','10개 단위 검사, 임계값 변경 시 AUC 불변, 부호·정상 보정 분리·직렬화 확인']], [115,386])
c=e['head_costs'];p(f"클라이언트당 헤드 직렬화 평균은 원래 CEN {c['centroid_original']['mean_serialized_head_bytes_per_client']/1024:.1f} KiB, 잔차 기억 {c['residual_knn_w025']['mean_serialized_head_bytes_per_client']/1024:.1f} KiB다. 기존 입력 scaler·신경망·프로세스 전체 RSS·GPU 피크 메모리는 이 비교에 포함하지 않는다. 기억 기반 방법의 비용 증가는 실제다.")
p('재계산에서 최대 약7.3e-7 점수 차이와 임계값 경계의 판정10개 변화가 관찰됐다. 엄격한 기준 재현 검사 실패를 보존하고, 현재 모든 헤드는 같은 새 임베딩을 공유했다. 큰 성능 차이를 이 미세 오차로 설명할 수는 없지만 경계 판정 안정성은 별도 위험이다.')
h('오탐 7%는 운영 준비 완료가 아니다')
p(f"공격 비율을 가상으로0.1%라 두면 신규 FPR·TPR로 계산한 경보 정밀도는 약 {e['hypothetical_base_rate']['implied_precision']*100:.2f}%다. 실제 운영 비율을 측정한 수치가 아니라 기저율 문제를 보여주는 예다. 높은 AUC만으로 실용성을 판단하면 안 된다. [7]")
page('08 결론·후속 우선순위·출처')
p('추천 연구 방향은 <b>검증 정상의 제한된 기억을 이용한 지속적 탐지층 갱신과 안전한 자료 승격</b>이다. A보다 좋은 탐색 후보는 확인했지만 전역 최선·외부 강건성·공격 오염 해결을 확보한 것은 아니다.')
table(['우선순위','다음에 확인할 것'],[['1','현재 가중치0.25·k5를 고정한 기기별 또는 새 자료 확인. 같은 평가 풀 반복으로 일반화를 주장하지 않음'],['2','입력 kNN 단순 기준 대비 SAE 추가 이득·지연·메모리 비교. 이득이 없으면 복잡성 축소'],['3','논문 제곱 손실과 원본 norm 손실을 별도 통제 학습으로 비교. 원본 src는 계속 보존'],['4','신뢰 정상 확보 비용·검증자 오류·초저오탐 보정 자료량·가드 적응 지연 평가']], [65,436])
p('근거 파일: risk_audit.json · evidence_summary.json · paper_code_numeric_check.json · comparison.csv · output/residual_full_v1/validation.json · output/trusted_guard_stress_v1/. 코드·모델 버전·실패 기록과 보류 평가 출처를 보존했다.','small')
for i,s in enumerate(sources,1):
 p(f"[{i}] <link href=\"{html.escape(s['url'],quote=True)}\" color=\"#23638a\">{html.escape(s['title'])}</link>",'small')
p('근거의 구분: [1]은 공개 논문과 수식, [2~6]은 아이디어·도구·가정의 근거, [7]은 운영 해석, [8]은 원자료 출처다. 이 출처들이 이번 IoT 성능 개선을 대신 증명하지 않으며, 성능 근거는 저장된 자체 실험과 독립 검증이다.','small')

def footer(canvas,doc):
 canvas.setFont('Body',8);canvas.setFillColor(MUTED);canvas.drawString(M,H-27,'FedMSE A안 감사와 후속 개선 / 2026.11.14 / 20222038 이진호');canvas.drawString(M,26,'확인된 개선 · 반증과 실패 · 남은 위험을 함께 기록');canvas.drawRightString(W-M,26,f'{doc.page} / 8')
path=O/'FedMSE_Audit_and_Head_Improvement.pdf'
SimpleDocTemplate(str(path),pagesize=A4,leftMargin=M,rightMargin=M,topMargin=48,bottomMargin=43).build(story,onFirstPage=footer,onLaterPages=footer)
(O/'FedMSE_Audit_and_Head_Improvement.md').write_text('\n\n'.join(md),encoding='utf-8')
from pypdf import PdfReader
r=PdfReader(path);assert len(r.pages)==8
assert all('20222038 이진호' in p.extract_text() for p in r.pages)
qa=D/'report_qa';qa.mkdir(exist_ok=True);subprocess.run(['pdftoppm','-scale-to','1150','-png',str(path),str(qa/'page')],check=True)
import pdfplumber
with pdfplumber.open(path) as pdf:
 for i,p in enumerate(pdf.pages):assert all(c['x0']>=-1 and c['x1']<=W+1 for c in p.chars),('overflow',i)
print(path)
