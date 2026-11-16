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
load=lambda p:json.loads(p.read_text(encoding='utf-8'))
a={r['method']:r for r in load(D/'output/full_v1/final_summary.json')};b={r['method']:r for r in load(D/'output/metrics_v1/final_summary.json')};candidate=b['input_chebyshev']
page('01 후속 탐색: 더 단순한 새 후보')
p('<b>20222038 이진호</b> / '+datetime.datetime.now().astimezone().isoformat(timespec='seconds'))
p('상관관계·밀도·트리 방식과 거리 선택을 추가로 구현했다. <b>입력 전용 Chebyshev kNN</b>은 앞서 동결했던 입력+SAE 후보보다 신규 AUC·FPR·TPR 세 지표가 모두 좋았다. 다만 SAE를 쓰지 않으므로 FedMSE 신경망의 개선 효과라고 부를 수 없다.')
table(['같은 현재 정상 기억','신규 AUC','신규 FPR','신규 TPR'],[name_row('앞선 입력+SAE 후보',a['residual_knn_w025']),name_row('입력 Euclidean kNN',a['input_knn5']),name_row('새 입력 Chebyshev kNN',candidate)],[205,95,100,101])
p('50클라이언트의 동일 분할·최대128개씩 기존/신규 정상 참조·동일 보정 풀을 사용했다. SAE 가중치는 바꾸지 않았다. 입력 전용 구성은 학습 시드에 의존하지 않아 5개 시드 경로의 결과가 동일하며, 독립적인 다섯 데이터 실험이 아니다.')
h('무엇이 달라졌나')
p('기존 kNN은 여러 특징 차이를 유클리드 거리로 합친다. Chebyshev 거리는 가장 큰 절대 특징 차이를 쓴다. 정상 참조로 적합한 표준화 후 이 거리를 적용하고, 가까운 정상5개의 평균 거리를 이상 점수로 사용했다. 특징 하나가 크게 달라지는 패턴에 유리할 가능성은 있지만, 원인 규명이나 모든 환경의 우월성을 증명한 것은 아니다. [1]')
h('추천의 범위')
p('신규 탐지율92.49%, 오탐률6.60%는 현재 자료의 탐색 결과다. 앞선 누적 기억 후보의93.73% / 7.67%와는 적응 정책도 다르므로 직접 우열을 단정하지 않는다. 새 후보와 누적 정책을 결합한 성능은 이번에 검증하지 않았다.')
page('02 실제 비교: 실패한 방향도 포함')
labels={'input_whiten_knn':'공분산 whitening kNN','input_density_knn':'지역 밀도 정규화 kNN','input_isoforest':'Isolation Forest128','input_two_cov':'기존/신규 두 공분산 거리','input_l1':'입력 Manhattan kNN','input_cosine':'입력 cosine kNN','input_chebyshev':'입력 Chebyshev kNN','residual_l1':'입력+SAE Manhattan'}
rows=[]
for m,label in labels.items():rows.append(name_row(label,(a if m in a else b)[m]))
table(['추가 방식','신규 AUC','신규 FPR','신규 TPR'],rows,[205,95,100,101])
p('각 8구성 비교에는 원래 CEN·임계값 갱신·입력 kNN·입력+SAE 대조군도 넣었다. 개발 평균 FPR 각각10% 이하, 최악 시드15% 이하, 두 영역 TPR80% 이상에서 낮은 쪽 TPR이 높은 구성을 선택한 후 보류 평가를 열었다.')
p('첫 비교는 앞선 입력+SAE가 다시 선택됐다. 두 번째 거리 비교에서는 Chebyshev가 개발 기준으로 선택됐다. 나머지 새 방식이 모두 더 좋았다는 결론은 내리지 않는다. 두 공분산 방식은 오탐이 조금 낮지만 탐지율도 낮았다.')
h('기존 영역과 SAE의 필요성')
table(['새 Chebyshev 후보','AUC','FPR','TPR'],[[d,f"{metric(candidate,k,'mean_client_auc'):.5f}",percent(metric(candidate,k,'fpr')),percent(metric(candidate,k,'tpr'))] for d,k in [('기존','old'),('신규','new')]],[105,130,130,136])
p('SAE가 없어도 높은 성능이 나오므로, 향후 연구는 단순 입력 기준을 반드시 포함해야 한다. 압축·통신 이득 등 별도 가치가 확인되지 않으면 복잡한 모델을 유지해야 한다고 가정하지 않는다.','small')
page('03 낮은 오탐과 검증자 오류')
op=load(D/'output/operating_v1/final_summary.json');mp=load(D/'output/operating_metrics_v1/final_summary.json')
table(['방식 / 정상 보정 q','신규 FPR','신규 TPR'],[[('입력+SAE' if r['method']=='residual_knn_w025' else 'Chebyshev')+' / '+str(r['quantile']),percent(r['metrics']['new']['fpr']),percent(r['metrics']['new']['tpr'])] for r in [x for x in op if x['method']=='residual_knn_w025']+[x for x in mp if x['method']=='input_chebyshev'] if r['quantile'] in [.95,.99,1.0]],[245,128,128])
p('정상 점수의95·97.5·99·99.5·100백분위 임계값을 비교했다. 두 영역 개발 평균 FPR1% 이하·최악 시드2% 이하라는 조건을 통과한 구성은 두 비교 모두 없었다. 보류 정답으로 임계값을 맞추지 않았고 목표 달성을 주장하지 않는다.')
p('클라이언트·영역별 정상 보정 자료는26~39개다. 1/(n+1) 순위 간격은 약2.5~3.7%로 작은 꼬리 확률을 안정적으로 추정하기 어렵다. 실제 교환가능성을 확인한 것이 아니므로 이 간격을 오탐 보장으로 해석하지 않는다. 표본 최대값을 써도 보류 오탐이0이 되지 않았다.')
stress=load(D/'output/chebyshev_stress_v1/results.json');ss=[]
import statistics
for rate in [0,.01,.05,.1]:
 rr=[r for r in stress if r['requested_reference_contamination']==rate];ss.append([percent(statistics.mean(r['actual_contamination'] for r in rr)),percent(statistics.mean(r['metrics']['new']['fpr'] for r in rr)),percent(statistics.mean(r['metrics']['new']['tpr'] for r in rr))])
table(['공격의 실제 기억 혼입','개발 신규 FPR','개발 신규 TPR'],ss,[245,128,128])
p('공격 후보와 개발 평가 공격 ID는 분리했다. 약4.79% 혼입 시 TPR이 약92.15%에서17.75%로 하락했다. 검증자 오판 문제는 새 거리로도 해결되지 않았다. 신경망 업데이트 공격에 대한 검증은 아니다.','small')
page('04 검증 근거와 다음 연구의 기준')
table(['완료한 작업','검사 범위'],[['추가 탐지기 비교','2회×80개 개발/보류 기록, 클라이언트 AUC16,000개 독립 재계산'],['저오탐 운용점','2회×350개 기록, 혼동행렬1,400개와 단조성1,120개 검사'],['기본 규칙','새 단위 검사8개. 정상 적합·보정 시 scaler 불변·직렬화·점수 방향'],['단독 CPU 추론','115열664행, 보류127행의 저장 점수·판정 일치. 정답·도메인 없이 입력 후보 사용'],['원본 보존','원본/과거 보호 파일35개 및 CSV150개 해시 불변'],['오염 스트레스','20개 반복 기록. 주입 후보의 공격 라벨·역할과 평가 공격 분리 검사']], [110,391])
p('위 기록 수는 독립 데이터셋 수가 아니다. 같은 자료와 입력 기준을 재사용한 비교·시드 반복을 포함한다. 입력 전용의 반복 결과 동일성도 숨기지 않았다.')
h('한계가 연구 과제로 이어지는 지점')
p('① 새로운 기기·독립 자료에서 기존 설정을 고정해 확인해야 한다. 현재 A/B는 모의 특징 분할이다. ② 정상 승인 오류에 대한 격리·신뢰 앵커·버전 롤백이 필요하다. ③ 오탐1% 이하를 확인할 보정 표본량과 운영 기저율을 확보해야 한다. ④ SAE의 탐지 이득이 없다면 압축·통신 비용을 포함한 필요성을 따로 증명해야 한다.')
p('이번에 하지 않은 것: SAE 재훈련, 논문 손실 교정 학습, 실제 기기/시간 드리프트 검증, DP/안전 집계, 누적 기억과 새 거리의 결합. 진행하지 않은 일을 완료로 기재하지 않는다.')
h('재현 경로와 사용 안내')
p('continual/followup_heads의 run.py, run_metrics.py, verify*.py, operating*.py, stress_metric.py, infer.py에 분리했다. 실행 버전과 선택 동결 파일은 output 아래 각 실행 폴더에 있다. 기존 파일과 모델은 덮어쓰지 않는다. 단위 검사는 torch를 먼저 로드하는 실행 명령을 사용했다. 단독 sklearn ensemble 로딩에서 Windows DLL 오류가 한 차례 발생한 사실도 README에 기록했다.','small')
for title,url in [('NearestNeighbors / distance metric','https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html'),('LedoitWolf covariance','https://scikit-learn.org/stable/modules/generated/sklearn.covariance.LedoitWolf.html'),('IsolationForest','https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html')]:p('<link href="'+url+'" color="#23638a">'+title+'</link>','small')
p('도구 문서는 구현 정의의 근거다. 이번 IoT 성능 수치는 저장된 자체 실험과 독립 검산을 근거로 한다.','small')
def footer(canvas,doc):
 canvas.setFont('Body',8);canvas.setFillColor(MUTED);canvas.drawString(M,H-27,'FedMSE 후속 거리·밀도·저오탐 비교 / 20222038 이진호');canvas.drawString(M,26,'같은 보류 풀의 탐색 결과 / 새 외부 확인 전');canvas.drawRightString(W-M,26,f'{doc.page} / 4')
path=O/'FedMSE_Further_Detector_Experiments.pdf'
SimpleDocTemplate(str(path),pagesize=A4,leftMargin=M,rightMargin=M,topMargin=48,bottomMargin=43).build(story,onFirstPage=footer,onLaterPages=footer)
(O/'FedMSE_Further_Detector_Experiments.md').write_text('\n\n'.join(md),encoding='utf-8')
from pypdf import PdfReader
assert len(PdfReader(path).pages)==4
qa=D/'report_qa';qa.mkdir(exist_ok=True);subprocess.run(['pdftoppm','-scale-to','1150','-png',str(path),str(qa/'page')],check=True)
print(path)
