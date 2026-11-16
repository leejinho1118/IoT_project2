"""A~E 각각 실제 분석 시점에 생성하는 한국어 PDF/Markdown 연구 보고서."""
from pathlib import Path
import json,argparse,html,datetime,subprocess
from reportlab.platypus import SimpleDocTemplate,Paragraph,Table,TableStyle,Spacer,PageBreak
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
P=argparse.ArgumentParser();P.add_argument('--out',required=True);P.add_argument('--round',required=True,choices=list('ABCDE'));a=P.parse_args()
O=Path(a.out);letter=a.round
load=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
config=load(O/'config.json');analysis=load(O/letter/'analysis.json');policy=analysis['policy'];validation=load(O/letter/'validation.json');all_analyses=load(O/'analyses.json');next_policy=load(O/letter/'next_policy.json')
pdfmetrics.registerFont(TTFont('Body','C:/Windows/Fonts/malgun.ttf'));pdfmetrics.registerFont(TTFont('Bold','C:/Windows/Fonts/malgunbd.ttf'));pdfmetrics.registerFontFamily('Body',normal='Body',bold='Bold',italic='Body',boldItalic='Bold')
W,H=A4;M=47;WIDTH=W-2*M;NAVY=colors.HexColor('#203c57');MUTED=colors.HexColor('#647b91');LIGHT=colors.HexColor('#edf4f8');GRID=colors.HexColor('#d5dfe7')
styles={name:ParagraphStyle(name,fontName=font,fontSize=size,leading=leading,spaceAfter=8,wordWrap='CJK',textColor=color) for name,font,size,leading,color in [('title','Bold',18,25,NAVY),('h','Bold',11.5,17,NAVY),('body','Body',10,15.7,colors.HexColor('#172430')),('small','Body',8.3,12.7,MUTED),('cell','Body',8.8,13,colors.HexColor('#172430')),('head','Bold',8.8,13,NAVY)]}
styles['h'].keepWithNext=True;story=[];md=[];page_count=0

def p(text,style='body'):
 story.append(Paragraph(text,styles[style]));md.append(text.replace('<br/>','\n'))
def h(text):p(text,'h')
def page(text):
 global page_count
 if page_count:story.append(PageBreak())
 page_count+=1;p(text,'title')
def table(headers,rows,widths):
 data=[[Paragraph(html.escape(str(x)),styles['head' if i==0 else 'cell']) for x in row] for i,row in enumerate([headers]+rows)]
 t=Table(data,colWidths=[WIDTH*x/sum(widths) for x in widths],repeatRows=1)
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),LIGHT),('LINEABOVE',(0,0),(-1,0),.8,NAVY),('LINEBELOW',(0,0),(-1,-1),.4,GRID),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]));story.extend([t,Spacer(1,10)])
 md.extend(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])
def fmt(m):return f"{m['mean']:.5f}"+(f" ± {m['std']:.5f}" if m['std'] is not None else ' (1시드)')
parent=next((x for x in all_analyses if x['round']==policy.get('parent')),None)
page(f'{letter}안 연구 보고서: '+{'A':'고정 리플레이 기준','B':'보관 자료 재사용 강화','C':'다양성 기반 보관','D':'상황별 비율 조정','E':'검증 후 채택과 되돌리기'}[letter])
p('<b>20222038 이진호</b> / '+datetime.datetime.now().astimezone().isoformat(timespec='seconds'))
p('이번 안의 목적: '+policy['direction']+'. 모든 안은 같은 초기 전역 모델과 개발 분할에서 다시 출발하며, 앞선 안의 학습 가중치를 이어받지 않는다. 앞선 개발 분석으로 구성만 선택한다.')
table(['항목','실제 수행'],[['실행 규모',f"{config['clients']}클라이언트 · {len(config['seeds'])}시드 · {config['stages']}유입 단계"],['학습 상한',f"로컬 {config['local_epochs']}에포크 · 전역 {config['update_rounds']}라운드 · batch12 · lr1e-5"],['초기 모델','완료된 V2 초기 전역 모델 재사용. 원본 로컬 model.cpt가 아님'],['개발 검증',f"{validation['records']}개 학습·개발 평가 기록 검증 통과"],['상태','개발 분석 완료. 최종 평가는 E 이후 공통 비교에서만 개방' if letter!='E' else '전체 안 구성 선택 고정 후 공통 최종 평가 개방']], [115,386])
h('마지막 유입 단계의 개발 평가 / 평균 ± 표본 표준편차')
table(['영역','클라이언트 평균 AUC','합산 FPR','합산 TPR'],[[domain]+[fmt(analysis['metrics'][key][metric]) for metric in ['mean_client_auc','fpr','tpr']] for domain,key in [('기존 정상 A','old'),('신규 정상 B','new')]],[100,145,128,128])
p('이 페이지 수치는 방향 선택에 사용한 개발 자료 결과다. 최종 시험 결과 또는 독립 외부 일반화 성능으로 해석하지 않는다. AUC는 클라이언트 단순 평균, FPR·TPR은 혼동행렬 합산으로 계산했다.','small')
page('02 이번 안에서 바꾼 것과 유지한 것')
table(['구성','이번 안'],[['부모 구성',str(policy.get('parent') or '없음: 기준 안')],['보관 표본','초기 정상에서 무작위 순서의 앞부분' if policy['memory']=='uniform' else '초기 정상의 표준화 특징에서 farthest-first로 다양성 확보'],['보관 용량',f"클라이언트당 {config['capacity']}개 정상"],['리플레이 비율',f"시작 {policy['ratio']*100:.1f}% / "+('이전 단계 개발 복원 오차로 조정' if policy['adaptive'] else '고정')],['갱신 채택','개발 점수 개선과 기존 AUC 한계 통과 시만 채택' if policy['gate'] else '학습된 새 모델을 항상 채택']], [115,386])
p('공통 조건은 SAE 115-50-11-50-115, lambda10, 원본 MSEAvg, 공유 모델의 순차 학습, 원본 두 if 전역 조기 종료다. 초기 A 입력 StandardScaler와 고정 [-20,20] 포화도 모든 안에 동일하게 적용했다. 최대 예산은 같지만 조기 종료 때문에 실제 스텝 수는 다를 수 있다.')
p('초기 A/B 정상 검증 풀을 로컬 조기 종료 40%, 임계값 30%, 방향 선택 30%로 분리했다. 공격 개발 평가는 후보 공격 일부를 별도 보류했으며 정상 학습에 넣지 않았다. 최종 보류 자료는 E 선택 고정 전 특징 추출·평가에서 제외했다.')
h('구현상 주의할 해석')
p('CEN은 해당 단계의 정상 학습 기준 자료로 다시 적합한다. 따라서 탐지 성능 변화에는 신경망 가중치와 CEN 기준의 변화가 함께 포함된다. 과거 정상 복원 MSE도 기록했지만, 순수 표현 망각만 인과적으로 분리한 실험은 아니다.')
p('재사용한 초기 모델의 원래 조기 종료에는 기존 검증 풀 전체가 사용됐다. 현재 방향 선택 자료는 개발 자료이며 독립 시험이 아니다. 최종 보류 풀도 이전 V2 연구에서 이미 결과를 본 자료이므로 이번 후속은 탐색적 비교이며 새 외부 검증이라고 주장하지 않는다.','small')
page('03 결과 분석과 다음 방향')
h('확인된 개발 결과')
p(f"기존 AUC의 초기 대비 평균 변화는 {analysis['old_auc_change_mean']:+.5f}, 기존 정상 복원 MSE 변화는 {analysis['old_mse_change_mean']:+.5f}다. AUC 변화가 음수면 저하, MSE 변화가 양수면 기존 정상 복원이 나빠진 것이다.")
if parent:
 delta=analysis['development_score_mean']-parent['development_score_mean']
 old=analysis['metrics']['old']['mean_client_auc']['mean']-parent['metrics']['old']['mean_client_auc']['mean'];new=analysis['metrics']['new']['mean_client_auc']['mean']-parent['metrics']['new']['mean_client_auc']['mean']
 p(f"부모 {parent['round']}안 대비 균형 개발 점수 변화는 {delta:+.5f}, 기존 AUC는 {old:+.5f}, 신규 AUC는 {new:+.5f}다. "+('개발 점수는 높아졌다.' if delta>0 else '개발 점수 우위는 관찰되지 않았다.'))
else:p('A안은 같은 초기 모델·개발 분할에서 후속 구성을 비교하기 위한 기준이다. 기존 105개 V2 평가에서 관찰한 적응·망각 절충을 출발 문제로 삼았으나 수치를 직접 빼서 개선 폭으로 계산하지 않는다.')
p(f"시드당 누적 추가 학습 시간 평균은 {analysis['training_seconds_mean']:.1f}초다. 전체 {analysis['total_updates']}개 갱신 후보 중 {analysis['accepted_updates']}개를 배포 버전으로 채택했다. 거부한 후보의 학습 비용도 포함한다.")
h('점수에 따른 실제 후속 결정')
p('다음 방향: '+str(next_policy.get('direction','E 이후 설정을 고정하고 최종 비교한다.')))
p('개발 점수 = 기존·신규 AUC의 평균 - 0.1 × max(0, 두 영역 FPR의 최댓값 - 0.1). 동률이면 추가 학습 시간이 짧은 안을 우선한다. 이 선택 규칙을 최종 평가 수치에 맞춰 바꾸지 않았다.')
if policy['adaptive']:p('적응형 비율은 이전 단계 기존 정상 MSE가 초기의 1.25배를 넘으면 +25%p(최대75%), 신규 MSE가 초기의 90% 이상이거나 기존 MSE가 1.10배 이하이면 -12.5%p(최소12.5%), 나머지는 유지한다. 다음 단계 자료는 이 결정을 할 때 보지 않는다.')
if policy['gate']:p('채택 게이트는 후보의 기존 개발 AUC가 초기보다 3%p 이상 떨어지지 않고, 균형 개발 점수가 직전 채택 모델보다 0.0001 이상 높을 때만 통과한다. 실패하면 가중치와 CEN 기준을 이전 버전으로 되돌린다. 모두 거부되는 결과도 유효한 결과다.')
page('04 누적 비교와 재현 근거')
table(['안','기존 개발 AUC','신규 개발 AUC','균형 개발 점수'],[[x['round'],f"{x['metrics']['old']['mean_client_auc']['mean']:.5f}",f"{x['metrics']['new']['mean_client_auc']['mean']:.5f}",f"{x['development_score_mean']:.5f}"] for x in all_analyses],[45,155,155,146])
p('위 표는 이 보고서 작성 시점까지 완료한 안만 포함한다. 불리한 시드·단계를 제외하지 않았고 마지막 단계 전체 시드를 요약했다. 여러 안을 같은 개발 자료로 고른 탐색 과정이므로 개발 점수 차이에 통계적 유의성을 주장하지 않는다.')
table(['근거','위치'],[['안별 분석',letter+'/analysis.json'],['정책과 학습 입력',letter+'/policy.json 및 seed_N/stage_K/training_input_ids.json'],['학습 로그','seed_N/stage_K/training/training.json'],['버전·예측','seed_N/stage_K/published/detector_bundle.pt 및 predictions.csv.gz'],['검증',letter+'/validation.json'],['개발·최종 분리','partition_manifest.json.gz 및 data_audit.json']], [115,386])
p('모든 작업은 C:/Users/dlwls/LocalProjects/fedmse/continual/research_ABCDE 아래에 저장한다. OneDrive에 원본 자료나 모델을 복사하지 않았다. 기존 원본·V1·V2 결과는 변경하지 않는다. 메모리 비용은 특징 페이로드이며 프로세스 RSS·GPU 피크 메모리와 다르다.','small')
if letter=='E' and (O/'final_summary.json').exists():
 page('05 고정 선택 이후 공통 최종 비교')
 final=load(O/'final_summary.json');selection=load(O/'selection_frozen.json')
 p('최종 보류 평가를 열기 전에 개발 자료로 선택한 안은 <b>'+selection['selected_round']+'안</b>이다. 아래 결과를 보고 선택을 다시 바꾸지 않았다.')
 table(['안','기존 AUC','신규 AUC','기존 FPR','신규 FPR'],[[x['round']]+[f"{x[d][k]['mean']:.5f}" for d,k in [('old','mean_client_auc'),('new','mean_client_auc'),('old','fpr'),('new','fpr')]] for x in final['summary']],[40,115,115,115,116])
 p('고정된 마지막 단계의 전체 시드 평균이다. 임계값·CEN·입력 scaler는 저장 버전 그대로 사용하고 최종 자료로 다시 적합하지 않았다. 이 자료는 앞선 V2에서 이미 본 같은 데이터셋의 보류 풀이다. 이번 캠페인 안에서는 잠갔지만 완전히 새 외부 확인 실험이라고 주장하지 않는다.')
 h('마지막으로 구분할 것')
 p('개발 점수로 고른 안과 최종 결과에서 사후 최고인 안은 다를 수 있다. 둘을 바꿔 쓰지 않는다. 모의 특징 변화이지 실제 시간 드리프트가 아니며, 데이터셋 라벨을 정상 검증자로 사용하는 보조 환경이다. 효과가 없는 안과 거부된 갱신도 근거에 보존했다.')

def footer(canvas,doc):
 canvas.setFont('Body',8);canvas.setFillColor(MUTED);canvas.drawString(M,H-27,f'FedMSE {letter}안 연구 보고서 / 20222038 이진호');canvas.drawString(M,26,'개발 분석과 최종 비교를 구분 · 모의 특징 분포 변화');canvas.drawRightString(W-M,26,str(doc.page))
reports=O/'reports';reports.mkdir(exist_ok=True);path=reports/f'{letter}_Plan_Report.pdf'
SimpleDocTemplate(str(path),pagesize=A4,leftMargin=M,rightMargin=M,topMargin=48,bottomMargin=43).build(story,onFirstPage=footer,onLaterPages=footer)
(reports/f'{letter}_Plan_Report.md').write_text('\n\n'.join(md),encoding='utf-8')
from pypdf import PdfReader
reader=PdfReader(path);assert len(reader.pages)==page_count
assert all('20222038 이진호' in page.extract_text() for page in reader.pages)
qa=O/'report_qa'/letter;qa.mkdir(parents=True,exist_ok=True)
subprocess.run(['pdftoppm','-scale-to','1100','-png',str(path),str(qa/'page')],check=True)
import pdfplumber
with pdfplumber.open(path) as pdf:
 for i,page in enumerate(pdf.pages):
  assert all(-1<=c['x0']<=W+1 and c['x1']<=W+1 for c in page.chars),('horizontal overflow',i)
(reports/f'{letter}_render_validation.json').write_text(json.dumps(dict(pages=len(reader.pages),rendered=True,name_on_every_page=True,no_horizontal_overflow=True,visual_review='pending human/model page inspection'),indent=2),encoding='utf-8')
print(path)

from PIL import Image,ImageDraw
thumbs=[]
for page_path in sorted(qa.glob('page-*.png')):
 im=Image.open(page_path).convert('RGB');im.thumbnail((520,736));canvas=Image.new('RGB',(540,770),'#dfe5eb');canvas.paste(im,((540-im.width)//2,25));ImageDraw.Draw(canvas).text((10,8),letter+' '+page_path.stem,fill='black');thumbs.append(canvas)
sheet=Image.new('RGB',(1080,770*((len(thumbs)+1)//2)),'white')
for i,im in enumerate(thumbs):sheet.paste(im,((i%2)*540,(i//2)*770))
sheet.save(qa/'contact.png')
