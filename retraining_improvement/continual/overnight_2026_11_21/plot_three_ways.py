"""Render the three requested retraining arms from fixed final metrics."""
from pathlib import Path
import json,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
H=Path(__file__).resolve().parent;A=json.loads((H/'analysis_v1/analysis.json').read_text());font_manager.fontManager.addfont('C:/Windows/Fonts/malgun.ttf');plt.rcParams.update({'font.family':'Malgun Gothic','font.size':10,'axes.unicode_minus':False,'svg.fonttype':'none'})
fig,axs=plt.subplots(3,3,figsize=(14,9),layout='constrained');datasets=[('raw_blocks_v2','N-BaIoT 원자료'),('rt_thing_to_mqtt_v1','RT: Thing → MQTT'),('rt_mqtt_to_thing_v1','RT: MQTT → Thing')];methods=[('no_update','고정 FedMSE','#222222','s','--'),('new_only','신규 정상 재학습','#377eb8','^','-'),('replay','리플레이 재학습','#d7191c','o','-')]
for i,(data,title) in enumerate(datasets):
 for j,(metric,label) in enumerate([('new_mean_client_auc','신규 ROC-AUC'),('new_fpr','신규 오탐률 FPR'),('new_tpr','신규 공격 탐지율 TPR')]):
  ax=axs[i,j]
  for method,name,color,marker,ls in methods:
   rows=sorted([r for r in A['summary'] if r['kind']=='main30' and r['mode']=='signed_log' and r['data']==data and r['method']==method],key=lambda r:r['stage']);x=[r['stage'] for r in rows];y=np.array([r[metric+'_mean'] for r in rows]);s=np.array([r[metric+'_sd'] or 0 for r in rows]);ax.plot(x,y,label=name,color=color,marker=marker,linestyle=ls,linewidth=2,markersize=5);ax.fill_between(x,np.clip(y-s,0,1),np.clip(y+s,0,1),color=color,alpha=.12)
  ax.set_title(title+' · '+label,fontsize=10);ax.set_xticks([1,2,3]);ax.set_xlabel('유입 단계');ax.grid(alpha=.25);ax.set_ylim(0,1.02);ax.spines[['top','right']].set_visible(False)
fig.legend(*axs[0,0].get_legend_handles_labels(),loc='outside lower center',ncol=3,fontsize=11);fig.suptitle('재학습 없음 / 신규 정상만 / 신규+기존 정상 리플레이 · 5개 시드 평균 ± 표준편차\n동일 마지막 보류 자료 평가 · 로그 전처리 조건 · 원 논문 표의 수치를 직접 비교한 그림이 아님',fontsize=12)
for ext in ['png','svg']:fig.savefig(H/f'analysis_v1/FedMSE_Retraining_Three_Ways.{ext}',dpi=220,bbox_inches='tight')
plt.close(fig)
