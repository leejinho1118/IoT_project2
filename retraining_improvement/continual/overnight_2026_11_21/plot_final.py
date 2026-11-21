"""Paper-style final curves: arrival stages, not fabricated training rounds."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
H=Path(__file__).resolve().parent;OUT=H/'analysis_v1';A=json.loads((OUT/'analysis.json').read_text());FONT=FontProperties(fname='C:/Windows/Fonts/malgun.ttf');plt.rcParams.update({'font.family':FONT.get_name(),'font.size':10,'axes.unicode_minus':False,'svg.fonttype':'none'})
selected=A['selection']['selected'][1];rows=[r for r in A['summary'] if r['kind']=='main30' and r['mode']==selected['mode'] and r['loss']==selected['loss']]
styles={'no_update':('고정 FedMSE','#222222','--','s'),'calibration_only':('임계값만 갱신','#909090',':','v'),'new_only':('신규 정상 재학습','#e6a000','-','^'),'replay':('리플레이 재학습','#377eb8','-','D'),'safe_replay':('검증 후 리플레이','#4daf4a','-','P'),'trigger_safe':('조건부 검증 재학습','#984ea3','-.','x'),'safe_new':('검증 후 신규 재학습','#d7191c','-','o')}
datasets=[('raw_blocks_v2','N-BaIoT: 9개 실제 장치 / 미라이 공격 평가'),('rt_thing_to_mqtt_v1','RT-IoT2022: Thing → MQTT (+미관측 Wipro 정상)'),('rt_mqtt_to_thing_v1','RT-IoT2022: MQTT → Thing (+미관측 Wipro 정상)')]
fig,axes=plt.subplots(3,2,figsize=(11,10),layout='constrained')
for i,(data,title) in enumerate(datasets):
 for j,(metric,label) in enumerate([('new_mean_client_auc','신규 영역 ROC-AUC'),('new_fpr','신규 정상 오탐률')]):
  ax=axes[i,j]
  for method,(name,color,ls,marker) in styles.items():
   rr=sorted([r for r in rows if r['data']==data and r['method']==method],key=lambda r:r['stage'])
   if not rr:continue
   x=[r['stage'] for r in rr];y=[r[metric+'_mean'] for r in rr];sd=np.array([r[metric+'_sd'] or 0 for r in rr]);ax.plot(x,y,label=name,color=color,linestyle=ls,marker=marker,linewidth=2.5 if method=='safe_new' else 1.5,markersize=5);ax.fill_between(x,np.clip(np.array(y)-sd,0,1),np.clip(np.array(y)+sd,0,1),color=color,alpha=.09)
  ax.set_title(title+'\n'+label,fontsize=10);ax.set_xticks([1,2,3]);ax.set_xlabel('유입 단계');ax.set_ylabel('ROC-AUC' if j==0 else 'FPR');ax.grid(alpha=.25);ax.set_ylim(bottom=0 if j else None,top=1.01);ax.spines[['top','right']].set_visible(False)
handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=4,fontsize=9);fig.suptitle('고정된 최종 평가 자료 비교 · 선=시드 평균, 음영=표준편차',fontsize=13)
for ext in ['png','svg','pdf']:fig.savefig(OUT/f'FedMSE_Overnight_Curves.{ext}',dpi=220,bbox_inches='tight')
plt.close(fig)
# Disentangle changing the preprocessing from adding continual adaptation.
fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
for ax,(data,title) in zip(axes,datasets):
 for mode,method,label,color,ls in [('standard','no_update','표준화 조건 · 고정','#222','--'),('standard','safe_new','표준화 조건 · 검증 재학습','#377eb8','-'),(selected['mode'],'safe_new','로그 전처리 · 검증 재학습','#d7191c','-')]:
  rr=sorted([r for r in A['summary'] if r['kind']=='main30' and r['data']==data and r['mode']==mode and r['method']==method],key=lambda r:r['stage']);ax.plot([r['stage'] for r in rr],[r['new_mean_client_auc_mean'] for r in rr],label=label,color=color,linestyle=ls,marker='o');ax.grid(alpha=.25);ax.set_xticks([1,2,3]);ax.set_xlabel('유입 단계');ax.set_title(title.split(':')[0]);ax.set_ylabel('신규 영역 ROC-AUC');ax.set_ylim(top=1.01)
fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=3,fontsize=9)
for ext in ['png','svg']:fig.savefig(OUT/f'Preprocessing_Ablation.{ext}',dpi=220,bbox_inches='tight')
plt.close(fig);print('Saved final scientific figures')
