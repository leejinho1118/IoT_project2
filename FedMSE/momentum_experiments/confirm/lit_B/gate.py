import json,glob,os,re,subprocess,hashlib,unicodedata,sys
from pypdf import PdfReader
os.chdir(os.path.dirname(os.path.abspath(__file__)))
LIG={'ﬁ':'fi','ﬂ':'fl','ﬀ':'ff','ﬃ':'ffi','ﬄ':'ffl','ﬅ':'st','ﬆ':'st'}
def norm(s):
    s=unicodedata.normalize('NFKC',s)
    for k,v in LIG.items(): s=s.replace(k,v)
    s=re.sub(r'[‘’‚′`´]',"'",s)
    s=re.sub(r'[“”„″]','"',s)
    s=re.sub(r'[‐-―−­]','-',s)
    s=re.sub(r'(\w)-\s*\n\s*(\w)',r'\1\2',s)
    s=s.lower()
    s=re.sub(r'\s+','',s)   # strip all whitespace on both sides
    s=s.replace('-','')     # hyphen-insensitive (line-break hyphens)
    return s
rep={}
for jf in sorted(glob.glob('collect/*.json')):
    i=os.path.basename(jf)[:-5]
    if i=='extra_candidates': continue
    d=json.load(open(jf)); pdf=f'pdf/{i}.pdf'
    for _ in range(2):
        if os.path.exists(pdf) and os.path.getsize(pdf)>1000: break
        subprocess.run(['curl','-sL','-o',pdf,f'https://arxiv.org/pdf/{i}'])
    sha=hashlib.sha256(open(pdf,'rb').read()).hexdigest()
    txt=norm('\n'.join((p.extract_text() or '') for p in PdfReader(pdf).pages))
    tm=norm(d.get('title',''))in txt[:6000] or norm(d.get('title','')) in txt
    tot=ok=0;failed=[]
    for q in d['quotes']:
        t=q['text']
        if t.strip()=='not found in the paper text': continue
        tot+=1
        if norm(t) in txt: ok+=1
        else: failed.append(t)
    rep[i]=dict(title=d.get('title'),title_match=tm,quotes_total=tot,passed=ok,failed=tot-ok,sha256=sha,failed_quotes=failed)
json.dump(rep,open('gate_report.json','w'),ensure_ascii=False,indent=1)
with open('gate_report.md','w') as f:
    f.write('| id | title match | total | passed | failed | sha256 |\n|---|---|---|---|---|---|\n')
    for i,r in rep.items(): f.write(f"| {i} | {r['title_match']} | {r['quotes_total']} | {r['passed']} | {r['failed']} | {r['sha256'][:16]} |\n")
    for i,r in rep.items():
        for t in r['failed_quotes']: f.write(f"\nFAILED {i}: {t}\n")
print(open('gate_report.md').read())
