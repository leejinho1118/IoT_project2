from pathlib import Path
import subprocess,json,datetime,hashlib,zipfile,time
root=Path(r'C:\Users\dlwls\LocalProjects\fedmse');out=root/'external_data/n_baiot_raw'
url='https://archive.ics.uci.edu/static/public/442/detection%2Bof%2Biot%2Bbotnet%2Battacks%2Bn%2Bbaiot.zip'
status=out/'download_status.json';part=out/'official_uci_442.zip.partial';dest=out/'official_uci_442.zip'
if dest.exists():raise SystemExit('Existing official archive preserved')
if part.exists():raise SystemExit('Existing partial download preserved; inspect before reuse')
state=dict(state='downloading',started=datetime.datetime.now().astimezone().isoformat(),source=url,source_page='https://archive.ics.uci.edu/dataset/442/detection+of+iot+botnet+attacks+n+baiot',license='CC BY 4.0; cite Meidan et al. UCI DOI 10.24432/C5RC8J',expected_size='Official page reports approximately 1.7 GB',destination=str(dest))
status.write_text(json.dumps(state,indent=2),encoding='utf-8')
with (out/'download.log').open('w',encoding='utf-8') as log:
 run=subprocess.run(['curl.exe','-L','--fail','--retry','2','--retry-delay','5','--connect-timeout','20','--max-time','1800','--max-filesize','2200000000','--output',str(part),url],stdout=log,stderr=log)
state.update(exit_code=run.returncode,finished=datetime.datetime.now().astimezone().isoformat(),bytes=part.stat().st_size if part.exists() else 0)
if run.returncode==0 and zipfile.is_zipfile(part):
 part.rename(dest)
 with zipfile.ZipFile(dest) as z:
  listing=[dict(name=i.filename,bytes=i.file_size,compressed_bytes=i.compress_size) for i in z.infolist()]
 (out/'archive_members.json').write_text(json.dumps(listing,indent=2),encoding='utf-8')
 state.update(state='complete',sha256=hashlib.file_digest(dest.open('rb'),'sha256').hexdigest(),members=len(listing))
else:state.update(state='failed',note='Partial file and log retained; no automatic restart beyond bounded curl network retries')
status.write_text(json.dumps(state,indent=2),encoding='utf-8');print(json.dumps(state,indent=2))
raise SystemExit(0 if state['state']=='complete' else 1)
