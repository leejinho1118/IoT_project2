"""Safely unpack the public UCI archive and inventory original device/type CSVs."""
from pathlib import Path,PurePosixPath
import zipfile,subprocess,json,hashlib,datetime,csv,os
ROOT=Path(__file__).resolve().parents[2];HOME=ROOT/'external_data/n_baiot_raw'
archive=HOME/'official_uci_442.zip';out=HOME/'unpacked_v1'
if not archive.exists():raise SystemExit('Official download not complete')
if out.exists():raise SystemExit('Existing extraction preserved; inspect its status')
out.mkdir();status=HOME/'extraction_status.json'
def state(**kw):status.write_text(json.dumps(kw,indent=2),encoding='utf-8')
def safe(name):
    p=PurePosixPath(name.replace('\\','/'))
    if p.is_absolute() or '..' in p.parts or any(':' in x for x in p.parts):raise ValueError('Unsafe archive member: '+name)
    return p
state(state='extracting_zip',started=datetime.datetime.now().astimezone().isoformat(),source=str(archive))
with zipfile.ZipFile(archive) as z:
    for i in z.infolist():
        safe(i.filename)
        if ((i.external_attr>>16)&0o170000)==0o120000:raise ValueError('Archive symlink rejected')
    z.extractall(out)
archives=[]
for rar in sorted(out.rglob('*.rar')):
    listing=subprocess.run(['tar.exe','-tf',str(rar)],capture_output=True,text=True,encoding='utf-8',errors='replace')
    if listing.returncode:raise RuntimeError('Native RAR listing failed: '+listing.stderr)
    members=[x.strip() for x in listing.stdout.splitlines() if x.strip()]
    for name in members:safe(name)
    target=rar.parent/(rar.stem+'_unpacked');target.mkdir()
    state(state='extracting_rar',current=str(rar.relative_to(out)),completed_rar=len(archives))
    run=subprocess.run(['tar.exe','-xf',str(rar),'-C',str(target)],capture_output=True,text=True,encoding='utf-8',errors='replace')
    if run.returncode:raise RuntimeError('Native RAR extraction failed: '+run.stderr)
    archives.append(dict(archive=str(rar.relative_to(out)),member_count=len(members),target=str(target.relative_to(out))))
inventory=[]
for p in sorted(out.rglob('*.csv')):
    with p.open('r',encoding='utf-8-sig',newline='') as f:
        header=next(csv.reader(f))
    if len(header)!=115:continue
    parts=p.relative_to(out).parts
    benign='benign' in p.name.lower()
    family='benign' if benign else 'mirai' if any('mirai' in x.lower() for x in parts) else 'gafgyt' if any('gafgyt' in x.lower() or 'bashlite' in x.lower() for x in parts) else 'unknown'
    inventory.append(dict(path=str(p.relative_to(out)),device=parts[0],family=family,attack_type='benign' if benign else p.stem,bytes=p.stat().st_size,header=header))
(HOME/'csv_inventory.json').write_text(json.dumps(inventory,indent=2),encoding='utf-8')
state(state='complete',finished=datetime.datetime.now().astimezone().isoformat(),csv_files=len(inventory),devices=sorted({x['device'] for x in inventory}),rar_archives=archives,source_archive_sha256=json.loads((HOME/'download_status.json').read_text(encoding='utf-8'))['sha256'])
print(json.dumps(dict(files=len(inventory),devices=sorted({x['device'] for x in inventory}),families=sorted({x['family'] for x in inventory})),indent=2))
