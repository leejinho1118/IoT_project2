"""Read-only Windows process counters; separate from logical replay payload."""
from pathlib import Path
import ctypes as ct,json,time,datetime,os
from ctypes import wintypes as w
H=Path(__file__).resolve().parent;OUT=H/'memory_observer_status.json';LOG=H/'memory_observations.jsonl'
if OUT.exists():raise RuntimeError('Existing observer preserved')
class Counters(ct.Structure):
 _fields_=[('cb',w.DWORD),('PageFaultCount',w.DWORD)]+[(n,ct.c_size_t) for n in ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage','PrivateUsage']]
k=ct.WinDLL('kernel32',use_last_error=True);k.OpenProcess.argtypes=[w.DWORD,w.BOOL,w.DWORD];k.OpenProcess.restype=w.HANDLE;k.CloseHandle.argtypes=[w.HANDLE];k.K32GetProcessMemoryInfo.argtypes=[w.HANDLE,ct.POINTER(Counters),w.DWORD];k.K32GetProcessMemoryInfo.restype=w.BOOL
state=dict(state='running',observer_pid=os.getpid(),started=datetime.datetime.now().astimezone().isoformat(),processes={},scope='Training process Windows working set/private usage; OS peak working set since process start when observed. No GPU allocation or whole-machine RSS attribution. Earlier completed runs before observer start have no retrospective process measurement.')
def save():OUT.write_text(json.dumps(state,indent=2),encoding='utf-8')
save()
with LOG.open('x',encoding='utf-8') as log:
 while datetime.datetime.now(datetime.timezone.utc)<datetime.datetime(2026,11,20,22,25,tzinfo=datetime.timezone.utc):
  active=[]
  for status in (H/'runs').glob('*/status.json'):
   try:
    s=json.loads(status.read_text());pid=s.get('pid')
    if s['state'] in ['complete','failed'] or not pid:continue
    handle=k.OpenProcess(0x0400|0x0010,False,pid)
    if not handle:continue
    c=Counters();c.cb=ct.sizeof(c);ok=k.K32GetProcessMemoryInfo(handle,ct.byref(c),c.cb);k.CloseHandle(handle)
    if not ok:continue
    name=status.parent.name;active.append(name);row=dict(time=datetime.datetime.now().astimezone().isoformat(),run=name,pid=pid,phase=s['state'],method=s.get('method'),stage=s.get('stage'),working_set_bytes=int(c.WorkingSetSize),peak_working_set_bytes=int(c.PeakWorkingSetSize),private_bytes=int(c.PrivateUsage));log.write(json.dumps(row)+'\n');log.flush();entry=state['processes'].setdefault(name,dict(pid=pid,first_observed=row['time'],observations=0,peak_working_set_bytes=0,max_sampled_private_bytes=0));entry['observations']+=1;entry['last_observed']=row['time'];entry['peak_working_set_bytes']=max(entry['peak_working_set_bytes'],row['peak_working_set_bytes']);entry['max_sampled_private_bytes']=max(entry['max_sampled_private_bytes'],row['private_bytes'])
   except (OSError,ValueError):continue
  state['active_runs']=active;save()
  done=[]
  for n in ['replication_status_v1.json','budget_status_v1.json','approval_status_v1.json']:
   try:done.append(json.loads((H/n).read_text())['state'] in ['complete','time_budget_stop','failed','blocked_prior_failure'])
   except (OSError,ValueError):done.append(False)
  if all(done) and not active:state['state']='complete';save();break
  time.sleep(5)
 else:state['state']='deadline_stop';save()
