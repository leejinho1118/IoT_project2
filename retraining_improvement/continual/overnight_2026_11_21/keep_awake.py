"""Temporary system-awake request for the authorized overnight job; display may sleep."""
from pathlib import Path
import ctypes,datetime,time,os,json
h=Path(__file__).resolve().parent;end=datetime.datetime(2026,11,20,22,40,tzinfo=datetime.timezone.utc)
if not ctypes.windll.kernel32.SetThreadExecutionState(0x80000001):raise RuntimeError('System-awake request failed')
try:
 while datetime.datetime.now(datetime.timezone.utc)<end:
  p=h/'status.json';s=json.loads(p.read_text(encoding='utf-8-sig'))
  if s.get('stop_requested') or s.get('state') in ['complete','stopped','paused']:break
  (h/'awake_status.json').write_text(json.dumps(dict(state='active',pid=os.getpid(),until_utc=end.isoformat(),permanent_power_settings_changed=False,display_kept_on=False),indent=2),encoding='utf-8')
  time.sleep(30)
finally:
 ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
 (h/'awake_status.json').write_text(json.dumps(dict(state='released',pid=os.getpid()),indent=2),encoding='utf-8')
