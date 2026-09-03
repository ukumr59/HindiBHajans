from __future__ import annotations
import json, os, subprocess, sys, urllib.error, urllib.request
from pathlib import Path

ROOT=Path('/kaggle/working'); REPO=ROOT/'HindiBHajans'; ROOT.mkdir(parents=True,exist_ok=True)
if not (REPO/'.git').exists(): subprocess.run(['git','clone','--depth','1','https://github.com/ukumr59/HindiBHajans.git',str(REPO)],check=True)
else: subprocess.run(['git','-C',str(REPO),'pull','--ff-only','origin','main'],check=True)
seconds=int(os.getenv('BH_RUN_SECONDS','10'))
if not (8<=seconds<=15 or (180<=seconds<=300 and seconds%15==0)): raise SystemExit('BH_RUN_SECONDS must be 8-15 for smoke or 180-300 divisible by 15 for production')
(REPO/'app'/'run_config.json').write_text(json.dumps({'run_id':'saturn','smoke_test':seconds<180,'video_seconds':seconds})+'\n',encoding='utf-8')
subprocess.run([sys.executable,str(REPO/'app'/'worker.py')],cwd=str(REPO),check=True)
src=ROOT/'bhajan_aabha_exact_identity.mp4'
if not src.exists(): raise SystemExit('SATURN_WORKER_FAILED: expected master MP4 not found')
token=os.getenv('BH_GITHUB_TOKEN','').strip()
if not token: raise SystemExit('SATURN_WORKER_FAILED: BH_GITHUB_TOKEN missing')
owner='ukumr59'; repo='HindiBHajans'; tag='bhajan-saturn-latest'; api=f'https://api.github.com/repos/{owner}/{repo}'
base_headers={'Accept':'application/vnd.github+json','Authorization':f'Bearer {token}','X-GitHub-Api-Version':'2026-03-10'}
def call(url,method='GET',data=None,content_type='application/json'):
    h=dict(base_headers); h['Content-Type']=content_type
    body=json.dumps(data).encode() if data is not None else None
    r=urllib.request.Request(url,data=body,headers=h,method=method)
    try:
        with urllib.request.urlopen(r,timeout=120) as x:return json.loads(x.read().decode())
    except urllib.error.HTTPError as e:
        if e.code==404:return None
        raise
release=call(f'{api}/releases/tags/{tag}')
if not release: release=call(f'{api}/releases','POST',{'tag_name':tag,'name':'Bhajan Aabha Saturn Latest','body':'Latest zero-cost Saturn GPU output','draft':False,'prerelease':False,'make_latest':False})
for asset in release.get('assets',[]):
    if asset['name']=='master.mp4': call(f"{api}/releases/assets/{asset['id']}",'DELETE')
data=src.read_bytes(); upload=f"https://uploads.github.com/repos/{owner}/{repo}/releases/{release['id']}/assets?name=master.mp4"
r=urllib.request.Request(upload,data=data,headers={**base_headers,'Content-Type':'video/mp4'},method='POST')
with urllib.request.urlopen(r,timeout=900) as x: uploaded=json.loads(x.read().decode())
print('SATURN_WORKER_OUTPUT_URL='+uploaded['browser_download_url'],flush=True)
print('SATURN_WORKER_OUTPUT_SIZE='+str(src.stat().st_size),flush=True)
