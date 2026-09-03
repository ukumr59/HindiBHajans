"""Provider-neutral zero-cost GPU routing for HindiBHajans."""
from __future__ import annotations
import json, os, time, urllib.error, urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any
ROOT=Path(__file__).resolve().parents[1]; STATE_DIR=ROOT/'state'; STATE_DIR.mkdir(parents=True,exist_ok=True)
@dataclass(frozen=True)
class Provider:
    name:str; endpoint:str; token:str|None; priority:int

def _env(n:str)->str|None:
    v=os.getenv(n,'').strip(); return v or None

def load_providers()->list[Provider]:
    out=[]; raw=_env('BH_PROVIDER_ORDER') or 'SATURN,HF_SPACE,BEAM,KAGGLE'
    for i,name in enumerate(x.strip().upper() for x in raw.split(',')):
        ep=_env(f'BH_{name}_ENDPOINT')
        if ep: out.append(Provider(name,ep,_env(f'BH_{name}_TOKEN'),i))
    return out

def _headers(p:Provider,body:bool)->dict[str,str]:
    h={'Accept':'application/json'}
    if body:h['Content-Type']='application/json'
    if p.token:h['Authorization']=f'token {p.token}' if p.name=='SATURN' else f'Bearer {p.token}'
    return h

def _request(url:str,p:Provider,payload:dict[str,Any]|None=None,timeout:int=45)->dict[str,Any]:
    body=json.dumps(payload).encode() if payload is not None else None
    req=urllib.request.Request(url,data=body,headers=_headers(p,payload is not None),method='POST' if payload is not None else 'GET')
    with urllib.request.urlopen(req,timeout=timeout) as r:
        raw=r.read().decode(); return json.loads(raw) if raw else {}

def _is_transient(e:Exception)->bool:
    if isinstance(e,urllib.error.HTTPError): return e.code in {408,425,429,500,502,503,504}
    return isinstance(e,(TimeoutError,urllib.error.URLError))

def _record(e:dict[str,Any])->None:
    with (STATE_DIR/'provider-events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(e,ensure_ascii=False)+'\n')

def _saturn_start(p:Provider,job:dict[str,Any])->dict[str,Any]:
    job_id=_env('BH_SATURN_JOB_ID')
    if not job_id: raise RuntimeError('SATURN_CONFIG_FAILED: BH_SATURN_JOB_ID is required')
    base=p.endpoint.rstrip('/')
    url=f'{base}/{job_id}/start'
    r=_request(url,p,None)
    # Saturn start is asynchronous. Keep the job id and the API status URL.
    return {'status':'queued','job_id':job_id,'status_url':f'{base}/{job_id}','saturn_response':r}

def _poll(p:Provider,result:dict[str,Any])->dict[str,Any]:
    status_url=result.get('status_url'); job_id=result.get('job_id')
    if not status_url and job_id: status_url=f'{p.endpoint.rstrip("/")}/{job_id}'
    if not status_url:return result
    deadline=time.time()+int(os.getenv('BH_PROVIDER_TIMEOUT_SECONDS','5400')); interval=max(15,int(os.getenv('BH_POLL_SECONDS','30')))
    while time.time()<deadline:
        cur=_request(str(status_url),p); status=str(cur.get('status') or cur.get('state') or '').lower()
        print(f'GPU_PROVIDER={p.name} JOB={job_id} STATUS={status}',flush=True)
        if status in {'completed','success','succeeded','finished'}:
            video=cur.get('video_url') or _env('BH_SATURN_RESULT_URL')
            if not video: raise RuntimeError('SATURN job completed but BH_SATURN_RESULT_URL is not configured')
            return {**result,**cur,'video_url':video}
        if status in {'failed','error','cancelled','canceled'}: raise RuntimeError(cur.get('error') or f'Provider job failed: {status}')
        time.sleep(interval)
    raise TimeoutError('Provider job timed out')

def run_with_failover(job:dict[str,Any])->dict[str,Any]:
    providers=load_providers()
    if not providers:raise RuntimeError('NO_FREE_GPU_PROVIDER_CONFIGURED')
    attempts=max(1,int(os.getenv('BH_PROVIDER_ATTEMPTS','2'))); backoff=[15,45,120]; last=None
    for p in providers:
        for attempt in range(1,attempts+1):
            started=time.time()
            try:
                print(f'GPU_PROVIDER={p.name} ATTEMPT={attempt}',flush=True)
                result=_saturn_start(p,job) if p.name=='SATURN' else _request(p.endpoint,p,job)
                status=str(result.get('status','')).lower()
                if status in {'queued','running','processing','pending','starting'} or (result.get('job_id') and not result.get('video_url')): result=_poll(p,result)
                if result.get('video_url'):
                    _record({'provider':p.name,'attempt':attempt,'status':'success','seconds':round(time.time()-started,2)}); return {**result,'provider':p.name}
                raise RuntimeError(result.get('error') or f'Provider returned status={status!r}')
            except Exception as e:
                last=e; transient=_is_transient(e); _record({'provider':p.name,'attempt':attempt,'status':'error','transient':transient,'error':repr(e)}); print(f'GPU_PROVIDER_ERROR={p.name} transient={transient} error={e}',flush=True)
                if not transient or attempt>=attempts:break
                delay=backoff[min(attempt-1,len(backoff)-1)]; print(f'GPU_PROVIDER_BACKOFF={delay}s',flush=True); time.sleep(delay)
        print(f'GPU_PROVIDER_FAILOVER={p.name}',flush=True)
    raise RuntimeError(f'ALL_CONFIGURED_FREE_GPU_PROVIDERS_FAILED: {last}')

if __name__=='__main__':print(json.dumps({'providers':[p.name for p in load_providers()]},indent=2))
