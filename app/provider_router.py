"""Model-aware zero-cost provider router for Bhajan Aabha."""
from __future__ import annotations
import json, os, time, urllib.error, urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

STATE=Path(__file__).resolve().parents[1]/"state"; STATE.mkdir(exist_ok=True)
MODELS={
 "LONGCAT_AVATAR_15":{"min_vram_gb":24,"caps":["image_audio","singing","identity_lock","full_body","long_video"]},
 "ECHOMIMIC_V3_FLASH":{"min_vram_gb":12,"caps":["image_audio","singing","identity_lock","long_video"]},
 "WAN22_S2V":{"min_vram_gb":24,"caps":["image_audio","singing","identity_lock","full_body","long_video"]},
}
@dataclass(frozen=True)
class Provider: name:str; endpoint:str; token:str|None; model:str

def env(k):
 v=os.getenv(k,"").strip(); return v or None

def providers():
 out=[]
 for name in (env("BH_PROVIDER_ORDER") or "LONGCAT,ECHOMIMIC,WAN22").split(","):
  name=name.strip().upper()
  ep=env(f"BH_{name}_ENDPOINT")
  if not ep: continue
  model=env(f"BH_{name}_MODEL") or {"LONGCAT":"LONGCAT_AVATAR_15","ECHOMIMIC":"ECHOMIMIC_V3_FLASH","WAN22":"WAN22_S2V"}.get(name,name)
  if model not in MODELS: raise RuntimeError(f"UNKNOWN_MODEL_PROFILE:{model}")
  out.append(Provider(name,ep,env(f"BH_{name}_TOKEN"),model))
 return out

def request(p,url,payload=None,timeout=60):
 data=json.dumps(payload).encode() if payload is not None else None
 h={"Accept":"application/json","User-Agent":"HindiBHajans/2.0"}
 if data: h["Content-Type"]="application/json"
 if p.token: h["Authorization"]=f"Bearer {p.token}"
 r=urllib.request.urlopen(urllib.request.Request(url,data=data,headers=h,method="POST" if data else "GET"),timeout=timeout)
 raw=r.read().decode(); return json.loads(raw) if raw else {}

def transient(e):
 return isinstance(e,urllib.error.HTTPError) and e.code in {408,425,429,500,502,503,504} or isinstance(e,(TimeoutError,urllib.error.URLError))

def poll(p,r):
 url=r.get("status_url") or (f"{p.endpoint.rstrip('/')}/{r['job_id']}" if r.get("job_id") else None)
 if not url:return r
 deadline=time.time()+int(env("BH_PROVIDER_TIMEOUT_SECONDS") or "5400")
 while time.time()<deadline:
  cur=request(p,url); s=str(cur.get("status") or cur.get("state") or "").lower()
  print(f"GPU_PROVIDER={p.name} MODEL={p.model} STATUS={s}",flush=True)
  if s in {"completed","success","succeeded","finished"}:
   if not cur.get("video_url"): raise RuntimeError("PROVIDER_COMPLETED_WITHOUT_VIDEO_URL")
   return {**r,**cur}
  if s in {"failed","error","cancelled","canceled","quota_exhausted"}: raise RuntimeError(cur.get("error") or s)
  time.sleep(max(10,int(env("BH_POLL_SECONDS") or "20")))
 raise TimeoutError("PROVIDER_JOB_TIMEOUT")

def run_with_failover(job:dict[str,Any])->dict[str,Any]:
 ps=providers()
 if not ps: raise RuntimeError("NO_CONFIGURED_ZERO_COST_GPU_PROVIDER")
 required={"image_audio","singing","identity_lock"}; attempts=max(1,int(env("BH_PROVIDER_ATTEMPTS") or "2")); last=None
 for p in ps:
  profile=MODELS[p.model]
  if not required.issubset(profile["caps"]): continue
  for n in range(1,attempts+1):
   try:
    payload={**job,"model":p.model,"model_requirements":profile,"zero_cost_required":True,"provider_contract_version":2}
    r=request(p,p.endpoint,payload)
    if r.get("job_id") and not r.get("video_url"): r=poll(p,r)
    if r.get("video_url"): return {**r,"provider":p.name,"model":p.model}
    raise RuntimeError(r.get("error") or "PROVIDER_RETURNED_NO_VIDEO")
   except Exception as e:
    last=e; print(f"GPU_PROVIDER_ERROR={p.name} attempt={n}: {e}",flush=True)
    if transient(e) and n<attempts: time.sleep((20,60)[min(n-1,1)])
    else: break
 raise RuntimeError(f"ALL_CONFIGURED_ZERO_COST_PROVIDERS_FAILED:{last}")

if __name__=="__main__": print(json.dumps([p.__dict__ for p in providers()],indent=2))
