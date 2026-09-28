from __future__ import annotations
import argparse, hashlib, json, subprocess, tempfile
from pathlib import Path
import cv2, numpy as np, torch
from scipy.io import wavfile
import python_speech_features

class S(torch.nn.Module):
    def __init__(self,n=1024):
        super().__init__()
        self.a=torch.nn.Sequential(
            torch.nn.Conv2d(1,64,3,padding=1),torch.nn.BatchNorm2d(64),torch.nn.ReLU(),torch.nn.MaxPool2d((1,1)),
            torch.nn.Conv2d(64,192,3,padding=1),torch.nn.BatchNorm2d(192),torch.nn.ReLU(),torch.nn.MaxPool2d((3,3),stride=(1,2)),
            torch.nn.Conv2d(192,384,3,padding=1),torch.nn.BatchNorm2d(384),torch.nn.ReLU(),
            torch.nn.Conv2d(384,256,3,padding=1),torch.nn.BatchNorm2d(256),torch.nn.ReLU(),
            torch.nn.Conv2d(256,256,3,padding=1),torch.nn.BatchNorm2d(256),torch.nn.ReLU(),torch.nn.MaxPool2d(3,stride=2),
            torch.nn.Conv2d(256,512,(5,4)),torch.nn.BatchNorm2d(512),torch.nn.ReLU())
        self.af=torch.nn.Sequential(torch.nn.Linear(512,512),torch.nn.BatchNorm1d(512),torch.nn.ReLU(),torch.nn.Linear(512,n))
        self.vf=torch.nn.Sequential(torch.nn.Linear(512,512),torch.nn.BatchNorm1d(512),torch.nn.ReLU(),torch.nn.Linear(512,n))
        self.v=torch.nn.Sequential(
            torch.nn.Conv3d(3,96,(5,7,7),stride=(1,2,2)),torch.nn.BatchNorm3d(96),torch.nn.ReLU(),torch.nn.MaxPool3d((1,3,3),stride=(1,2,2)),
            torch.nn.Conv3d(96,256,(1,5,5),stride=(1,2,2),padding=(0,1,1)),torch.nn.BatchNorm3d(256),torch.nn.ReLU(),torch.nn.MaxPool3d((1,3,3),stride=(1,2,2),padding=(0,1,1)),
            torch.nn.Conv3d(256,256,(1,3,3),padding=(0,1,1)),torch.nn.BatchNorm3d(256),torch.nn.ReLU(),
            torch.nn.Conv3d(256,256,(1,3,3),padding=(0,1,1)),torch.nn.BatchNorm3d(256),torch.nn.ReLU(),
            torch.nn.Conv3d(256,256,(1,3,3),padding=(0,1,1)),torch.nn.BatchNorm3d(256),torch.nn.ReLU(),torch.nn.MaxPool3d((1,3,3),stride=(1,2,2)),
            torch.nn.Conv3d(256,512,(1,6,6)),torch.nn.BatchNorm3d(512),torch.nn.ReLU())
    def aud(self,x): return self.af(self.a(x).flatten(1))
    def vid(self,x): return self.vf(self.v(x).flatten(1))

def load_model(path,device):
    m=S().to(device); state=torch.load(path,map_location='cpu',weights_only=True); ms=m.state_dict()
    for k,v in state.items():
        if k in ms: ms[k].copy_(v)
    return m.eval()

def crop(frame,det):
    h,w=frame.shape[:2]; det.setInputSize((w,h)); _,faces=det.detect(frame)
    if faces is None or len(faces)==0:return None
    x,y,bw,bh=max(faces,key=lambda z:float(z[2]*z[3]))[:4]; side=max(float(bw),float(bh)); cx=float(x+bw/2); cy=float(y+bh/2); half=side*1.7
    x0=max(0,int(cx-half)); x1=min(w,int(cx+half)); y0=max(0,int(cy-half)); y1=min(h,int(cy+half))
    z=frame[y0:y1,x0:x1]
    return cv2.resize(z,(224,224),interpolation=cv2.INTER_AREA) if z.size else None

def sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def evaluate(video,model_path,yunet_path,batch=32,vshift=15):
    device='cuda' if torch.cuda.is_available() else 'cpu'; model=load_model(model_path,device)
    det=cv2.FaceDetectorYN.create(str(yunet_path),'',(320,320),0.85,0.30,20)
    with tempfile.TemporaryDirectory() as td:
        wav=Path(td)/'a.wav'
        subprocess.run(['ffmpeg','-y','-v','error','-i',str(video),'-ac','1','-ar','16000','-vn',str(wav)],check=True)
        sr,audio=wavfile.read(wav); mfcc=np.stack([np.asarray(x) for x in zip(*python_speech_features.mfcc(audio,sr))])
        cap=cv2.VideoCapture(str(video)); total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cache=[]; missing=0; imf={}; aff={}
        def frame(i):
            nonlocal missing
            if i<len(cache):return cache[i]
            cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,f=cap.read()
            if not ok: missing+=1; cache.append(None); return None
            z=crop(f,det); cache.append(z); return z
        last=min(total,len(audio)//640)-5
        for i in range(0,max(0,last),batch):
            lips=[]; aud=[]; ids=[]
            for j in range(i,min(last,i+batch)):
                fs=[frame(k) for k in range(j,j+5)]
                if any(x is None for x in fs):continue
                lips.append(np.transpose(np.stack(fs),(3,0,1,2))); aud.append(mfcc[:,j*4:j*4+20]); ids.append(j)
            if not lips:continue
            lt=torch.from_numpy(np.stack(lips).astype('float32')).to(device)
            at=torch.from_numpy(np.stack(aud).astype('float32'))[:,None,:,:].to(device)
            with torch.no_grad(): lv=model.vid(lt).cpu(); av=model.aud(at).cpu()
            for k,j in enumerate(ids):imf[j]=lv[k]; aff[j]=av[k]
        cap.release()
        ids=sorted(set(imf)&set(aff))
        if len(ids)<100:raise RuntimeError(f'LIPSYNC_TOO_FEW_VALID_WINDOWS={len(ids)}')
        iv=torch.stack([imf[i] for i in ids]); av=torch.stack([aff[i] for i in ids])
        pad=torch.nn.functional.pad(av,(0,0,vshift,vshift)); d=[]
        for i in range(len(iv)):d.append(torch.nn.functional.pairwise_distance(iv[i:i+1].repeat(2*vshift+1,1),pad[i:i+2*vshift+1]))
        dm=torch.stack(d,1).mean(1); minval,minidx=torch.min(dm,0); offset=vshift-int(minidx); conf=float(torch.median(dm)-minval)
        fdist=np.stack([x[int(minidx)].numpy() for x in d]); fconf=float(torch.median(dm))-fdist
        local=[float(np.median(fconf[i:i+375])) for i in range(0,len(fconf),375) if len(fconf[i:i+375])]
        r={'device':device,'windows':len(ids),'av_offset_frames':offset,'av_offset_ms':round(offset*40,1),'confidence':round(conf,3),'local_median_confidence':round(float(np.median(local)),3),'local_min_confidence':round(min(local),3),'missing_face_windows':missing,'syncnet_model_sha256':sha256(model_path),'thresholds':{'min_confidence':3.0,'min_local_confidence':2.5,'max_abs_offset_frames':2}}
        r['PASS']=conf>=3.0 and min(local)>=2.5 and abs(offset)<=2
        return r

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('video',type=Path); ap.add_argument('--model',type=Path,required=True); ap.add_argument('--yunet',type=Path,required=True); ap.add_argument('--report',type=Path,required=True); a=ap.parse_args()
    r=evaluate(a.video,a.model,a.yunet); print(json.dumps(r,indent=2),flush=True); a.report.write_text(json.dumps(r,indent=2))
    if not r['PASS']:raise SystemExit('LIPSYNC_QUALITY_GATE_FAILED')
if __name__=='__main__':main()
