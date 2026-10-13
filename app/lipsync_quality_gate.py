from __future__ import annotations
import argparse, hashlib, json, subprocess, tempfile
from pathlib import Path
import cv2, numpy as np, torch
from scipy.io import wavfile
from scipy.fft import dct

# ---------------------------------------------------------------------------
# Self-contained MFCC — bit-for-bit replicates python_speech_features.mfcc()
# output shape (N, 13) float64 so SyncNet acoustic features match exactly.
# Only depends on scipy.fft.dct + numpy, which are always present.
# ---------------------------------------------------------------------------
def hz2mel(hz):
    return 2595 * np.log10(1 + hz / 700.0)

def mel2hz(mel):
    return 700 * (10 ** (mel / 2595.0) - 1)

def get_filterbanks(nfilt=26, nfft=512, samplerate=16000, lowfreq=0, highfreq=None):
    highfreq = highfreq or samplerate / 2
    lowmel = hz2mel(lowfreq)
    highmel = hz2mel(highfreq)
    melpoints = np.linspace(lowmel, highmel, nfilt + 2)
    bin_pts = np.floor((nfft + 1) * mel2hz(melpoints) / samplerate)
    fbank = np.zeros([nfilt, nfft // 2 + 1])
    for j in range(0, nfilt):
        for i in range(int(bin_pts[j]), int(bin_pts[j + 1])):
            fbank[j, i] = (i - bin_pts[j]) / (bin_pts[j + 1] - bin_pts[j])
        for i in range(int(bin_pts[j + 1]), int(bin_pts[j + 2])):
            fbank[j, i] = (bin_pts[j + 2] - i) / (bin_pts[j + 2] - bin_pts[j + 1])
    return fbank

def lifter(cepstra, L=22):
    if L > 0:
        nframes, ncoeff = np.shape(cepstra)
        n = np.arange(ncoeff)
        lift = 1 + (L / 2.0) * np.sin(np.pi * n / L)
        return lift * cepstra
    return cepstra

def _mfcc(signal, samplerate=16000, winlen=0.025, winstep=0.01, numcep=13,
          nfilt=26, nfft=512, lowfreq=0, highfreq=None, preemph=0.97, ceplifter=22, appendEnergy=True):
    highfreq = highfreq or samplerate / 2
    # Pre-emphasis
    signal = np.append(signal[0], signal[1:] - preemph * signal[:-1])
    # Framing (rectangular window by default in python_speech_features)
    frame_len = int(round(winlen * samplerate))
    frame_step = int(round(winstep * samplerate))
    slen = len(signal)
    numframes = 1 if slen <= frame_len else 1 + int(np.ceil((1.0 * slen - frame_len) / frame_step))
    padlen = int((numframes - 1) * frame_step + frame_len)
    padsignal = np.concatenate((signal, np.zeros(padlen - slen)))
    indices = (np.tile(np.arange(0, frame_len), (numframes, 1)) +
               np.tile(np.arange(0, numframes * frame_step, frame_step), (frame_len, 1)).T).astype(np.int32)
    frames = padsignal[indices]
    # Power spectrum
    mag = np.absolute(np.fft.rfft(frames, nfft))
    pspec = (1.0 / nfft) * (mag ** 2)
    pspec = np.where(pspec == 0, np.finfo(float).eps, pspec)
    energy = np.sum(pspec, 1)
    energy = np.where(energy == 0, np.finfo(float).eps, energy)
    # Filterbanks
    fb = get_filterbanks(nfilt, nfft, samplerate, lowfreq, highfreq)
    feat = np.dot(pspec, fb.T)
    feat = np.where(feat == 0, np.finfo(float).eps, feat)
    feat = np.log(feat)
    feat = dct(feat, type=2, axis=1, norm='ortho')[:, :numcep]
    feat = lifter(feat, ceplifter)
    if appendEnergy:
        feat[:, 0] = np.log(energy)
    return feat

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

def detect_bbox(frame,det):
    h,w=frame.shape[:2]; det.setInputSize((w,h)); _,faces=det.detect(frame)
    if faces is None or len(faces)==0:return None
    x,y,bw,bh=max(faces,key=lambda z:float(z[2]*z[3]))[:4]
    side=max(float(bw),float(bh)); cx=float(x+bw/2); cy=float(y+bh/2); half=side*0.75
    x0=max(0,int(cx-half)); x1=min(w,int(cx+half)); y0=max(0,int(cy-half)); y1=min(h,int(cy+half))
    return (x0,y0,x1,y1) if x1>x0 and y1>y0 else None

def sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def evaluate(video,model_path,yunet_path,batch=32,vshift=15):
    from collections import deque
    device='cuda' if torch.cuda.is_available() else 'cpu'; model=load_model(model_path,device)
    det=cv2.FaceDetectorYN.create(str(yunet_path),'',(320,320),0.60,0.30,20)
    with tempfile.TemporaryDirectory() as td:
        wav=Path(td)/'a.wav'
        subprocess.run(['ffmpeg','-y','-v','error','-i',str(video),'-ac','1','-ar','16000','-vn',str(wav)],check=True)
        sr,audio=wavfile.read(wav); mfcc=_mfcc(audio.astype(np.float64),sr).T
        cap=cv2.VideoCapture(str(video)); total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        last=min(total,mfcc.shape[1]//4)-5
        bbox=None; missing=0; imf={}; aff={}
        win=deque(maxlen=5)
        lips_b=[]; aud_b=[]; ids_b=[]
        def flush_batch():
            if not lips_b: return
            lt=torch.from_numpy(np.stack(lips_b).astype('float32')).to(device)
            at=torch.from_numpy(np.stack(aud_b).astype('float32'))[:,None,:,:].to(device)
            with torch.no_grad(): lv=model.vid(lt).cpu(); av=model.aud(at).cpu()
            for k,j in enumerate(ids_b): imf[j]=lv[k]; aff[j]=av[k]
            lips_b.clear(); aud_b.clear(); ids_b.clear()
        for idx in range(max(0,last+4)):
            ok,f=cap.read()
            if not ok or f is None:
                missing+=1; win.append(None)
            else:
                if bbox is None:
                    bbox=detect_bbox(f,det)
                if bbox is None:
                    missing+=1; win.append(None)
                else:
                    x0,y0,x1,y1=bbox
                    z=f[y0:y1,x0:x1]
                    win.append(cv2.resize(z,(224,224),interpolation=cv2.INTER_AREA) if z.size else None)
            if idx>=4:
                j=idx-4
                if j<last and len(win)==5 and all(x is not None for x in win) and (j*4+20)<=mfcc.shape[1]:
                    lips_b.append(np.transpose(np.stack(list(win)),(3,0,1,2)))
                    aud_b.append(mfcc[:,j*4:j*4+20])
                    ids_b.append(j)
                    if len(lips_b)>=batch:
                        flush_batch()
        flush_batch()
        cap.release()
        ids=sorted(set(imf)&set(aff))
        if len(ids)<100:raise RuntimeError(f'LIPSYNC_TOO_FEW_VALID_WINDOWS={len(ids)}')
        iv=torch.stack([imf[i] for i in ids]); av=torch.stack([aff[i] for i in ids])
        pad=torch.nn.functional.pad(av,(0,0,vshift,vshift)); d=[]
        for i in range(len(iv)):d.append(torch.nn.functional.pairwise_distance(iv[i:i+1].repeat(2*vshift+1,1),pad[i:i+2*vshift+1]))
        dm=torch.stack(d,1).mean(1); minval,minidx=torch.min(dm,0); offset=vshift-int(minidx); conf=float(torch.median(dm)-minval)
        fconf=np.asarray([float(torch.median(x)-x[int(minidx)]) for x in d],dtype=np.float64)
        local=[float(np.median(fconf[i:i+375])) for i in range(0,len(fconf),375) if len(fconf[i:i+375])]
        r={'device':device,'windows':len(ids),'av_offset_frames':offset,'av_offset_ms':round(offset*40,1),'confidence':round(conf,3),'local_median_confidence':round(float(np.median(local)),3) if local else 0.0,'local_min_confidence':round(min(local),3) if local else 0.0,'missing_face_windows':missing,'syncnet_model_sha256':sha256(model_path),'thresholds':{'min_confidence':0.05,'min_windows':100}}
        r['PASS']=len(ids)>=100 and conf>=0.05
        return r

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('video',type=Path); ap.add_argument('--model',type=Path,required=True); ap.add_argument('--yunet',type=Path,required=True); ap.add_argument('--report',type=Path,required=True); a=ap.parse_args()
    r=evaluate(a.video,a.model,a.yunet); print(json.dumps(r,indent=2),flush=True); a.report.write_text(json.dumps(r,indent=2))
    if not r['PASS']:raise SystemExit('LIPSYNC_QUALITY_GATE_FAILED')
if __name__=='__main__':main()
