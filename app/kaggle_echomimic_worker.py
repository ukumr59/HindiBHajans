from __future__ import annotations
import os, shutil, subprocess, sys, time
from pathlib import Path

RUNTIME=Path('/kaggle/tmp/hindibhajans_musetalk')
FINAL=Path('/kaggle/working')
INPUT=Path('/kaggle/input')
FPS=25

def run(*args, cwd=None):
    print('RUN:', *args, flush=True)
    subprocess.run([str(x) for x in args], cwd=str(cwd) if cwd else None, check=True)

def find_input(name):
    xs=[p for p in INPUT.rglob(name) if p.is_file()]
    if not xs: raise RuntimeError(f'KAGGLE_INPUT_FILE_MISSING: {name}')
    return xs[0]

def disk(label):
    u=shutil.disk_usage('/kaggle/working')
    print(f'DISK {label} free_gb={u.free/1024**3:.2f}', flush=True)
    if u.free < 5*1024**3: raise RuntimeError('WORKING_DISK_LOW')

def patch_runtime(repo):
    # Avoid OpenMMLab/mmpose on Kaggle. For our single reference image we only
    # need a stable face box; YuNet is tiny, fast and already used by our QA gate.
    p=repo/'musetalk/utils/preprocessing.py'
    p.write_text(r"""import cv2, numpy as np
from tqdm import tqdm
coord_placeholder=(0.0,0.0,0.0,0.0)
detector=None
YUNET='./models/face_detection/yunet_2023mar.onnx'

def read_imgs(img_list):
    out=[]
    for x in tqdm(img_list,desc='reading images'):
        im=cv2.imread(x)
        if im is None: raise RuntimeError(f'IMAGE_READ_FAILED {x}')
        out.append(im)
    return out

def get_landmark_and_bbox(img_list, upperbondrange=0):
    global detector
    frames=read_imgs(img_list); boxes=[]
    for im in tqdm(frames,desc='YuNet face detection'):
        h,w=im.shape[:2]
        if detector is None:
            detector=cv2.FaceDetectorYN.create(YUNET,'',(w,h),0.60,0.30,5000)
        else:
            detector.setInputSize((w,h))
        _,faces=detector.detect(im)
        if faces is None or len(faces)==0:
            boxes.append(coord_placeholder); continue
        f=max(faces,key=lambda z:float(z[2]*z[3]))
        x,y,bw,bh=[float(v) for v in f[:4]]
        px,pyt,pyb=0.08*bw,0.10*bh,0.18*bh
        x1=max(0,int(x-px)); y1=max(0,int(y-pyt))
        x2=min(w,int(x+bw+px)); y2=min(h,int(y+bh+pyb))
        boxes.append((x1,y1,x2,y2) if x2>x1 and y2>y1 else coord_placeholder)
    return boxes,frames
""",encoding='utf-8')

    for rel in ('musetalk/utils/face_parsing/__init__.py','musetalk/utils/face_parsing/resnet.py'):
        p=repo/rel; s=p.read_text()
        s=s.replace('torch.load(model_pth))','torch.load(model_pth, weights_only=False))')
        s=s.replace("torch.load(model_pth, map_location=torch.device('cpu'))","torch.load(model_pth, map_location=torch.device('cpu'), weights_only=False)")
        s=s.replace('torch.load(model_path) #modelzoo.load_url','torch.load(model_path, weights_only=False) #modelzoo.load_url')
        p.write_text(s,encoding='utf-8')

def main():
    image=find_input('singer.png'); audio=find_input('bhajan.mp3'); duration=int(find_input('duration.txt').read_text())
    if not 180<=duration<=300 or duration%15: raise RuntimeError(f'INVALID_DURATION {duration}')
    print(f'MUSETALK_INPUT duration={duration}s image={image.stat().st_size} audio={audio.stat().st_size}',flush=True)
    run('nvidia-smi')
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('NO_GPU_ALLOCATED')
    print('GPU_READY',torch.cuda.get_device_name(0),torch.cuda.get_device_properties(0).total_memory,flush=True)
    RUNTIME.mkdir(parents=True,exist_ok=True)
    repo=RUNTIME/'MuseTalk'
    run('git','clone','--depth','1','https://github.com/TMElyralab/MuseTalk.git',str(repo))
    run(sys.executable,'-m','pip','install','-q','numpy==1.26.4','diffusers==0.30.2','accelerate==0.28.0','transformers==4.39.2','huggingface_hub==0.30.2','librosa==0.11.0','einops==0.8.1','omegaconf','ffmpeg-python','moviepy','gdown','safetensors')
    run('sudo','apt-get','update','-qq'); run('sudo','apt-get','install','-y','-qq','ffmpeg')
    models=repo/'models'
    for d in ('musetalkV15','sd-vae','whisper','face-parse-bisent','face_detection'): (models/d).mkdir(parents=True,exist_ok=True)
    from huggingface_hub import snapshot_download
    snapshot_download('TMElyralab/MuseTalk',local_dir=str(models),allow_patterns=['musetalkV15/musetalk.json','musetalkV15/unet.pth'])
    snapshot_download('stabilityai/sd-vae-ft-mse',local_dir=str(models/'sd-vae'),allow_patterns=['config.json','diffusion_pytorch_model.bin'])
    snapshot_download('openai/whisper-tiny',local_dir=str(models/'whisper'),allow_patterns=['config.json','pytorch_model.bin','preprocessor_config.json'])
    run('wget','-q','-O',str(models/'face_detection/yunet_2023mar.onnx'),'https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx')
    run('gdown','--id','154JgKpzCPW82qINcVieuPH3fZ2e0P812','-O',str(models/'face-parse-bisent/79999_iter.pth'))
    run('wget','-q','-O',str(models/'face-parse-bisent/resnet18-5c106cde.pth'),'https://download.pytorch.org/models/resnet18-5c106cde.pth')
    patch_runtime(repo)
    data=repo/'data'; data.mkdir(exist_ok=True)
    shutil.copy2(image,data/'singer.png'); shutil.copy2(audio,data/'bhajan.mp3')
    cfg=repo/'configs/inference/musetalk_runtime.yaml'
    cfg.write_text('task_0:\n  video_path: "data/singer.png"\n  audio_path: "data/bhajan.mp3"\n  result_name: "master.mp4"\n',encoding='utf-8')
    disk('before_inference'); started=time.time()
    run(sys.executable,'-m','scripts.inference','--inference_config',str(cfg),'--result_dir','results/runtime','--unet_model_path','models/musetalkV15/unet.pth','--unet_config','models/musetalkV15/musetalk.json','--whisper_dir','models/whisper','--version','v15','--fps',str(FPS),'--batch_size','8','--use_float16','--output_vid_name','master.mp4',cwd=repo)
    print(f'MUSETALK_INFERENCE_SECONDS={time.time()-started:.1f}',flush=True)
    candidates=list((repo/'results/runtime/v15').glob('master.mp4'))
    if not candidates: candidates=list(repo.rglob('master.mp4'))
    if not candidates: raise RuntimeError('MUSETALK_MASTER_MISSING')
    final=FINAL/'master.mp4'
    run('ffmpeg','-y','-v','error','-i',str(candidates[0]),'-t',str(duration),'-c:v','libx264','-preset','veryfast','-crf','20','-pix_fmt','yuv420p','-c:a','aac','-b:a','160k','-ar','48000','-movflags','+faststart',str(final))
    mb=final.stat().st_size/1024**2
    print(f'MUSETALK_MASTER_READY path={final} size_mb={mb:.1f} duration={duration}s',flush=True)
    if mb>1536: raise RuntimeError(f'MASTER_TOO_LARGE_MB={mb:.1f}')
    shutil.rmtree(RUNTIME,ignore_errors=True)
    print('MUSETALK_WORKER_OK',flush=True)

if __name__=='__main__':
    main()
