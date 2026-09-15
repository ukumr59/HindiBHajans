from __future__ import annotations
import json, os, shutil, subprocess, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output'
KDIR=ROOT/'.kaggle_worker'
KERNEL='bhajanaabha/hindibhajans-echomimic-v3'


def worker(seconds:int)->str:
    return f'''import os, shutil, subprocess, sys
from pathlib import Path
ROOT=Path('/kaggle/working'); IN=Path('/kaggle/input'); REPO=ROOT/'echomimic_v3'; MODELS=ROOT/'models'; SEG=ROOT/'segments'; OUT=ROOT/'outputs'
SECONDS={seconds}; FPS=25; FRAMES=81; SEG_SECONDS=FRAMES/FPS

def run(*a):
    print('RUN:',*a,flush=True); subprocess.run([str(x) for x in a],check=True)
def find(name):
    xs=list(IN.rglob(name))
    if not xs: raise RuntimeError('KAGGLE_INPUT_FILE_MISSING: '+name)
    return xs[0]
def mp4(d):
    xs=sorted(Path(d).rglob('*.mp4'),key=lambda p:p.stat().st_mtime,reverse=True)
    if not xs: raise RuntimeError('NO_MP4_GENERATED')
    return xs[0]

def main():
    image=find('singer.png'); audio=find('bhajan.mp3'); dur=find('duration.txt')
    if int(dur.read_text().strip())!=SECONDS: raise RuntimeError('KAGGLE_DURATION_MISMATCH')
    print('INPUT_READY',image,audio,image.stat().st_size,audio.stat().st_size,flush=True)
    run('nvidia-smi')
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('NO_GPU_ALLOCATED')
    if torch.cuda.get_device_properties(0).total_memory < 13_000_000_000: raise RuntimeError('GPU_VRAM_TOO_SMALL')
    run('git','clone','--depth','1','https://github.com/antgroup/echomimic_v3.git',str(REPO))
    run(sys.executable,'-m','pip','install','-q','-r',str(REPO/'requirements.txt'))
    run(sys.executable,'-m','pip','install','-q','huggingface_hub')
    from huggingface_hub import snapshot_download
    MODELS.mkdir(exist_ok=True)
    base=MODELS/'Wan2.1-Fun-V1.1-1.3B-InP'; wav=MODELS/'chinese-wav2vec2-base'; flash=MODELS/'echomimicv3-flash-pro'
    if not base.exists(): snapshot_download('alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP',local_dir=str(base))
    if not wav.exists(): snapshot_download('TencentGameMate/chinese-wav2vec2-base',local_dir=str(wav))
    if not flash.exists(): snapshot_download('BadToBest/EchoMimicV3',local_dir=str(flash),allow_patterns=['echomimicv3-flash-pro/*'])
    SEG.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)
    norm=ROOT/'audio16k.wav'; run('ffmpeg','-y','-v','error','-i',str(audio),'-ac','1','-ar','16000','-c:a','pcm_s16le',str(norm))
    n=int((SECONDS+SEG_SECONDS-1)//SEG_SECONDS)
    for i in range(n):
        start=i*SEG_SECONDS; remain=max(.1,min(SEG_SECONDS,SECONDS-start))
        if remain<.5: break
        a=SEG/f'audio_{i:04d}.wav'; run('ffmpeg','-y','-v','error','-ss',f'{start:.3f}','-i',str(norm),'-t',f'{remain:.3f}','-ar','16000','-ac','1',str(a))
        od=SEG/f'raw_{i:04d}'; od.mkdir(exist_ok=True)
        run(sys.executable,str(REPO/'infer_flash.py'),'--image_path',str(image),'--audio_path',str(a),'--prompt','A single Indian devotional singer performing a Hindi bhajan in traditional Indian clothing before the specified Hindu deity in a serene temple setting; only the same singer is visible; natural singing mouth movement, subtle expressive head and upper-body motion, stable identity.','--num_inference_steps','8','--config_path',str(REPO/'config/config.yaml'),'--model_name',str(base),'--ckpt_idx','50000','--transformer_path',str(flash/'echomimicv3-flash-pro/transformer/diffusion_pytorch_model.safetensors'),'--save_path',str(od),'--wav2vec_model_dir',str(wav),'--sampler_name','Flow_Unipc','--video_length',str(FRAMES),'--guidance_scale','5.0','--audio_guidance_scale','2.5','--audio_scale','1.0','--neg_scale','1.0','--neg_steps','0','--seed',str(4300+i),'--enable_teacache','--teacache_threshold','0.1','--num_skip_start_steps','5','--weight_dtype','float16','--sample_size','768','768','--fps',str(FPS),'--negative_prompt','blurry, distorted face, identity drift, extra person, duplicate person, malformed hands, fused fingers, deformed mouth, jitter, flicker, camera cut, text, watermark')
        raw=mp4(od); silent=SEG/f'video_{i:04d}.mp4'; run('ffmpeg','-y','-v','error','-i',str(raw),'-an','-c:v','libx264','-preset','veryfast','-crf','20','-pix_fmt','yuv420p',str(silent))
    files=sorted(SEG.glob('video_*.mp4'))
    if not files: raise RuntimeError('NO_SEGMENTS_GENERATED')
    concat=SEG/'concat.txt'; concat.write_text(''.join(f"file \'{p.resolve()}\'\\n" for p in files))
    visual=OUT/'visual.mp4'; run('ffmpeg','-y','-v','error','-f','concat','-safe','0','-i',str(concat),'-c','copy',str(visual))
    final=OUT/'master.mp4'; run('ffmpeg','-y','-v','error','-i',str(visual),'-i',str(audio),'-map','0:v:0','-map','1:a:0','-t',str(SECONDS),'-c:v','copy','-c:a','aac','-b:a','192k','-ar','48000','-movflags','+faststart',str(final))
    if not final.exists() or final.stat().st_size<500000: raise RuntimeError('MASTER_NOT_CREATED')
    shutil.copy2(final,ROOT/'master.mp4'); print('BHAJAN_KAGGLE_WORKER_OK',flush=True)
main()
'''

def main(seconds):
    token=os.getenv('KAGGLE_API_TOKEN') or os.getenv('KAGGLE_API_TOKEN3')
    if not token: raise RuntimeError('KAGGLE_API_TOKEN missing')
    image=ROOT/'assets'/'singer_image.png'; audio=OUT/'bhajan_source.mp3'
    if not image.exists() or not audio.exists(): raise RuntimeError('canonical inputs missing')
    if not 180<=seconds<=300 or seconds%15: raise RuntimeError('invalid duration')
    shutil.rmtree(KDIR,ignore_errors=True); (KDIR/'input').mkdir(parents=True)
    shutil.copy2(image,KDIR/'input'/'singer.png'); shutil.copy2(audio,KDIR/'input'/'bhajan.mp3'); (KDIR/'input'/'duration.txt').write_text(str(seconds))
    (KDIR/'worker.py').write_text(worker(seconds))
    (KDIR/'kernel-metadata.json').write_text(json.dumps({'id':KERNEL,'title':'hindibhajans-echomimic-v3','code_file':'worker.py','language':'python','kernel_type':'script','is_private':True,'enable_gpu':True,'enable_internet':True,'machine_shape':'NvidiaTeslaT4','dataset_sources':[],'competition_sources':[],'kernel_sources':[],'model_sources':[]},indent=2))
    print('KAGGLE_WORKER_PACKAGE worker_bytes=',(KDIR/'worker.py').stat().st_size,'image_bytes=',image.stat().st_size,'audio_bytes=',audio.stat().st_size,flush=True)
    env=dict(os.environ); env['KAGGLE_API_TOKEN']=token
    subprocess.run(['kaggle','kernels','push','-p',str(KDIR),'--accelerator','NvidiaTeslaT4','--timeout','39600'],cwd=ROOT,env=env,check=True)
    deadline=time.time()+39600
    while time.time()<deadline:
        p=subprocess.run(['kaggle','kernels','status',KERNEL],capture_output=True,text=True,env=env); print(p.stdout or p.stderr,flush=True); t=(p.stdout+p.stderr).lower()
        if 'complete' in t: break
        if any(x in t for x in ('error','failed','cancelled','canceled')):
            log=subprocess.run(['kaggle','kernels','logs',KERNEL],capture_output=True,text=True,env=env); print(log.stdout or log.stderr,flush=True); raise RuntimeError('KAGGLE_KERNEL_FAILED')
        time.sleep(30)
    else: raise TimeoutError('KAGGLE_KERNEL_TIMEOUT')
    dest=OUT/'kaggle_output'; shutil.rmtree(dest,ignore_errors=True); dest.mkdir(parents=True)
    subprocess.run(['kaggle','kernels','output',KERNEL,'-p',str(dest),'--force','--file-pattern','.*master\\.mp4$'],cwd=ROOT,env=env,check=True)
    xs=list(dest.rglob('master.mp4'))
    if not xs: raise RuntimeError('KAGGLE_COMPLETED_BUT_MASTER_MP4_MISSING')
    shutil.copy2(xs[0],OUT/'master.mp4')
    print('KAGGLE_ECHOMIMIC_MASTER_READY',flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('--seconds',type=int,default=180); main(p.parse_args().seconds)
