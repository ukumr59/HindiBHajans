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

    inf_p = repo/'scripts/inference.py'
    inf_s = inf_p.read_text(encoding='utf-8')
    start_marker = '            # Pad generated images to original video size'
    end_marker = '        except Exception as e:'
    s_idx = inf_s.find(start_marker)
    e_idx = inf_s.find(end_marker)
    if s_idx != -1 and e_idx != -1 and s_idx < e_idx:
        optimized_code = '''            # Pad generated images to original video size (OPTIMIZED ZERO-DISK STREAMING)
            print("Starting zero-disk video stream composition...", flush=True)
            from musetalk.utils.blending import get_image_prepare_material
            from PIL import Image

            # Pre-compute blending mask once per unique frame to avoid redundant neural inferences
            material_cache = {}
            last_valid_coord = None
            for idx_c in range(len(coord_list_cycle)):
                cbx, cby, cbx1, cby1 = coord_list_cycle[idx_c]
                if cbx1 > cbx and cby1 > cby:
                    last_valid_coord = (cbx, cby, cbx1, cby1)
                elif last_valid_coord is not None:
                    coord_list_cycle[idx_c] = last_valid_coord
            for idx_c in range(len(coord_list_cycle)):
                cbx, cby, cbx1, cby1 = coord_list_cycle[idx_c]
                if (cbx1 <= cbx or cby1 <= cby) and last_valid_coord is not None:
                    coord_list_cycle[idx_c] = last_valid_coord

            for idx_c in range(len(frame_list_cycle)):
                ori_c = frame_list_cycle[idx_c]
                bx, by, bx1, by1 = coord_list_cycle[idx_c]
                if args.version == "v15":
                    by1 = min(by1 + args.extra_margin, ori_c.shape[0])
                m_arr, c_box = get_image_prepare_material(ori_c, [bx, by, bx1, by1], fp=fp, mode=args.parsing_mode)
                m_pil = Image.fromarray(m_arr).convert("L")
                material_cache[idx_c] = (bx, by, bx1, by1, m_pil, c_box)
            print("Facial blending masks cached successfully", flush=True)

            temp_vid_path = f"{temp_dir}/temp_{input_basename}_{audio_basename}.mp4"
            h_out, w_out = frame_list_cycle[0].shape[:2]
            ffmpeg_cmd = [
                "ffmpeg", "-y", "-v", "error",
                "-f", "rawvideo", "-vcodec", "rawvideo",
                "-s", f"{w_out}x{h_out}", "-pix_fmt", "bgr24",
                "-r", str(fps), "-i", "-",
                "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                temp_vid_path
            ]
            ff_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
            for i, res_frame in enumerate(res_frame_list):
                c_idx = i % len(frame_list_cycle)
                x, y, x1, y1, m_pil, (x_s, y_s, x_e, y_e) = material_cache[c_idx]
                ori_frame = copy.deepcopy(frame_list_cycle[c_idx])
                try:
                    res_frame = cv2.resize(res_frame.astype(np.uint8), (x1 - x, y1 - y))
                    body = Image.fromarray(ori_frame[:, :, ::-1])
                    face = Image.fromarray(res_frame[:, :, ::-1])
                    face_large = body.crop((x_s, y_s, x_e, y_e))
                    face_large.paste(face, (x - x_s, y - y_s, x1 - x_s, y1 - y_s))
                    body.paste(face_large, (x_s, y_s), m_pil)
                    combine_frame = np.array(body)[:, :, ::-1]
                except Exception as ex:
                    if i == 0: print(f"Frame blending fallback: {ex}", flush=True)
                    combine_frame = ori_frame
                ff_proc.stdin.write(combine_frame.tobytes())
                if (i + 1) % 500 == 0 or i == len(res_frame_list) - 1:
                    print(f"Composited {i + 1}/{len(res_frame_list)} frames", flush=True)
            ff_proc.stdin.close()
            ff_proc.wait()
            if ff_proc.returncode != 0:
                raise RuntimeError(f"FFmpeg compositing pipe failed with code {ff_proc.returncode}")

            if args.output_vid_name is None:
                output_vid_name = os.path.join(temp_dir, output_basename + ".mp4")
            else:
                output_vid_name = os.path.join(temp_dir, args.output_vid_name)
            cmd_combine_audio = [
                "ffmpeg", "-y", "-v", "error",
                "-i", audio_path, "-i", temp_vid_path,
                "-c:v", "copy", "-c:a", "aac", output_vid_name
            ]
            subprocess.run(cmd_combine_audio, check=True)
            if os.path.exists(temp_vid_path): os.remove(temp_vid_path)
            if "save_dir_full" in locals() and os.path.exists(save_dir_full): shutil.rmtree(save_dir_full, ignore_errors=True)
            if not args.saved_coord and os.path.exists(crop_coord_save_path): os.remove(crop_coord_save_path)
            print(f"Results saved to {output_vid_name}", flush=True)
'''
        inf_s = inf_s[:s_idx] + optimized_code + inf_s[e_idx:]
    else:
        t_err = 'y2 = min(y2, frame.shape[0])'
        idx = inf_s.rfind(t_err)
        if idx != -1:
            inf_s = inf_s[:idx] + 'y2 = min(y2, ori_frame.shape[0])' + inf_s[idx+len(t_err):]
        inf_s = inf_s.replace('shutil.rmtree(save_dir_full)', 'if "save_dir_full" in locals(): shutil.rmtree(save_dir_full, ignore_errors=True)')

    inf_s = inf_s.replace(
        'except Exception as e:\n            print("Error occurred during processing:", e)',
        'except Exception as e:\n            import traceback; traceback.print_exc()\n            raise'
    )
    inf_p.write_text(inf_s, encoding='utf-8')

def generate_liveportrait_motion(singer_image: Path, output_video: Path) -> bool:
    try:
        print("STAGE 1: GENERATING NATURAL DEVOTIONAL SINGER MOTION VIA LIVEPORTRAIT...", flush=True)
        lp_runtime = Path('/kaggle/tmp/liveportrait')
        lp_runtime.mkdir(parents=True, exist_ok=True)
        lp_repo = lp_runtime / 'LivePortrait'
        if not lp_repo.exists():
            run('git', 'clone', '--depth', '1', 'https://github.com/KlingAIResearch/LivePortrait.git', str(lp_repo))

        weights_dir = lp_repo / 'pretrained_weights'
        from huggingface_hub import snapshot_download, hf_hub_download
        snapshot_download('KwaiVGI/LivePortrait', local_dir=str(weights_dir), ignore_patterns=['liveportrait_animals/*'])

        # Patch helper.py to ensure torch.load uses weights_only=False for PyTorch 2.6+
        helper_file = lp_repo / 'src/utils/helper.py'
        if helper_file.exists():
            h_text = helper_file.read_text(encoding='utf-8')
            h_text = h_text.replace('torch.load(ckpt_path,', 'torch.load(ckpt_path, weights_only=False,')
            helper_file.write_text(h_text, encoding='utf-8')

        # Install tyro & rich if needed
        run(sys.executable, '-m', 'pip', 'install', '-q', 'tyro', 'rich')

        # Devotional driving video: calm demeanor, glasses, natural eye blinks and subtle nods
        driving_video = lp_repo / 'assets/examples/driving/d13.mp4'
        if not driving_video.exists():
            candidates = list((lp_repo / 'assets/examples/driving').glob('*.mp4'))
            if candidates:
                driving_video = candidates[0]
            else:
                driving_video = Path(hf_hub_download(repo_id='KlingAIResearch/LivePortrait', filename='assets/examples/driving/d13.mp4', local_dir=str(lp_repo)))

        lp_out = lp_runtime / 'output'
        lp_out.mkdir(parents=True, exist_ok=True)

        run(
            sys.executable, 'inference.py',
            '-s', str(singer_image),
            '-d', str(driving_video),
            '--flag_pasteback', 'True',
            '--flag_do_crop', 'True',
            '--flag_relative_motion', 'True',
            '--driving_option', 'expression-friendly',
            '-o', str(lp_out),
            cwd=lp_repo
        )

        candidates = [f for f in lp_out.glob('*.mp4') if not f.name.endswith('_concat.mp4')]
        if not candidates:
            candidates = list(lp_out.glob('*.mp4'))
        if candidates:
            # Re-encode to standard 25 fps H.264 matching MuseTalk FPS
            run('ffmpeg', '-y', '-v', 'error', '-i', str(candidates[0]), '-r', str(FPS), '-pix_fmt', 'yuv420p', str(output_video))
            print(f"STAGE 1 COMPLETE: Devotional motion base generated: {output_video} (size={output_video.stat().st_size})", flush=True)
            return True
        else:
            print("STAGE 1: No output video found in LivePortrait output dir", flush=True)
            return False
    except Exception as e:
        import traceback
        print(f"STAGE 1 LIVEPORTRAIT FALLBACK: {e}", flush=True)
        traceback.print_exc()
        return False

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
    # Use Kaggle's pre-installed PyTorch/Diffusers/Transformers/NumPy 2.x stack directly
    # (downgrading numpy to 1.26.4 breaks Python 3.13 jax/opencv/transformers on Kaggle).
    run(sys.executable,'-c','import diffusers, accelerate, transformers, omegaconf, librosa, einops, scipy, safetensors; from diffusers import AutoencoderKL; print("DEPENDENCY_SMOKE_OK", diffusers.__version__, transformers.__version__)')
    run('sudo','apt-get','update','-qq'); run('sudo','apt-get','install','-y','-qq','ffmpeg')
    models=repo/'models'
    for d in ('musetalkV15','sd-vae','whisper','face-parse-bisent','face_detection'): (models/d).mkdir(parents=True,exist_ok=True)
    from huggingface_hub import snapshot_download
    snapshot_download('TMElyralab/MuseTalk',local_dir=str(models),allow_patterns=['musetalkV15/musetalk.json','musetalkV15/unet.pth'])
    snapshot_download('stabilityai/sd-vae-ft-mse',local_dir=str(models/'sd-vae'),allow_patterns=['config.json','diffusion_pytorch_model.bin'])
    snapshot_download('openai/whisper-tiny',local_dir=str(models/'whisper'),allow_patterns=['config.json','pytorch_model.bin','preprocessor_config.json'])
    from huggingface_hub import hf_hub_download
    hf_hub_download(repo_id='opencv/face_detection_yunet', filename='face_detection_yunet_2023mar.onnx', local_dir=str(models/'face_detection'))
    # Ensure standard filename location expected by MuseTalk
    yunet_f = list((models/'face_detection').rglob('*yunet*.onnx'))
    if yunet_f and not (models/'face_detection/yunet_2023mar.onnx').exists():
        shutil.copy2(yunet_f[0], models/'face_detection/yunet_2023mar.onnx')

    # Replace fragile Google Drive gdown with reliable Hugging Face download
    hf_hub_download(repo_id='afrizalha/musetalk-models', filename='face-parse-bisent/79999_iter.pth', local_dir=str(models))
    hf_hub_download(repo_id='afrizalha/musetalk-models', filename='face-parse-bisent/resnet18-5c106cde.pth', local_dir=str(models))
    bisent_dir = models / 'face-parse-bisent'
    bisent_dir.mkdir(parents=True, exist_ok=True)
    for f in models.rglob('*79999_iter.pth'):
        if f != bisent_dir / '79999_iter.pth': shutil.copy2(f, bisent_dir / '79999_iter.pth')
    for f in models.rglob('*resnet18-5c106cde.pth'):
        if f != bisent_dir / 'resnet18-5c106cde.pth': shutil.copy2(f, bisent_dir / 'resnet18-5c106cde.pth')
    patch_runtime(repo)
    data=repo/'data'; data.mkdir(exist_ok=True)
    shutil.copy2(image,data/'singer.png'); shutil.copy2(audio,data/'bhajan.mp3')

    # Stage 1: Generate natural devotional motion base with LivePortrait
    motion_video = data / 'singer_motion.mp4'
    has_motion = generate_liveportrait_motion(data / 'singer.png', motion_video)
    target_video_input = 'data/singer_motion.mp4' if (has_motion and motion_video.exists() and motion_video.stat().st_size > 50000) else 'data/singer.png'
    print(f'MUSETALK_TARGET_INPUT: {target_video_input}', flush=True)

    cfg=repo/'configs/inference/musetalk_runtime.yaml'
    cfg.write_text(f'task_0:\n  video_path: "{target_video_input}"\n  audio_path: "data/bhajan.mp3"\n  result_name: "master.mp4"\n',encoding='utf-8')
    disk('before_inference'); started=time.time()
    run(sys.executable,'-m','scripts.inference','--inference_config',str(cfg),'--result_dir','results/runtime','--unet_model_path','models/musetalkV15/unet.pth','--unet_config','models/musetalkV15/musetalk.json','--whisper_dir','models/whisper','--version','v15','--fps',str(FPS),'--batch_size','8','--use_float16','--extra_margin','15','--output_vid_name','master.mp4',cwd=repo)
    print(f'MUSETALK_INFERENCE_SECONDS={time.time()-started:.1f}',flush=True)
    candidates=list((repo/'results/runtime/v15').glob('master.mp4'))
    if not candidates: candidates=list(repo.rglob('master.mp4'))
    if not candidates: raise RuntimeError('MUSETALK_MASTER_MISSING')
    final=FINAL/'master.mp4'
    run('ffmpeg','-y','-v','error','-i',str(candidates[0]),'-t',str(duration),'-c:v','libx264','-preset','veryfast','-crf','20','-pix_fmt','yuv420p','-c:a','aac','-b:a','160k','-ar','48000','-movflags','+faststart',str(final))
    mb=final.stat().st_size/1024**2
    print(f'MUSETALK_MASTER_READY path={final} size_mb={mb:.1f} duration={duration}s',flush=True)
    if mb>1536: raise RuntimeError(f'MASTER_TOO_LARGE_MB={mb:.1f}')
    # HARD audio/video lip-sync gate.
    syncnet_path=FINAL/'syncnet_v2.model'
    syncnet_src=hf_hub_download(repo_id='ByteDance/LatentSync-1.6',filename='auxiliary/syncnet_v2.model')
    shutil.copy2(syncnet_src,syncnet_path)
    import hashlib
    expected='961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442'
    got=hashlib.sha256(syncnet_path.read_bytes()).hexdigest()
    if got!=expected: raise RuntimeError(f'SYNCNET_MODEL_SHA256_MISMATCH={got}')
    run(sys.executable,'/kaggle/working/lipsync_quality_gate.py',str(final),'--model',str(syncnet_path),'--yunet',str(models/'face_detection/yunet_2023mar.onnx'),'--report',str(FINAL/'lipsync_qa.json'))
    print('LIPSYNC_QUALITY_GATE=PASS',flush=True)
    syncnet_path.unlink(missing_ok=True)
    shutil.rmtree(RUNTIME,ignore_errors=True)
    shutil.rmtree(Path('/kaggle/tmp/liveportrait'),ignore_errors=True)
    print('MUSETALK_WORKER_OK',flush=True)

if __name__=='__main__':
    main()
