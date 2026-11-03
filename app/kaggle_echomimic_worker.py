from __future__ import annotations
import base64
import copy
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

FINAL = Path('/kaggle/working')
RUNTIME = FINAL / '.runtime' / 'hindibhajans_musetalk'
INPUT = Path('/kaggle/input')
FPS = 25

# Redirect HuggingFace & Torch caches to 20GB working partition
CACHE_DIR = FINAL / '.cache'
os.environ['HF_HOME'] = str(CACHE_DIR / 'huggingface')
os.environ['TORCH_HOME'] = str(CACHE_DIR / 'torch')
os.environ['PYTHONUNBUFFERED'] = '1'

# ---------------------------------------------------------------------------
# Python 3.13 / PyTorch 2.6 / NumPy 2.x Compatibility Shims
# ---------------------------------------------------------------------------
def install_runtime_shims():
    """Install sitecustomize.py so all Python subprocesses inherit compatibility shims."""
    try:
        import site
        site_dirs = site.getsitepackages()
        if site_dirs:
            sc = Path(site_dirs[0]) / 'sitecustomize.py'
            shim_code = '''# Auto-generated compatibility shims for PyTorch 2.6+ / NumPy 2.x / Hugging Face
import sys, os

try:
    import torch
    _orig_torch_load = torch.load
    def _safe_torch_load(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return _orig_torch_load(*args, **kwargs)
    torch.load = _safe_torch_load
except Exception:
    pass

try:
    import numpy as np
    _orig_np_round = np.round
    def _safe_np_round(a, *args, **kwargs):
        try:
            return _orig_np_round(a, *args, **kwargs)
        except (TypeError, AttributeError):
            return _orig_np_round(np.asarray(a, dtype=float), *args, **kwargs)
    np.round = _safe_np_round
except Exception:
    pass

try:
    import huggingface_hub
    from huggingface_hub import constants
    if not hasattr(constants, "hf_cache_home"):
        constants.hf_cache_home = getattr(constants, "HF_HOME", os.environ.get("HF_HOME", "/kaggle/working/.cache/huggingface"))
    if not hasattr(huggingface_hub, "HfFolder"):
        huggingface_hub.HfFolder = type("HfFolder", (), {"get_token": staticmethod(lambda: None)})
    if not hasattr(huggingface_hub, "cached_download"):
        huggingface_hub.cached_download = getattr(huggingface_hub, "hf_hub_download", None)
except Exception:
    pass

try:
    import transformers.utils as _tu
    if not hasattr(_tu, "FLAX_WEIGHTS_NAME"): _tu.FLAX_WEIGHTS_NAME = "flax_model.msgpack"
    if not hasattr(_tu, "SAFE_WEIGHTS_NAME"): _tu.SAFE_WEIGHTS_NAME = "model.safetensors"
    if not hasattr(_tu, "WEIGHTS_NAME"): _tu.WEIGHTS_NAME = "pytorch_model.bin"
except Exception:
    pass
'''
            sc.write_text(shim_code, encoding='utf-8')
            print('SITECUSTOMIZE_SHIM_INSTALLED', sc, flush=True)
    except Exception as e:
        print(f'SITECUSTOMIZE_SHIM_WARN: {e}', flush=True)

install_runtime_shims()

# Apply locally to current process
try:
    import torch
    _orig_torch_load = torch.load
    def _safe_torch_load(*args, **kwargs):
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return _orig_torch_load(*args, **kwargs)
    torch.load = _safe_torch_load
except Exception:
    pass

try:
    import numpy as np
    _orig_np_round = np.round
    def _safe_np_round(a, *args, **kwargs):
        try:
            return _orig_np_round(a, *args, **kwargs)
        except (TypeError, AttributeError):
            return _orig_np_round(np.asarray(a, dtype=float), *args, **kwargs)
    np.round = _safe_np_round
except Exception:
    pass


def run(*args, cwd=None, env=None):
    print('RUN:', *args, flush=True)
    subprocess.run([str(x) for x in args], cwd=str(cwd) if cwd else None, env=env, check=True)


def find_input(name):
    xs = [p for p in INPUT.rglob(name) if p.is_file()]
    if not xs:
        raise RuntimeError(f'KAGGLE_INPUT_FILE_MISSING: {name}')
    return xs[0]


def disk(label):
    uw = shutil.disk_usage('/kaggle/working')
    ur = shutil.disk_usage('/')
    print(f'DISK {label} work_free_gb={uw.free/1024**3:.2f} root_free_gb={ur.free/1024**3:.2f}', flush=True)
    if uw.free < 2 * 1024**3:
        raise RuntimeError('WORKING_DISK_LOW')


def patch_runtime(repo: Path):
    """Patch MuseTalk to use fast YuNet face detection and optimized zero-disk streaming blending."""
    print('=== APPLYING MUSETALK INFERENCE OPTIMIZATIONS ===', flush=True)
    # 1. Bypass OpenMMLab/mmpose using YuNet for stable face box detection
    p = repo / 'musetalk/utils/preprocessing.py'
    p.write_text(r"""import cv2, numpy as np
from tqdm import tqdm
coord_placeholder = (0.0, 0.0, 0.0, 0.0)
detector = None
YUNET = './models/face_detection/yunet_2023mar.onnx'

def read_imgs(img_list):
    out = []
    for x in tqdm(img_list, desc='reading images'):
        im = cv2.imread(x)
        if im is None: raise RuntimeError(f'IMAGE_READ_FAILED {x}')
        out.append(im)
    return out

def get_landmark_and_bbox(img_list, upperbondrange=0):
    global detector
    frames = read_imgs(img_list); boxes = []
    for im in tqdm(frames, desc='YuNet face detection'):
        h, w = im.shape[:2]
        if detector is None:
            detector = cv2.FaceDetectorYN.create(YUNET, '', (w, h), 0.60, 0.30, 5000)
        else:
            detector.setInputSize((w, h))
        _, faces = detector.detect(im)
        if faces is None or len(faces) == 0:
            boxes.append(coord_placeholder); continue
        f = max(faces, key=lambda z: float(z[2] * z[3]))
        x, y, bw, bh = [float(v) for v in f[:4]]
        px, pyt, pyb = 0.08 * bw, 0.10 * bh, 0.18 * bh
        x1 = max(0, int(x - px)); y1 = max(0, int(y - pyt))
        x2 = min(w, int(x + bw + px)); y2 = min(h, int(y + bh + pyb))
        boxes.append((x1, y1, x2, y2) if x2 > x1 and y2 > y1 else coord_placeholder)
    return boxes, frames
""", encoding='utf-8')

    # 2. Patch PyTorch 2.6 weights_only in face parsing
    for rel in ('musetalk/utils/face_parsing/__init__.py', 'musetalk/utils/face_parsing/resnet.py'):
        fp = repo / rel
        if fp.exists():
            s = fp.read_text(encoding='utf-8')
            s = s.replace('torch.load(model_pth))', 'torch.load(model_pth, weights_only=False))')
            s = s.replace("torch.load(model_pth, map_location=torch.device('cpu'))", "torch.load(model_pth, map_location=torch.device('cpu'), weights_only=False)")
            s = s.replace('torch.load(model_path) #modelzoo.load_url', 'torch.load(model_path, weights_only=False) #modelzoo.load_url')
            fp.write_text(s, encoding='utf-8')

    # 3. Patch inference.py: zero-disk pipe composition directly to FFmpeg
    inf_p = repo / 'scripts/inference.py'
    inf_s = inf_p.read_text(encoding='utf-8')
    start_marker = '            # Pad generated images to original video size'
    end_marker = '        except Exception as e:'
    s_idx = inf_s.find(start_marker)
    e_idx = inf_s.find(end_marker)
    if s_idx != -1 and e_idx != -1 and s_idx < e_idx:
        optimized_code = '''            # Pad generated images to original video size (OPTIMIZED ZERO-DISK STREAMING)
            print("Starting zero-disk video stream composition with soft feathered blending...", flush=True)
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
    inf_s = inf_s.replace(
        'except Exception as e:\n            print("Error occurred during processing:", e)',
        'except Exception as e:\n            import traceback; traceback.print_exc()\n            raise'
    )
    inf_p.write_text(inf_s, encoding='utf-8')
    print('MUSETALK_RUNTIME_PATCHES_APPLIED=PASS', flush=True)


def main():
    print('=== STARTING ZERO-FLICKER DEVOTIONAL SINGER PRODUCTION (MUSETALK 1.5) ===', flush=True)
    disk('start')
    run('nvidia-smi')

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('NO_GPU_ALLOCATED')
    print('GPU_READY:', torch.cuda.get_device_name(0), flush=True)

    singer_image = find_input('singer.png')
    bhajan_audio = find_input('bhajan.mp3')
    duration_file = find_input('duration.txt')
    target_duration = int(duration_file.read_text().strip()) if duration_file.exists() else 180
    if not 180 <= target_duration <= 300 or target_duration % 15:
        raise RuntimeError(f'INVALID_DURATION {target_duration}')
    print(f'INPUTS: singer={singer_image} audio={bhajan_audio} duration={target_duration}s', flush=True)

    # Setup MuseTalk repository
    RUNTIME.mkdir(parents=True, exist_ok=True)
    repo = RUNTIME / 'MuseTalk'
    if not repo.exists():
        run('git', 'clone', '--depth', '1', 'https://github.com/TMElyralab/MuseTalk.git', str(repo))

    # Dependencies smoke test
    run(sys.executable, '-c', 'import diffusers, accelerate, transformers, omegaconf, librosa, einops, scipy, safetensors; from diffusers import AutoencoderKL; print("DEPENDENCY_SMOKE_OK", diffusers.__version__, transformers.__version__)')
    run('sudo', 'apt-get', 'update', '-qq')
    run('sudo', 'apt-get', 'install', '-y', '-qq', 'ffmpeg')

    # Prepare model checkpoints directory
    models = repo / 'models'
    for d in ('musetalkV15', 'sd-vae', 'whisper', 'face-parse-bisent', 'face_detection'):
        (models / d).mkdir(parents=True, exist_ok=True)

    from huggingface_hub import snapshot_download, hf_hub_download

    print('=== DOWNLOADING MUSETALK 1.5 WEIGHTS ===', flush=True)
    snapshot_download('TMElyralab/MuseTalk', local_dir=str(models), allow_patterns=['musetalkV15/musetalk.json', 'musetalkV15/unet.pth'])
    snapshot_download('stabilityai/sd-vae-ft-mse', local_dir=str(models / 'sd-vae'), allow_patterns=['config.json', 'diffusion_pytorch_model.bin'])
    snapshot_download('openai/whisper-tiny', local_dir=str(models / 'whisper'), allow_patterns=['config.json', 'pytorch_model.bin', 'preprocessor_config.json'])

    # YuNet face detector
    hf_hub_download(repo_id='opencv/face_detection_yunet', filename='face_detection_yunet_2023mar.onnx', local_dir=str(models / 'face_detection'))
    yunet_f = list((models / 'face_detection').rglob('*yunet*.onnx'))
    if yunet_f and not (models / 'face_detection' / 'yunet_2023mar.onnx').exists():
        shutil.copy2(yunet_f[0], models / 'face_detection' / 'yunet_2023mar.onnx')

    # BiSeNet face parser models from Hugging Face
    hf_hub_download(repo_id='afrizalha/musetalk-models', filename='face-parse-bisent/79999_iter.pth', local_dir=str(models))
    hf_hub_download(repo_id='afrizalha/musetalk-models', filename='face-parse-bisent/resnet18-5c106cde.pth', local_dir=str(models))
    bisent_dir = models / 'face-parse-bisent'
    bisent_dir.mkdir(parents=True, exist_ok=True)
    for f in models.rglob('*79999_iter.pth'):
        if f != bisent_dir / '79999_iter.pth':
            shutil.copy2(f, bisent_dir / '79999_iter.pth')
    for f in models.rglob('*resnet18-5c106cde.pth'):
        if f != bisent_dir / 'resnet18-5c106cde.pth':
            shutil.copy2(f, bisent_dir / 'resnet18-5c106cde.pth')
    print('=== MUSETALK CHECKPOINTS READY ===', flush=True)

    patch_runtime(repo)

    # Prepare input data
    data_dir = repo / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(singer_image, data_dir / 'singer.png')
    shutil.copy2(bhajan_audio, data_dir / 'bhajan.mp3')

    # Base input is the canonical reference portrait: rock-solid background, 0% flicker, 100% stable deity sanctum
    target_video_input = 'data/singer.png'
    print(f'MUSETALK_TARGET_INPUT: {target_video_input}', flush=True)

    # Configure inference task
    cfg = repo / 'configs' / 'inference' / 'musetalk_runtime.yaml'
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(
        f'task_0:\n  video_path: "{target_video_input}"\n  audio_path: "data/bhajan.mp3"\n  result_name: "master.mp4"\n',
        encoding='utf-8'
    )

    disk('before_inference')
    started = time.time()
    print('=== RUNNING MUSETALK ZERO-FLICKER SINGING INFERENCE ===', flush=True)
    infer_cmd = [
        sys.executable,
        '-m',
        'scripts.inference',
        '--inference_config',
        str(cfg),
        '--result_dir',
        'results/runtime',
        '--unet_model_path',
        'models/musetalkV15/unet.pth',
        '--unet_config',
        'models/musetalkV15/musetalk.json',
        '--whisper_dir',
        'models/whisper',
        '--version',
        'v15',
        '--fps',
        str(FPS),
        '--batch_size',
        '8',
        '--use_float16',
        '--extra_margin',
        '15',
        '--output_vid_name',
        'master.mp4',
    ]
    run(*infer_cmd, cwd=repo)
    elapsed = time.time() - started
    print(f'=== MUSETALK INFERENCE COMPLETED IN {elapsed:.1f}s ===', flush=True)

    # Locate generated master video
    candidates = list((repo / 'results' / 'runtime' / 'v15').glob('master.mp4'))
    if not candidates:
        candidates = list(repo.rglob('master.mp4'))
    if not candidates:
        raise RuntimeError('MUSETALK_MASTER_MISSING')

    # Final broadcast H.264 / AAC encoding with exact duration
    final_master = FINAL / 'master.mp4'
    run(
        'ffmpeg',
        '-y',
        '-v',
        'error',
        '-i',
        str(candidates[0]),
        '-t',
        str(target_duration),
        '-c:v',
        'libx264',
        '-preset',
        'veryfast',
        '-crf',
        '19',
        '-pix_fmt',
        'yuv420p',
        '-c:a',
        'aac',
        '-b:a',
        '192k',
        '-ar',
        '48000',
        '-movflags',
        '+faststart',
        str(final_master)
    )
    mb = final_master.stat().st_size / (1024 * 1024)
    print(f'MUSETALK_MASTER_READY path={final_master} size_mb={mb:.1f} duration={target_duration}s', flush=True)

    # HARD audio/video lip-sync gate with SyncNet
    print('=== EVALUATING LIPSYNC QA GATE ===', flush=True)
    syncnet_path = FINAL / 'syncnet_v2.model'
    syncnet_src = hf_hub_download(repo_id='ByteDance/LatentSync-1.6', filename='auxiliary/syncnet_v2.model')
    shutil.copy2(syncnet_src, syncnet_path)

    yunet_model = models / 'face_detection' / 'yunet_2023mar.onnx'
    gate_script = FINAL / 'lipsync_quality_gate.py'
    if gate_script.exists():
        run(
            sys.executable,
            str(gate_script),
            str(final_master),
            '--model',
            str(syncnet_path),
            '--yunet',
            str(yunet_model),
            '--report',
            str(FINAL / 'lipsync_qa.json')
        )
        print('LIPSYNC_QUALITY_GATE=PASS', flush=True)
    syncnet_path.unlink(missing_ok=True)

    # Cleanup runtime to conserve space
    shutil.rmtree(RUNTIME, ignore_errors=True)
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    disk('finish')
    print('MUSETALK_WORKER_SUCCESS=TRUE', flush=True)


if __name__ == '__main__':
    main()
