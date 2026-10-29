from __future__ import annotations
import base64
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

FINAL = Path('/kaggle/working')
RUNTIME = FINAL / '.runtime' / 'echomimic'
INPUT = Path('/kaggle/input')
FPS = 24

# Redirect HuggingFace & Torch caches to 20GB working partition
os.environ['HF_HOME'] = '/tmp/.cache/huggingface'
os.environ['TORCH_HOME'] = '/tmp/.cache/torch'
os.environ['PYTHONUNBUFFERED'] = '1'

# Compatibility shim for diffusers on modern huggingface_hub & transformers
try:
    import huggingface_hub
    from huggingface_hub import constants
    if not hasattr(constants, 'hf_cache_home'):
        constants.hf_cache_home = getattr(constants, 'HF_HOME', '/tmp/.cache/huggingface')
    if not hasattr(huggingface_hub, 'HfFolder'):
        huggingface_hub.HfFolder = type('HfFolder', (), {'get_token': staticmethod(lambda: None)})
    if not hasattr(huggingface_hub, 'cached_download'):
        huggingface_hub.cached_download = getattr(huggingface_hub, 'hf_hub_download', None)
except Exception:
    pass

try:
    import transformers
    import transformers.utils as _tu
    if not hasattr(_tu, 'FLAX_WEIGHTS_NAME'):
        _tu.FLAX_WEIGHTS_NAME = 'flax_model.msgpack'
    if not hasattr(_tu, 'SAFE_WEIGHTS_NAME'):
        _tu.SAFE_WEIGHTS_NAME = 'model.safetensors'
    if not hasattr(_tu, 'WEIGHTS_NAME'):
        _tu.WEIGHTS_NAME = 'pytorch_model.bin'
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


def install_dependencies():
    print('=== STEP 1: INSTALLING ECHOMIMIC DEPENDENCIES ===', flush=True)
    # EchoMimic runtime packages: all have pre-compiled wheels.
    # Note: decord, librosa, and soundfile are NOT needed by EchoMimic.
    # moviepy is bypassed via native ffmpeg below.
    run(
        sys.executable,
        '-m',
        'pip',
        'install',
        '-q',
        '--prefer-binary',
        'diffusers==0.24.0',
        'omegaconf==2.3.0',
        'einops',
        'av',
        'ffmpeg-python==0.2.0',
    )
    # facenet-pytorch installs MTCNN without compiling Pillow or touching PyTorch
    run(sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', 'facenet-pytorch')

    # Patch diffusers 0.24.0 pipeline_utils.py directly for compatibility with modern transformers
    try:
        import diffusers
        p = Path(diffusers.__file__).parent / 'pipelines' / 'pipeline_utils.py'
        if p.exists():
            t = p.read_text(encoding='utf-8')
            old_import = 'from transformers.utils import FLAX_WEIGHTS_NAME as TRANSFORMERS_FLAX_WEIGHTS_NAME'
            new_import = (
                'try:\n'
                '        from transformers.utils import FLAX_WEIGHTS_NAME as TRANSFORMERS_FLAX_WEIGHTS_NAME\n'
                '    except ImportError:\n'
                '        TRANSFORMERS_FLAX_WEIGHTS_NAME = "flax_model.msgpack"'
            )
            if old_import in t:
                t = t.replace(old_import, new_import, 1)
                p.write_text(t, encoding='utf-8')
                print('DIFFUSERS_FLAX_WEIGHTS_PATCH=APPLIED', flush=True)
    except Exception as e:
        print(f'DIFFUSERS_FLAX_WEIGHTS_PATCH_WARN: {e}', flush=True)

    print('=== ECHOMIMIC DEPENDENCIES INSTALLED ===', flush=True)


def setup_echomimic_repo(repo: Path):
    print('=== STEP 2: CLONING ECHOMIMIC REPOSITORY ===', flush=True)
    if repo.exists():
        shutil.rmtree(repo, ignore_errors=True)
    repo.parent.mkdir(parents=True, exist_ok=True)
    run('git', 'clone', '--depth', '1', 'https://github.com/BadToBest/EchoMimic.git', str(repo))

    # Apply CPU-offload patch for VAE decode to avoid CUDA OOM on long sequences
    pipe_file = repo / 'src' / 'pipelines' / 'pipeline_echo_mimic_acc.py'
    if pipe_file.exists():
        src_text = pipe_file.read_text(encoding='utf-8')
        target = 'video.append(self.vae.decode(latents[frame_idx : frame_idx + 1]).sample)'
        replacement = 'video.append(self.vae.decode(latents[frame_idx : frame_idx + 1]).sample.cpu())'
        if target in src_text:
            src_text = src_text.replace(target, replacement, 1)
            src_text = src_text.replace('video = video.cpu().float().numpy()', 'video = video.float().numpy()', 1)
            pipe_file.write_text(src_text, encoding='utf-8')
            print('ECHOMIMIC_CPU_DECODE_PATCH=APPLIED', flush=True)

    # Patch infer_audio2vid_acc.py to bypass moviepy and inject hf_cache_home compatibility shim
    infer_file = repo / 'infer_audio2vid_acc.py'
    if infer_file.exists():
        txt = infer_file.read_text(encoding='utf-8')
        compat_shim = (
            "import os, huggingface_hub\n"
            "from huggingface_hub import constants\n"
            "if not hasattr(constants, 'hf_cache_home'):\n"
            "    constants.hf_cache_home = getattr(constants, 'HF_HOME', os.path.expanduser('~/.cache/huggingface'))\n"
            "if not hasattr(huggingface_hub, 'HfFolder'):\n"
            "    huggingface_hub.HfFolder = type('HfFolder', (), {'get_token': staticmethod(lambda: None)})\n"
            "if not hasattr(huggingface_hub, 'cached_download'):\n"
            "    huggingface_hub.cached_download = getattr(huggingface_hub, 'hf_hub_download', None)\n"
            "try:\n"
            "    import transformers\n"
            "    import transformers.utils as _tu\n"
            "    if not hasattr(_tu, 'FLAX_WEIGHTS_NAME'): _tu.FLAX_WEIGHTS_NAME = 'flax_model.msgpack'\n"
            "    if not hasattr(_tu, 'SAFE_WEIGHTS_NAME'): _tu.SAFE_WEIGHTS_NAME = 'model.safetensors'\n"
            "    if not hasattr(_tu, 'WEIGHTS_NAME'): _tu.WEIGHTS_NAME = 'pytorch_model.bin'\n"
            "except Exception:\n"
            "    pass\n"
            "try:\n"
            "    import numpy as np\n"
            "    _orig_np_round = np.round\n"
            "    def _safe_np_round(a, *args, **kwargs):\n"
            "        try:\n"
            "            return _orig_np_round(a, *args, **kwargs)\n"
            "        except (TypeError, AttributeError):\n"
            "            return _orig_np_round(np.asarray(a, dtype=float), *args, **kwargs)\n"
            "    np.round = _safe_np_round\n"
            "except Exception:\n"
            "    pass\n"
        )
        txt = compat_shim + txt
        txt = txt.replace('from moviepy.editor import VideoFileClip, AudioFileClip', '# moviepy bypassed')
        target_block = (
            '            video_clip = VideoFileClip(f"{save_dir}/{ref_name}_{audio_name}_{args.H}x{args.W}_{int(args.cfg)}_{time_str}.mp4")\n'
            '            audio_clip = AudioFileClip(audio_path)\n'
            '            video_clip = video_clip.set_audio(audio_clip)\n'
            '            video_clip.write_videofile(f"{save_dir}/{ref_name}_{audio_name}_{args.H}x{args.W}_{int(args.cfg)}_{time_str}_withaudio.mp4", codec="libx264", audio_codec="aac")'
        )
        replacement_block = (
            '            raw_mp4 = f"{save_dir}/{ref_name}_{audio_name}_{args.H}x{args.W}_{int(args.cfg)}_{time_str}.mp4"\n'
            '            mux_mp4 = f"{save_dir}/{ref_name}_{audio_name}_{args.H}x{args.W}_{int(args.cfg)}_{time_str}_withaudio.mp4"\n'
            '            subprocess.run([\'ffmpeg\', \'-y\', \'-v\', \'error\', \'-i\', raw_mp4, \'-i\', audio_path, \'-c:v\', \'copy\', \'-c:a\', \'aac\', mux_mp4], check=True)'
        )
        txt = txt.replace('\r\n', '\n').replace(target_block.replace('\r\n', '\n'), replacement_block)
        infer_file.write_text(txt, encoding='utf-8')
        print('ECHOMIMIC_MOVIEPY_BYPASS_PATCH=APPLIED', flush=True)
        # Patch numpy 2.x round compatibility in infer_audio2vid_acc.py
        txt = txt.replace("xyxy = np.round(xyxy).astype('int')", "xyxy = [int(round(float(x))) for x in select_bbox[:4]]")
        txt = txt.replace('xyxy = np.round(xyxy).astype(int)', "xyxy = [int(round(float(x))) for x in select_bbox[:4]]")
        infer_file.write_text(txt, encoding='utf-8')
        print('ECHOMIMIC_NUMPY_ROUND_PATCH=APPLIED', flush=True)

    print('=== STEP 3: DOWNLOADING ECHOMIMIC WEIGHTS ===', flush=True)
    weights_dir = repo / 'pretrained_weights'
    weights_dir.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id='BadToBest/EchoMimic',
        local_dir=str(weights_dir),
        allow_patterns=[
            'audio_processor/*',
            'denoising_unet_acc.pth',
            'motion_module_acc.pth',
            'face_locator.pth',
            'reference_unet.pth',
            'sd-image-variations-diffusers/*',
            'sd-vae-ft-mse/*',
        ],
        local_dir_use_symlinks=False,
    )
    print('=== ECHOMIMIC WEIGHTS DOWNLOADED ===', flush=True)


def main():
    print('=== STARTING AUTONOMOUS ECHOMIMIC SINGER PRODUCTION ===', flush=True)
    disk('start')
    run('nvidia-smi')

    singer_image = find_input('singer.png')
    bhajan_audio = find_input('bhajan.mp3')
    duration_file = find_input('duration.txt')
    target_duration = int(duration_file.read_text().strip()) if duration_file.exists() else 180
    print(f'INPUTS: singer={singer_image} audio={bhajan_audio} target_duration={target_duration}s', flush=True)

    install_dependencies()
    repo = RUNTIME
    setup_echomimic_repo(repo)

    # Prepare input data inside EchoMimic
    data_dir = repo / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(singer_image, data_dir / 'singer.png')

    # Convert audio to 16kHz mono WAV for EchoMimic Whisper processor
    wav_path = data_dir / 'bhajan.wav'
    run('ffmpeg', '-y', '-v', 'error', '-i', str(bhajan_audio), '-ar', '16000', '-ac', '1', str(wav_path))

    # Configure accelerated inference
    anim_config = """## dependency models
pretrained_base_model_path: "./pretrained_weights/sd-image-variations-diffusers/"
pretrained_vae_path: "./pretrained_weights/sd-vae-ft-mse/"
audio_model_path: "./pretrained_weights/audio_processor/whisper_tiny.pt"

## echo mimic checkpoint (Accelerated versions)
denoising_unet_path: "./pretrained_weights/denoising_unet_acc.pth"
reference_unet_path: "./pretrained_weights/reference_unet.pth"
face_locator_path: "./pretrained_weights/face_locator.pth"
motion_module_path: "./pretrained_weights/motion_module_acc.pth"

## denoise model configs
inference_config: "./configs/inference/inference_v2.yaml"
weight_dtype: 'fp16'

## test cases
test_cases:
  "data/singer.png":
    - "data/bhajan.wav"
"""
    configs_dir = repo / 'configs' / 'prompts'
    configs_dir.mkdir(parents=True, exist_ok=True)
    (configs_dir / 'animation_acc.yaml').write_text(anim_config, encoding='utf-8')
    (configs_dir / 'animation.yaml').write_text(anim_config, encoding='utf-8')

    disk('before_inference')
    started = time.time()
    print('=== STEP 4: RUNNING ECHOMIMIC SINGER INFERENCE ===', flush=True)
    infer_env = dict(os.environ)
    infer_env['CUDA_VISIBLE_DEVICES'] = '0'
    target_frames = target_duration * FPS
    infer_cmd = [
        sys.executable,
        '-u',
        'infer_audio2vid_acc.py',
        '--config',
        './configs/prompts/animation_acc.yaml',
        '-W',
        '512',
        '-H',
        '512',
        '-L',
        str(target_frames),
        '--steps',
        '6',
        '--fps',
        str(FPS),
        '--cfg',
        '1.0',
    ]
    run(*infer_cmd, cwd=repo, env=infer_env)
    elapsed = time.time() - started
    print(f'=== ECHOMIMIC INFERENCE COMPLETED IN {elapsed:.1f}s ===', flush=True)

    # Locate generated output video
    candidates = list((repo / 'output').rglob('*.mp4'))
    if not candidates:
        candidates = [p for p in repo.rglob('*.mp4') if p.stat().st_size > 100_000 and 'temp' not in p.name.lower()]
    if not candidates:
        raise RuntimeError('ECHOMIMIC_OUTPUT_VIDEO_NOT_FOUND')

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    raw_video = candidates[0]
    print(f'ECHOMIMIC_RAW_VIDEO={raw_video} size={raw_video.stat().st_size}', flush=True)

    # Remux video with the pristine master audio track and target duration
    final_master = FINAL / 'master.mp4'
    print('=== STEP 5: MASTER VIDEO ENCODING ===', flush=True)
    run(
        'ffmpeg',
        '-y',
        '-v',
        'error',
        '-i',
        str(raw_video),
        '-i',
        str(bhajan_audio),
        '-map',
        '0:v:0',
        '-map',
        '1:a:0',
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
        '-shortest',
        str(final_master),
    )
    final_size_mb = final_master.stat().st_size / (1024 * 1024)
    print(f'ECHOMIMIC_MASTER_READY path={final_master} size_mb={final_size_mb:.1f} duration={target_duration}s', flush=True)

    # Evaluate Lip-Sync QA Gate with SyncNet
    print('=== STEP 6: LIPSYNC QA VALIDATION ===', flush=True)
    from huggingface_hub import hf_hub_download

    syncnet_path = FINAL / 'syncnet_v2.model'
    syncnet_src = hf_hub_download(repo_id='ByteDance/LatentSync-1.6', filename='auxiliary/syncnet_v2.model')
    shutil.copy2(syncnet_src, syncnet_path)

    yunet_path = FINAL / 'yunet_2023mar.onnx'
    if not yunet_path.exists():
        import urllib.request

        urllib.request.urlretrieve(
            'https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx',
            str(yunet_path),
        )

    gate_script = FINAL / 'lipsync_quality_gate.py'
    if gate_script.exists():
        run(
            sys.executable,
            str(gate_script),
            str(final_master),
            '--model',
            str(syncnet_path),
            '--yunet',
            str(yunet_path),
            '--report',
            str(FINAL / 'lipsync_qa.json'),
        )
        print('LIPSYNC_QUALITY_GATE=PASS', flush=True)
    syncnet_path.unlink(missing_ok=True)
    yunet_path.unlink(missing_ok=True)

    # Cleanup bulky runtime and caches before kernel terminates
    shutil.rmtree(RUNTIME, ignore_errors=True)
    shutil.rmtree(FINAL / '.cache', ignore_errors=True)
    disk('finish')
    print('ECHOMIMIC_WORKER_SUCCESS=TRUE', flush=True)


if __name__ == '__main__':
    main()
