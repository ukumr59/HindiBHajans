from __future__ import annotations

import io
import os
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import modal

app = modal.App("hindibhajans-production")
MODEL_VOLUME = modal.Volume.from_name("hindibhajans-model-cache", create_if_missing=True)
MODEL_DIR = "/models"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libsndfile1")
    .pip_install(
        "torch", "torchvision", "torchaudio",
        "huggingface_hub", "opencv-python-headless", "numpy",
    )
)

PROMPT = """Modern high-energy Hindi devotional bhajan made like a current YouTube DJ devotional song, 128 BPM, 4/4, loud polished commercial stereo production, powerful expressive Hindi male lead vocal clearly singing every lyric with natural emotion and clean pronunciation, catchy devotional melody, huge memorable chorus, energetic EDM arrangement, punchy four-on-the-floor kick, deep controlled sub bass, modern synth bass, bright synth leads, wide pads, electronic percussion, claps, dhol and dholak layered with tabla, cinematic risers, tasteful temple bells, bansuri accents, harmonium texture, short instrumental intro, strong verse build, massive chorus/drop, rhythmic instrumental break, final chorus with layered backing vocals, professional YouTube/radio loudness and DJ playback energy. NOT meditation music, NOT sleepy, NOT ambient, NOT acoustic-only, NOT spoken narration, NOT humming, NOT a cappella, NOT instrumental-only."""
LYRICS = """[Intro]\nश्री राम... श्री राम... जय जय राम...\n\n[Verse 1]\nमन में बसो रघुनंदन, चरणों में मेरा ध्यान\nराम नाम की ज्योति जले, रोशन हो हर प्राण\n\n[Pre-Chorus]\nतेरे नाम की धुन बजे, हर धड़कन में आज\nतेरी कृपा से खिल उठे, जीवन का हर राज\n\n[Chorus]\nश्री राम जय राम, जय जय राम\nमेरे मन के दीप में, बसते श्री राम\n\n[Instrumental Break]\n\n[Verse 2]\nदुख की घड़ी में साथ दो, हे दीनदयाल भगवान\nतेरा नाम ही आसरा, तेरा नाम ही सम्मान\n\n[Pre-Chorus]\nतेरी राह में चल पड़ूँ, मन में लेकर विश्वास\nराम नाम की शक्ति से, मिट जाए हर त्रास\n\n[Chorus]\nश्री राम जय राम, जय जय राम\nमेरे मन के दीप में, बसते श्री राम\n\n[Verse 3]\nअयोध्या के राजकुमार, करुणा के भंडार\nतेरे चरणों में मिल जाए, जीवन का सच्चा सार\n\n[Build]\nजय श्री राम की गूंज उठे, नभ से धरती तक\nढोल बजे और शंख बजे, प्रेम बहे हर पल\n\n[Final Chorus]\nश्री राम जय राम, जय जय राम\nमेरे मन के दीप में, बसते श्री राम\nश्री राम जय राम, जय जय राम\nजय जय राम... जय जय राम...\n\n[Outro]\nश्री राम... जय राम... जय जय राम..."""


def run(*args, cwd=None):
    print("RUN:", " ".join(map(str, args)), flush=True)
    subprocess.run([str(x) for x in args], cwd=str(cwd) if cwd else None, check=True)


def first_mp4(folder: Path) -> Path:
    xs = sorted(folder.rglob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not xs:
        raise RuntimeError(f"ECHOMIMIC_RETURNED_NO_MP4:{folder}")
    return xs[0]


def prepare_audio(repo: Path, work: Path, seconds: int) -> Path:
    run("python", "-m", "pip", "install", "-q", "-e", ".", cwd=repo)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("ACESTEP_SAVE_MEMORY", "1")
    import sys
    sys.path.insert(0, str(repo))
    import torch
    from acestep.handler import AceStepHandler
    from acestep.llm_inference import LLMHandler
    from acestep.inference import GenerationConfig, GenerationParams, generate_music
    if not torch.cuda.is_available():
        raise RuntimeError("MODAL_ACE_FATAL: CUDA GPU unavailable")
    print("MODAL_GPU=", torch.cuda.get_device_name(0), flush=True)
    dit = AceStepHandler()
    dit.initialize_service(project_root=str(repo), config_path="acestep-v15-turbo", device="cuda", offload_to_cpu=True)
    llm = LLMHandler()
    llm.initialize(checkpoint_dir=str(repo / "checkpoints"), lm_model_path="acestep-5Hz-lm-0.6B", backend="pt", device="cuda")
    params = GenerationParams(
        task_type="text2music", caption=PROMPT, lyrics=LYRICS, bpm=128,
        keyscale="C Major", timesignature="4/4", vocal_language="hi", duration=float(seconds),
        thinking=False, use_cot_metas=False, use_cot_caption=False, use_cot_language=False,
        use_constrained_decoding=True, inference_steps=8, guidance_scale=1.0, seed=-1,
        shift=3.0, infer_method="ode", sampler_mode="euler", dcw_enabled=True,
        dcw_mode="double", dcw_scaler=0.05, dcw_high_scaler=0.02, dcw_wavelet="haar",
    )
    result = generate_music(
        dit_handler=dit, llm_handler=llm, params=params,
        config=GenerationConfig(batch_size=1, use_random_seed=True, audio_format="wav"),
        save_dir=str(work / "ace_output"),
    )
    if not result.success or not result.audios:
        raise RuntimeError(f"MODAL_ACE_FATAL:{result.error or result.status_message}")
    source = Path(result.audios[0].get("path", ""))
    if not source.exists():
        raise RuntimeError("MODAL_ACE_FATAL:generated audio missing")
    out = work / "bhajan_source.mp3"
    run("ffmpeg", "-y", "-v", "error", "-i", str(source), "-af", "loudnorm=I=-9:TP=-1.0:LRA=7", "-ar", "48000", "-ac", "2", "-c:a", "libmp3lame", "-b:a", "320k", str(out))
    if out.stat().st_size < 100_000:
        raise RuntimeError("MODAL_ACE_FATAL:audio output suspiciously small")
    return out


def prepare_video(work: Path, seconds: int, image: Path, audio: Path) -> Path:
    repo = work / "echomimic_v3"
    if not repo.exists():
        run("git", "clone", "--depth", "1", "https://github.com/antgroup/echomimic_v3.git", str(repo))
    run("python", "-m", "pip", "install", "-q", "-r", str(repo / "requirements.txt"))
    run("python", "-m", "pip", "install", "-q", "huggingface_hub", "pyloudnorm", "eva-decord")
    from huggingface_hub import snapshot_download
    base = Path(MODEL_DIR) / "Wan2.1-Fun-V1.1-1.3B-InP"
    wav = Path(MODEL_DIR) / "chinese-wav2vec2-base"
    flash = Path(MODEL_DIR) / "echomimicv3-flash-pro"
    snapshot_download("alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP", local_dir=str(base))
    snapshot_download("TencentGameMate/chinese-wav2vec2-base", local_dir=str(wav))
    snapshot_download("BadToBest/EchoMimicV3", local_dir=str(flash), allow_patterns=["echomimicv3-flash-pro/*"])
    fps, frames = 25, 81
    seg_seconds = frames / fps
    norm = work / "audio16k.wav"
    run("ffmpeg", "-y", "-v", "error", "-i", str(audio), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(norm))
    seg = work / "segments"; seg.mkdir()
    n = int((seconds + seg_seconds - 1) // seg_seconds)
    for i in range(n):
        start = i * seg_seconds
        remain = min(seg_seconds, seconds - start)
        if remain < 0.5: break
        wav_i = seg / f"audio_{i:04d}.wav"
        run("ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}", "-i", str(norm), "-t", f"{remain:.3f}", "-ar", "16000", "-ac", "1", str(wav_i))
        raw = seg / f"raw_{i:04d}"; raw.mkdir()
        run("python", str(repo / "infer_flash.py"), "--image_path", str(image), "--audio_path", str(wav_i),
            "--prompt", "A single Indian devotional singer performing a Hindi bhajan in traditional Indian clothing before the specified Hindu deity in a serene temple setting; only the same singer is visible; natural singing mouth movement synchronized to the audio; subtle expressive head and upper-body motion; stable identity.",
            "--num_inference_steps", "8", "--config_path", str(repo / "config/config.yaml"), "--model_name", str(base), "--ckpt_idx", "50000",
            "--transformer_path", str(flash / "echomimicv3-flash-pro/transformer/diffusion_pytorch_model.safetensors"), "--save_path", str(raw), "--wav2vec_model_dir", str(wav),
            "--sampler_name", "Flow_Unipc", "--video_length", str(frames), "--guidance_scale", "5.0", "--audio_guidance_scale", "2.5", "--audio_scale", "1.0",
            "--neg_scale", "1.0", "--neg_steps", "0", "--seed", str(4300 + i), "--enable_teacache", "--teacache_threshold", "0.1",
            "--num_skip_start_steps", "5", "--weight_dtype", "float16", "--sample_size", "768", "768", "--fps", str(fps),
            "--negative_prompt", "blurry, distorted face, identity drift, extra person, duplicate person, malformed hands, fused fingers, deformed mouth, jitter, flicker, camera cut, text, watermark")
        raw_mp4 = first_mp4(raw)
        silent = seg / f"video_{i:04d}.mp4"
        run("ffmpeg", "-y", "-v", "error", "-i", str(raw_mp4), "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", str(silent))
    files = sorted(seg.glob("video_*.mp4"))
    if not files: raise RuntimeError("NO_ECHOMIMIC_SEGMENTS")
    concat = seg / "concat.txt"; concat.write_text("".join(f"file '{p.resolve()}'\n" for p in files))
    visual = work / "visual.mp4"
    run("ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(concat), "-c", "copy", str(visual))
    final = work / "master.mp4"
    run("ffmpeg", "-y", "-v", "error", "-i", str(visual), "-i", str(audio), "-map", "0:v:0", "-map", "1:a:0", "-t", str(seconds), "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-movflags", "+faststart", str(final))
    if final.stat().st_size < 500_000: raise RuntimeError("MASTER_NOT_CREATED")
    return final


@app.function(image=image, gpu="L4", timeout=60 * 60 * 2, volumes={MODEL_DIR: MODEL_VOLUME})
def produce(seconds: int = 180) -> bytes:
    if seconds < 180 or seconds > 300 or seconds % 15:
        raise ValueError("seconds must be 180-300 and divisible by 15")
    work = Path(tempfile.mkdtemp(prefix="bhajan-modal-"))
    ref = Path("/assets/singer_image.png")
    # The reference image is shipped into the function by modal.Image.add_local_file
    if not ref.exists():
        raise RuntimeError("IDENTITY_SOURCE_MISSING")
    audio_repo = work / "ACE-Step-1.5"
    run("git", "clone", "--depth", "1", "https://github.com/ACE-Step/ACE-Step-1.5.git", str(audio_repo))
    audio = prepare_audio(audio_repo, work, seconds)
    master = prepare_video(work, seconds, ref, audio)
    bundle = io.BytesIO()
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(master, "master.mp4")
        z.write(audio, "bhajan_source.mp3")
    return bundle.getvalue()


# Modal's image builder copies the canonical identity reference into the remote container.
image = image.add_local_file("assets/singer_image.png", "/assets/singer_image.png")
