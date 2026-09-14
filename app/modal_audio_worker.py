from __future__ import annotations
import os, subprocess, tempfile
from pathlib import Path
import modal

app = modal.App("hindibhajans-audio")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libsndfile1")
)

LYRICS = """[Intro]
श्री राम... श्री राम... जय जय राम...

[Verse 1]
मन में बसो रघुनंदन, चरणों में मेरा ध्यान
राम नाम की ज्योति जले, रोशन हो हर प्राण

[Chorus]
श्री राम जय राम, जय जय राम
मेरे मन के दीप में, बसते श्री राम

[Verse 2]
दुख की घड़ी में साथ दो, हे दीनदयाल भगवान
तेरा नाम ही आसरा, तेरा नाम ही सम्मान

[Chorus]
श्री राम जय राम, जय जय राम
मेरे मन के दीप में, बसते श्री राम

[Verse 3]
अयोध्या के राजकुमार, करुणा के भंडार
तेरे चरणों में मिल जाए, जीवन का सच्चा सार

[Final Chorus]
श्री राम जय राम, जय जय राम
मेरे मन के दीप में, बसते श्री राम
श्री राम जय राम, जय जय राम
जय जय राम... जय जय राम...

[Outro]
श्री राम... जय राम... जय जय राम..."""

PROMPT = "Modern high-energy Hindi devotional bhajan, 128 BPM, 4/4, powerful expressive Hindi male lead vocal clearly singing every lyric with natural emotion and clean pronunciation, catchy devotional melody, memorable chorus, energetic polished arrangement with dhol/dholak, tabla, temple bells, bansuri, harmonium texture and modern synth production. NOT meditation, NOT ambient, NOT spoken narration, NOT humming, NOT instrumental-only."


def run(cmd, **kwargs):
    subprocess.run([str(x) for x in cmd], check=True, **kwargs)


@app.function(image=image, gpu="T4", timeout=3600)
def generate(seconds: int = 180) -> bytes:
    if seconds < 180 or seconds > 300 or seconds % 15:
        raise ValueError("seconds must be 180-300 and divisible by 15")

    work = Path(tempfile.mkdtemp(prefix="bhajan-audio-"))
    repo = work / "ACE-Step-1.5"
    run(["git", "clone", "--depth", "1", "https://github.com/ACE-Step/ACE-Step-1.5.git", repo])

    # ACE-Step 1.5 now declares nano-vllm as a local uv source. Plain pip
    # cannot resolve that dependency and fails with "No matching distribution
    # found for nano-vllm". We use the project's requirements for the Linux
    # CUDA build, but deliberately omit flash-attn/triton because this worker
    # uses the supported PyTorch ("pt") 5Hz-LM backend and does not need the
    # nano-vLLM stack.
    requirements = repo / "modal-runtime-requirements.txt"
    lines = (repo / "requirements.txt").read_text().splitlines()
    filtered = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("flash-attn") or stripped.startswith("triton"):
            continue
        filtered.append(line)
    requirements.write_text("\n".join(filtered) + "\n")
    run(["python", "-m", "pip", "install", "-q", "-r", requirements])
    # Install the ACE-Step package itself without dependency resolution; the
    # dependency set above is already installed and nano-vllm is intentionally
    # not required for backend="pt".
    run(["python", "-m", "pip", "install", "-q", "--no-deps", "-e", repo])

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["ACESTEP_SAVE_MEMORY"] = "1"
    import sys
    sys.path.insert(0, str(repo))
    import torch

    from acestep.handler import AceStepHandler
    from acestep.llm_inference import LLMHandler
    from acestep.inference import GenerationConfig, GenerationParams, generate_music

    if not torch.cuda.is_available():
        raise RuntimeError("MODAL_AUDIO_NO_CUDA")

    dit = AceStepHandler()
    dit.initialize_service(
        project_root=str(repo),
        config_path="acestep-v15-turbo",
        device="cuda",
        offload_to_cpu=True,
    )
    llm = LLMHandler()
    llm.initialize(
        checkpoint_dir=str(repo / "checkpoints"),
        lm_model_path="acestep-5Hz-lm-0.6B",
        backend="pt",
        device="cuda",
    )

    params = GenerationParams(
        task_type="text2music",
        caption=PROMPT,
        lyrics=LYRICS,
        bpm=128,
        keyscale="C Major",
        timesignature="4/4",
        vocal_language="hi",
        duration=float(seconds),
        thinking=False,
        use_cot_metas=False,
        use_cot_caption=False,
        use_cot_language=False,
        use_constrained_decoding=True,
        inference_steps=8,
        guidance_scale=1.0,
        seed=-1,
        shift=3.0,
        infer_method="ode",
        sampler_mode="euler",
        dcw_enabled=True,
        dcw_mode="double",
        dcw_scaler=0.05,
        dcw_high_scaler=0.02,
        dcw_wavelet="haar",
    )
    result = generate_music(
        dit_handler=dit,
        llm_handler=llm,
        params=params,
        config=GenerationConfig(batch_size=1, use_random_seed=True, audio_format="wav"),
        save_dir=str(work / "out"),
    )
    if not result.success or not result.audios:
        raise RuntimeError(f"MODAL_AUDIO_GENERATION_FAILED:{result.error or result.status_message}")

    source = Path(result.audios[0]["path"])
    out = work / "bhajan_source.mp3"
    run([
        "ffmpeg", "-y", "-v", "error", "-i", source,
        "-af", "loudnorm=I=-9:TP=-1.0:LRA=7",
        "-ar", "48000", "-ac", "2", "-c:a", "libmp3lame", "-b:a", "320k", out,
    ])
    return out.read_bytes()


@app.local_entrypoint()
def main(seconds: int = 180):
    data = generate.remote(seconds)
    Path("output").mkdir(exist_ok=True)
    Path("output/bhajan_source.mp3").write_bytes(data)
    print("MODAL_AUDIO_READY", len(data), flush=True)
