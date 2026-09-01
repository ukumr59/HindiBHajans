from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

WORK = Path('/kaggle/working')
REPO = WORK / 'HindiBHajans'
OUTPUT = WORK / 'bhajan_aabha_exact_identity.mp4'
AUDIO = WORK / 'bhajan_source.mp3'
MANIFEST = WORK / 'manifest.json'

REPO_URL = 'https://github.com/ukumr59/HindiBHajans.git'
REQUEST = {
    'caption': (
        'Modern high-energy Hindi devotional bhajan, 128 BPM, 4/4, polished commercial production, '
        'powerful expressive Hindi male lead vocal clearly singing every lyric, catchy devotional melody, '
        'energetic electronic arrangement, punchy kick, controlled bass, synth pads, dhol, dholak, tabla, '
        'temple bells, bansuri and harmonium accents, strong chorus, professional loud master. '
        'NOT meditation music, NOT sleepy, NOT ambient, NOT spoken narration, NOT humming, NOT a cappella, '
        'NOT instrumental-only.'
    ),
    'lyrics': '''[Intro]\nश्री राम... श्री राम... जय जय राम...\n\n[Verse 1]\nमन में बसो रघुनंदन, चरणों में मेरा ध्यान\nराम नाम की ज्योति जले, रोशन हो हर प्राण\n\n[Pre-Chorus]\nतेरे नाम की धुन बजे, हर धड़कन में आज\nतेरी कृपा से खिल उठे, जीवन का हर राज\n\n[Chorus]\nश्री राम जय राम, जय जय राम\nमेरे मन के दीप में, बसते श्री राम\n\n[Verse 2]\nदुख की घड़ी में साथ दो, हे दीनदयाल भगवान\nतेरा नाम ही आसरा, तेरा नाम ही सम्मान\n\n[Chorus]\nश्री राम जय राम, जय जय राम\nमेरे मन के दीप में, बसते श्री राम\n\n[Verse 3]\nअयोध्या के राजकुमार, करुणा के भंडार\nतेरे चरणों में मिल जाए, जीवन का सच्चा सार\n\n[Build]\nजय श्री राम की गूंज उठे, नभ से धरती तक\nढोल बजे और शंख बजे, प्रेम बहे हर पल\n\n[Final Chorus]\nश्री राम जय राम, जय जय राम\nमेरे मन के दीप में, बसते श्री राम\nश्री राम जय राम, जय जय राम\nजय जय राम... जय जय राम...\n\n[Outro]\nश्री राम... जय राम... जय जय राम...''',
    'bpm': 128,
    'keyscale': 'C Major',
    'timesignature': '4/4',
    'vocal_language': 'hi',
    'duration': 180,
}


def run(*args: str, cwd: Path | None = None) -> None:
    print('RUN:', ' '.join(map(str, args)), flush=True)
    subprocess.run([str(x) for x in args], cwd=str(cwd) if cwd else None, check=True)


def ensure_repo() -> None:
    if (REPO / '.git').exists():
        run('git', '-C', str(REPO), 'pull', '--ff-only', 'origin', 'main')
        return
    run('git', 'clone', '--depth', '1', REPO_URL, str(REPO))


def prepare_request() -> None:
    (WORK / 'bhajan_request.json').write_text(
        json.dumps(REQUEST, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )


def require_gpu() -> None:
    run('nvidia-smi')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('KAGGLE_PRODUCTION_FATAL: CUDA GPU is unavailable')
    count = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(i) for i in range(count)]
    print(f'GPU_COUNT={count}', flush=True)
    print(f'GPU_NAMES={names}', flush=True)


def generate_audio() -> None:
    print('AUDIO_STAGE=KAGGLE_LOCAL_ACE_STEP', flush=True)
    run(sys.executable, str(REPO / 'app' / 'kaggle_ace_step_worker.py'))
    generated = WORK / 'bhajan_source.mp3'
    if not generated.exists() or generated.stat().st_size < 100_000:
        raise RuntimeError('KAGGLE_PRODUCTION_FATAL: ACE-Step did not produce bhajan_source.mp3')
    duration = float(subprocess.check_output([
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1', str(generated)
    ], text=True).strip())
    print(f'AUDIO_OK={generated} duration={duration:.2f}s', flush=True)


def assemble_video() -> None:
    image = REPO / 'assets' / 'uks model image.png'
    if not image.exists():
        raise RuntimeError(f'KAGGLE_PRODUCTION_FATAL: singer asset missing: {image}')
    print(f'IDENTITY_SOURCE={image}', flush=True)
    run(
        sys.executable,
        str(REPO / 'app' / 'kaggle_exact_identity_video.py'),
        '--image', str(image),
        '--audio', str(AUDIO),
        '--output', str(OUTPUT),
    )
    if not OUTPUT.exists() or OUTPUT.stat().st_size < 500_000:
        raise RuntimeError('KAGGLE_PRODUCTION_FATAL: final MP4 missing or suspiciously small')


def write_manifest() -> None:
    duration = float(subprocess.check_output([
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1', str(OUTPUT)
    ], text=True).strip())
    manifest = {
        'status': 'OK',
        'backend': 'Kaggle GPU + local ACE-Step 1.5 + deterministic exact-identity assembly',
        'lightning_ai': False,
        'huggingface_zerogpu': False,
        'identity_source': 'assets/uks model image.png',
        'identity_regeneration': False,
        'duration_seconds': duration,
        'video': OUTPUT.name,
        'audio': AUDIO.name,
        'note': 'Approved singer pixels are preserved; no synthetic face/body regeneration or lip-sync is performed in this safe identity build.',
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')


def main() -> None:
    print('KAGGLE_BHAJAN_AABHA_PRODUCTION_START', flush=True)
    print('LIGHTNING_AI=DISABLED', flush=True)
    print('HF_ZEROGPU=DISABLED', flush=True)
    require_gpu()
    ensure_repo()
    prepare_request()
    generate_audio()
    assemble_video()
    write_manifest()
    print(f'KAGGLE_BHAJAN_AABHA_PRODUCTION_OK={OUTPUT}', flush=True)


if __name__ == '__main__':
    main()
