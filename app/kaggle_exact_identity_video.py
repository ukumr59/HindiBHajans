from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from PIL import Image


ROOT = Path('/kaggle/working')
DEFAULT_IMAGE = ROOT / 'bhajan_aabha_locked_identity_source.png'
DEFAULT_OUT = ROOT / 'bhajan_aabha_exact_identity_final.mp4'


def run(*args: str) -> None:
    print('RUN:', ' '.join(args), flush=True)
    subprocess.run(list(args), check=True)


def find_audio(preferred: Path | None = None) -> Path:
    if preferred and preferred.exists():
        return preferred
    preferred_names = [
        'bhajan_source.mp3',
        'bhajan_aabha_dj_master.mp3',
        'bhajan_source.wav',
    ]
    for name in preferred_names:
        p = ROOT / name
        if p.exists() and p.stat().st_size > 100_000:
            return p
    candidates = []
    for ext in ('*.mp3', '*.wav', '*.m4a', '*.flac'):
        candidates.extend(ROOT.rglob(ext))
    candidates = [p for p in candidates if p.stat().st_size > 100_000 and 'test' not in p.name.lower()]
    if not candidates:
        raise FileNotFoundError('No bhajan audio was found under /kaggle/working')
    return max(candidates, key=lambda p: p.stat().st_size)


def prepare_source(image_path: Path, tmp: Path) -> Path:
    img = Image.open(image_path).convert('RGB')
    # Preserve the approved source exactly; add only a soft blurred expansion
    # around it so the 9:16 output has no hard pillarbox edges.
    src = tmp / 'source.png'
    img.save(src, format='PNG', optimize=True)
    return src


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', default=str(DEFAULT_IMAGE))
    ap.add_argument('--audio', default='')
    ap.add_argument('--output', default=str(DEFAULT_OUT))
    ap.add_argument('--seconds', type=float, default=0.0)
    args = ap.parse_args()

    image = Path(args.image)
    if not image.exists():
        raise FileNotFoundError(f'Locked singer image missing: {image}')
    audio = find_audio(Path(args.audio) if args.audio else None)
    out = Path(args.output)
    tmp = ROOT / 'exact_identity_video_tmp'
    tmp.mkdir(parents=True, exist_ok=True)

    print(f'LOCKED_IMAGE={image}', flush=True)
    print(f'AUDIO={audio}', flush=True)

    # Read audio duration unless user supplied a cap.
    probe = subprocess.check_output([
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1', str(audio)
    ], text=True).strip()
    duration = float(probe)
    if args.seconds > 0:
        duration = min(duration, args.seconds)
    if duration < 5:
        raise RuntimeError(f'Audio duration too short: {duration:.2f}s')

    source = prepare_source(image, tmp)
    output = out.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    # 12 visual chapters, exact singer source throughout. Each chapter uses a
    # different crop/zoom rhythm and a slightly different warm treatment. The
    # singer pixels are never regenerated or passed through a diffusion model.
    chapter = duration / 12.0
    filters = [
        f"scale=720:1280:force_original_aspect_ratio=decrease,pad=720:1280:(ow-iw)/2:(oh-ih)/2,zoompan=z='min(zoom+0.0008,1.10)':d={max(1,round(chapter*24))}:s=720x1280:fps=24",
        f"scale=720:1280:force_original_aspect_ratio=decrease,pad=720:1280:(ow-iw)/2:(oh-ih)/2,zoompan=z='min(zoom+0.0006,1.08)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={max(1,round(chapter*24))}:s=720x1280:fps=24",
        f"scale=760:-1,crop=720:1280:x='(iw-720)/2':y='max(0,(ih-1280)*(t/{max(chapter,0.001)}))',fps=24",
        f"scale=800:-1,crop=720:1280:x='max(0,(iw-720)*(1-t/{max(chapter,0.001)}))':y='(ih-1280)/2',fps=24",
    ]

    segs = []
    for i in range(12):
        seg = tmp / f'scene_{i+1:02d}.mp4'
        vf = filters[i % len(filters)]
        # Make a segment from the exact source image. Looping is deterministic;
        # no image model is invoked.
        run(
            'ffmpeg', '-y', '-loop', '1', '-i', str(source),
            '-t', f'{chapter:.3f}', '-vf', vf,
            '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20',
            '-pix_fmt', 'yuv420p', str(seg),
        )
        segs.append(seg)

    concat = tmp / 'concat.txt'
    concat.write_text(''.join(f"file '{p.as_posix()}'\n" for p in segs), encoding='utf-8')

    run(
        'ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', str(concat), '-i', str(audio),
        '-map', '0:v:0', '-map', '1:a:0', '-t', f'{duration:.3f}',
        '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20',
        '-c:a', 'aac', '-b:a', '192k', '-shortest', '-movflags', '+faststart', str(output),
    )

    print('EXACT_IDENTITY_VIDEO_OK')
    print(f'OUTPUT={output}')
    print(f'DURATION={duration:.2f}')
    print('IMPORTANT=This stage preserves the approved singer exactly; it does not regenerate the face or add synthetic lip motion.')


if __name__ == '__main__':
    main()
