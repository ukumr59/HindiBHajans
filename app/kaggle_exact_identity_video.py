from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from PIL import Image

ROOT = Path('/kaggle/working')
DEFAULT_IMAGE = ROOT / 'bhajan_aabha_locked_identity_source.png'
DEFAULT_OUT = ROOT / 'bhajan_aabha_exact_identity_final.mp4'


def run(*args: str) -> None:
    print('RUN:', ' '.join(args), flush=True)
    subprocess.run(list(args), check=True)


def resolve_image(requested: Path) -> Path:
    """Resolve the exact approved singer image without making the user guess paths.

    Priority:
      1) explicitly requested existing file
      2) known locked-source filenames under /kaggle/working
      3) the original 1024x1536 identity-sheet PNG/JPG under /kaggle/input
    """
    if requested.exists():
        return requested

    working_names = [
        'bhajan_aabha_locked_identity_source.png',
        'bhajan_aabha_locked_singer_highres.png',
        'bhajan_aabha_locked_singer_cutout.png',
        'bhajan_aabha_locked_singer_cutout_v2.png',
        'bhajan_aabha_locked_singer_cutout_v3.png',
        'uks model image.png',
    ]
    for name in working_names:
        p = ROOT / name
        if p.exists() and p.is_file():
            try:
                with Image.open(p) as im:
                    if im.width >= 200 and im.height >= 400:
                        return p
            except Exception:
                pass

    candidates: list[tuple[int, Path]] = []
    input_root = Path('/kaggle/input')
    if input_root.exists():
        for p in input_root.rglob('*'):
            if not p.is_file() or p.suffix.lower() not in {'.png', '.jpg', '.jpeg'}:
                continue
            try:
                with Image.open(p) as im:
                    score = 0
                    if im.size == (1024, 1536):
                        score += 10000
                    if p.name.lower() == 'uks model image.png':
                        score += 5000
                    if 'virtual-singer-master' in str(p).lower():
                        score += 2000
                    if im.width >= 200 and im.height >= 400:
                        score += min(im.width * im.height, 1000000) // 1000
                    if score:
                        candidates.append((score, p))
            except Exception:
                continue

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        chosen = candidates[0][1]
        print(f'IMAGE_AUTO_RESOLVED={chosen}', flush=True)
        return chosen

    raise FileNotFoundError(
        'No approved singer image could be resolved. Expected the locked source '
        'or the original 1024x1536 identity-sheet image under /kaggle/input.'
    )


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
    candidates: list[Path] = []
    for ext in ('*.mp3', '*.wav', '*.m4a', '*.flac'):
        candidates.extend(ROOT.rglob(ext))
    candidates = [p for p in candidates if p.stat().st_size > 100_000 and 'test' not in p.name.lower()]
    if not candidates:
        raise FileNotFoundError('No bhajan audio was found under /kaggle/working')
    return max(candidates, key=lambda p: p.stat().st_size)


def prepare_source(image_path: Path, tmp: Path) -> Path:
    img = Image.open(image_path).convert('RGB')
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

    image = resolve_image(Path(args.image))
    audio = find_audio(Path(args.audio) if args.audio else None)
    out = Path(args.output)
    tmp = ROOT / 'exact_identity_video_tmp'
    tmp.mkdir(parents=True, exist_ok=True)

    print(f'LOCKED_IMAGE={image}', flush=True)
    print(f'AUDIO={audio}', flush=True)

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

    chapter = duration / 12.0
    filters = [
        f"scale=720:1280:force_original_aspect_ratio=decrease,pad=720:1280:(ow-iw)/2:(oh-ih)/2,zoompan=z='min(zoom+0.0008,1.10)':d={max(1,round(chapter*24))}:s=720x1280:fps=24",
        f"scale=720:1280:force_original_aspect_ratio=decrease,pad=720:1280:(ow-iw)/2:(oh-ih)/2,zoompan=z='min(zoom+0.0006,1.08)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={max(1,round(chapter*24))}:s=720x1280:fps=24",
        f"scale=760:-1,crop=720:1280:x='(iw-720)/2':y='max(0,(ih-1280)*(t/{max(chapter,0.001)}))',fps=24",
        f"scale=800:-1,crop=720:1280:x='max(0,(iw-720)*(1-t/{max(chapter,0.001)}))':y='(ih-1280)/2',fps=24",
    ]

    segs: list[Path] = []
    for i in range(12):
        seg = tmp / f'scene_{i+1:02d}.mp4'
        vf = filters[i % len(filters)]
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
    print('IMPORTANT=This stage preserves the selected singer source exactly; it does not regenerate the face or add synthetic lip motion.')


if __name__ == '__main__':
    main()
