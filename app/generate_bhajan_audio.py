"""Generate only the Hindi bhajan audio using the already-proven ACE-Step path.
The visual stage is intentionally separate so GPU failures cannot invalidate music."""
from __future__ import annotations
import os
from pathlib import Path
import app.zero_cost_pipeline_v5 as base
import app.zero_cost_pipeline_v5_2 as music

LYRICS = """[Intro]\nश्री राम... श्री राम... जय जय राम...\n\n[Verse 1]\nमन में बसो रघुनंदन, चरणों में मेरा ध्यान\nराम नाम की ज्योति जले, रोशन हो हर प्राण\n\n[Pre-Chorus]\nतेरे नाम की धुन बजे, हर धड़कन में आज\nतेरी कृपा से खिल उठे, जीवन का हर राज\n\n[Chorus]\nश्री राम जय राम, जय जय राम\nमेरे मन के दीप में, बसते श्री राम\n\n[Instrumental Break]\n\n[Verse 2]\nदुख की घड़ी में साथ दो, हे दीनदयाल भगवान\nतेरा नाम ही आसरा, तेरा नाम ही सम्मान\n\n[Pre-Chorus]\nतेरी राह में चल पड़ूँ, मन में लेकर विश्वास\nराम नाम की शक्ति से, मिट जाए हर त्रास\n\n[Chorus]\nश्री राम जय राम, जय जय राम\nमेरे मन के दीप में, बसते श्री राम\n\n[Verse 3]\nअयोध्या के राजकुमार, करुणा के भंडार\nतेरे चरणों में मिल जाए, जीवन का सच्चा सार\n\n[Build]\nजय श्री राम की गूंज उठे, नभ से धरती तक\nढोल बजे और शंख बजे, प्रेम बहे हर पल\n\n[Final Chorus]\nश्री राम जय राम, जय जय राम\nमेरे मन के दीप में, बसते श्री राम\nश्री राम जय राम, जय जय राम\nजय जय राम... जय जय राम...\n\n[Outro]\nश्री राम... जय राम... जय जय राम..."""

PROMPT = music.DJ_MUSIC_PROMPT

def main():
    seconds = int(os.getenv('VIDEO_SECONDS','180'))
    if not 180 <= seconds <= 300 or seconds % 15: raise SystemExit('VIDEO_SECONDS must be 180-300 and divisible by 15')
    base.VIDEO_SECONDS = seconds
    base.PACK['lyrics'] = LYRICS
    base.PACK['music_prompt'] = PROMPT
    out = Path('output/bhajan_source.mp3')
    out.parent.mkdir(parents=True, exist_ok=True)
    print('MUSIC_BACKEND=ACE_STEP_V1_5_OFFICIAL_HF_ZEROGPU', flush=True)
    generated = Path(music.generate_music_gradio())
    if not generated.exists() or generated.stat().st_size < 20000: raise RuntimeError('MUSIC_GENERATION_FAILED')
    generated.replace(out) if generated.parent == out.parent else __import__('shutil').copy2(generated, out)
    print(f'MUSIC_OK={out} BYTES={out.stat().st_size}', flush=True)

if __name__ == '__main__': main()
