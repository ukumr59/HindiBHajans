"""Generate Hindi bhajan audio on a zero-cost Kaggle GPU.

Hugging Face/ZeroGPU is deliberately NOT used: it previously exhausted/aborted
quota and made the daily production pipeline fail before video generation.
"""
from __future__ import annotations
import os
from pathlib import Path

LYRICS = """[Intro]\nश्री राम... श्री राम... जय जय राम...\n\n[Verse 1]\nमन में बसो रघुनंदन, चरणों में मेरा ध्यान\nराम नाम की ज्योति जले, रोशन हो हर प्राण\n\n[Pre-Chorus]\nतेरे नाम की धुन बजे, हर धड़कन में आज\nतेरी कृपा से खिल उठे, जीवन का हर राज\n\n[Chorus]\nश्री राम जय राम, जय जय राम\nमेरे मन के दीप में, बसते श्री राम\n\n[Instrumental Break]\n\n[Verse 2]\nदुख की घड़ी में साथ दो, हे दीनदयाल भगवान\nतेरा नाम ही आसरा, तेरा नाम ही सम्मान\n\n[Pre-Chorus]\nतेरी राह में चल पड़ूँ, मन में लेकर विश्वास\nराम नाम की शक्ति से, मिट जाए हर त्रास\n\n[Chorus]\nश्री राम जय राम, जय जय राम\nमेरे मन के दीप में, बसते श्री राम\n\n[Verse 3]\nअयोध्या के राजकुमार, करुणा के भंडार\nतेरे चरणों में मिल जाए, जीवन का सच्चा सार\n\n[Build]\nजय श्री राम की गूंज उठे, नभ से धरती तक\nढोल बजे और शंख बजे, प्रेम बहे हर पल\n\n[Final Chorus]\nश्री राम जय राम, जय जय राम\nमेरे मन के दीप में, बसते श्री राम\nश्री राम जय राम, जय जय राम\nजय जय राम... जय जय राम...\n\n[Outro]\nश्री राम... जय राम... जय जय राम..."""

PROMPT = """Modern high-energy Hindi devotional bhajan made like a current YouTube DJ devotional song, 128 BPM, 4/4, loud polished commercial stereo production, powerful expressive Hindi male lead vocal clearly singing every lyric with natural emotion and clean pronunciation, catchy devotional melody, huge memorable chorus, energetic EDM arrangement, punchy four-on-the-floor kick, deep controlled sub bass, modern synth bass, bright synth leads, wide pads, electronic percussion, claps, dhol and dholak layered with tabla, cinematic risers, tasteful temple bells, bansuri accents, harmonium texture, short instrumental intro, strong verse build, massive chorus/drop, rhythmic instrumental break, final chorus with layered backing vocals, professional YouTube/radio loudness and DJ playback energy. NOT meditation music, NOT sleepy, NOT ambient, NOT acoustic-only, NOT spoken narration, NOT humming, NOT a cappella, NOT instrumental-only."""

def main():
    seconds = int(os.getenv('VIDEO_SECONDS','180'))
    if not 180 <= seconds <= 300 or seconds % 15:
        raise SystemExit('VIDEO_SECONDS must be 180-300 and divisible by 15')
    os.environ['BH_LYRICS'] = LYRICS
    os.environ['BH_MUSIC_PROMPT'] = PROMPT
    print('MUSIC_BACKEND=KAGGLE_FREE_T4_ACE_STEP_1_5', flush=True)
    print('MUSIC_POLICY=NO_HUGGINGFACE_ZEROGPU', flush=True)
    from app.kaggle_audio_dispatch import main as dispatch_audio
    dispatch_audio()
    out = Path('output/bhajan_source.mp3')
    if not out.exists() or out.stat().st_size < 100_000:
        raise RuntimeError('MUSIC_GENERATION_FAILED: Kaggle did not return a valid MP3')
    print(f'MUSIC_OK={out} BYTES={out.stat().st_size}', flush=True)

if __name__ == '__main__': main()
