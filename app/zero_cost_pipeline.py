from __future__ import annotations

import html
import json
import os
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from PIL import Image, ImageDraw, ImageFilter, ImageFont

OUT = Path(os.getenv("OUTPUT_DIR", "output"))
VIDEOS = OUT / "videos"
MAX_VIDEOS = max(1, min(3, int(os.getenv("MAX_VIDEOS", "3"))))
SECONDS = max(30, min(60, int(os.getenv("VIDEO_SECONDS", "45"))))
FPS = 8
W, H = 720, 1280

# Fully self-contained zero-cost packs. No external image/audio downloads.
PACKS = [
    {"slug":"ram","deity":"श्री राम","title":"राम नाम की भक्ति","mantra":"श्री राम जय राम जय जय राम","accent":(222,157,72),"bg":(72,24,18),"captions":["राम नाम में मन को शांति मिले","भक्ति की ज्योति हर हृदय में जले","श्री राम का स्मरण जीवन को उजला करे","हर सांस में राम, हर धड़कन में राम"],"narration":"श्री राम। राम नाम में मन को शांति मिले। भक्ति की ज्योति हर हृदय में जले। श्री राम का स्मरण जीवन को उजला करे। हर सांस में राम, हर धड़कन में राम। श्री राम जय राम जय जय राम।"},
    {"slug":"krishna","deity":"श्री कृष्ण","title":"कृष्ण भक्ति की मधुर धुन","mantra":"राधे कृष्ण, राधे कृष्ण","accent":(86,158,218),"bg":(16,39,72),"captions":["मुरली की मधुर धुन मन को छू जाए","श्याम नाम से हर चिंता दूर हो जाए","राधे कृष्ण की भक्ति मन में बस जाए","हर पल प्रेम, हर पल कृष्ण स्मरण"],"narration":"श्री कृष्ण। मुरली की मधुर धुन मन को छू जाए। श्याम नाम से हर चिंता दूर हो जाए। राधे कृष्ण की भक्ति मन में बस जाए। हर पल प्रेम, हर पल कृष्ण स्मरण। राधे कृष्ण, राधे कृष्ण।"},
    {"slug":"bhakti","deity":"भक्ति संध्या","title":"भक्ति की मधुर प्रार्थना","mantra":"ॐ शांति शांति शांति","accent":(198,116,68),"bg":(43,24,54),"captions":["भक्ति में मन को ठहरने दो","दीप की लौ में शांति को महसूस करो","प्रार्थना के इन पलों को अपने नाम करो","मन शांत हो, हृदय भक्ति से भर जाए"],"narration":"भक्ति संध्या। भक्ति में मन को ठहरने दो। दीप की लौ में शांति को महसूस करो। प्रार्थना के इन पलों को अपने नाम करो। मन शांत हो, हृदय भक्ति से भर जाए। ॐ शांति शांति शांति।"},
]

KEYWORDS={"ram":["ram","राम","ayodhya","अयोध्या","sita","सीता","raghu","रघुनाथ"],"krishna":["krishna","कृष्ण","kanha","कान्हा","radha","राधा","janmashtami","जन्माष्टमी","vrindavan","वृंदावन"],"bhakti":["bhajan","भजन","aarti","आरती","mantra","मंत्र","bhakti","भक्ति","kirtan","कीर्तन"]}


def fetch_trends() -> list[dict]:
    for url in ("https://trends.google.com/trending/rss?geo=IN","https://trends.google.co.in/trends/trendingsearches/daily/rss?geo=IN"):
        try:
            req=Request(url,headers={"User-Agent":"BhajanAabha/3.0"})
            root=ET.fromstring(urlopen(req,timeout=20).read())
            items=[]
            for item in root.findall(".//item"):
                title=html.unescape(item.findtext("title",default="").strip())
                if title: items.append({"title":title,"traffic":item.findtext("{*}approx_traffic",default=""),"source":url})
            if items: return items
        except Exception as exc: print(f"TREND_SOURCE_FAILED {url}: {exc}")
    return []


def choose_packs(trends:list[dict])->list[dict]:
    selected=[]; seen=set()
    for trend in trends:
        text=trend["title"].lower(); scored=[]
        for p in PACKS:
            score=sum(1 for k in KEYWORDS[p["slug"]] if k.lower() in text)
            if score: scored.append((score,p))
        if scored:
            p=max(scored,key=lambda x:x[0])[1]
            if p["slug"] not in seen: selected.append(p); seen.add(p["slug"])
        if len(selected)>=MAX_VIDEOS: break
    for p in PACKS:
        if len(selected)>=MAX_VIDEOS: break
        if p["slug"] not in seen: selected.append(p); seen.add(p["slug"])
    return selected[:MAX_VIDEOS]


def font(size:int,bold:bool=False):
    names=["/usr/share/fonts/truetype/noto/NotoSansDevanagari-Bold.ttf" if bold else "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf","/usr/share/fonts/opentype/noto/NotoSansDevanagari-Bold.ttf" if bold else "/usr/share/fonts/opentype/noto/NotoSansDevanagari-Regular.ttf"]
    for path in names:
        if Path(path).exists(): return ImageFont.truetype(path,size)
    return ImageFont.load_default()


def _audio_duration(path:Path)->float:
    probe=shutil.which("ffprobe")
    if not probe: return 0.0
    try:
        out=subprocess.check_output([probe,"-v","error","-show_entries","format=duration","-of","default=noprint_wrappers=1:nokey=1",str(path)],text=True,timeout=15).strip()
        return float(out or 0)
    except Exception: return 0.0


def ensure_voice(text:str,path:Path,seconds:int)->None:
    """Generate real Hindi speech locally with eSpeak; never substitute a tone/hum."""
    candidates=[shutil.which("espeak-ng"),shutil.which("espeak")]
    espeak=next((x for x in candidates if x),None)
    if not espeak: raise RuntimeError("No eSpeak executable is installed")
    path.unlink(missing_ok=True)
    attempts=[
        [espeak,"-q","-v","hi","-s","142","-p","48","-a","125","-w",str(path),text],
        [espeak,"-q","-v","hi+f2","-s","142","-p","48","-a","125","-w",str(path),text],
    ]
    errors=[]
    for cmd in attempts:
        path.unlink(missing_ok=True)
        try:
            result=subprocess.run(cmd,check=False,timeout=60,capture_output=True,text=True)
            size=path.stat().st_size if path.exists() else 0
            duration=_audio_duration(path) if size else 0
            if result.returncode==0 and size>=500 and duration>=1.0:
                print(f"VOICE_OK executable={Path(espeak).name} bytes={size} duration={duration:.2f}s")
                return
            errors.append(f"rc={result.returncode},bytes={size},duration={duration:.2f},stderr={result.stderr[-300:]}")
        except Exception as exc: errors.append(repr(exc))
    raise RuntimeError("Hindi narration generation failed; no valid spoken WAV was produced. " + " | ".join(errors))


def make_background(pack:dict,frame:int)->Image.Image:
    t=frame/FPS; base=Image.new("RGB",(W,H),pack["bg"]); px=base.load(); r0,g0,b0=pack["bg"]
    for y in range(H):
        v=int(22*(1-y/H))
        for x in range(0,W,4):
            glow=int(18*max(0.0,1.0-(((x-W*.5)/(W*.7))**2+((y-H*.42)/(H*.75))**2)))
            c=(min(255,r0+v+glow),min(255,g0+v+glow),min(255,b0+v+glow))
            for xx in range(x,min(x+4,W)): px[xx,y]=c
    d=ImageDraw.Draw(base,"RGBA"); cx,cy=W//2,int(H*.42); radius=180+int(25*__import__('math').sin(t*.7))
    for rr in range(radius,20,-18): d.ellipse((cx-rr,cy-rr,cx+rr,cy+rr),outline=pack["accent"]+(max(15,90-int(rr*.3)),),width=2)
    d.ellipse((cx-90,cy-90,cx+90,cy+90),fill=(255,210,120,18))
    return base.filter(ImageFilter.GaussianBlur(.15))


def make_video(path:Path,audio:Path,pack:dict,seconds:int)->None:
    title_font=font(42,True); body_font=font(30); small_font=font(20)
    proc=subprocess.Popen(["ffmpeg","-y","-f","rawvideo","-pix_fmt","rgb24","-s",f"{W}x{H}","-r",str(FPS),"-i","-","-i",str(audio),"-t",str(seconds),"-map","0:v:0","-map","1:a:0","-vf","format=yuv420p","-c:v","libx264","-preset","veryfast","-crf","21","-c:a","aac","-b:a","160k","-ar","44100","-af","apad","-t",str(seconds),"-movflags","+faststart",str(path)],stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    try:
        import math
        for n in range(seconds*FPS):
            t=n/FPS; im=make_background(pack,n).convert("RGBA"); d=ImageDraw.Draw(im,"RGBA"); accent=pack["accent"]
            # diya / flame
            cx,cy=W//2,int(H*.55); d.ellipse((cx-80,cy-20,cx+80,cy+35),fill=(120,65,25,220)); d.ellipse((cx-8,cy-70,cx+8,cy-5),fill=(255,210,80,255)); d.ellipse((cx-20,cy-48,cx+20,cy-5),fill=(255,245,190,240))
            for i in range(12):
                a=t*.35+i*.52; x=cx+int(math.cos(a)*260); y=cy+int(math.sin(a)*260); r=3+(i%3); d.ellipse((x-r,y-r,x+r,y+r),fill=accent+(130,))
            d.rounded_rectangle((30,30,W-30,125),radius=24,fill=(8,8,12,190),outline=accent+(220,),width=2)
            d.text((W//2,52),"BHAJAN AABHA",font=small_font,anchor="ma",fill=(255,240,210,255)); d.text((W//2,80),pack["deity"],font=title_font,anchor="ma",fill=(255,250,235,255))
            cap=pack["captions"][min(len(pack["captions"])-1,int(t/seconds*len(pack["captions"])))]; y=H-210
            d.rounded_rectangle((30,y-35,W-30,H-72),radius=24,fill=(8,8,12,205)); d.text((W//2,y),cap,font=body_font,anchor="ma",fill=(255,255,255,255)); d.text((W//2,H-48),pack["mantra"],font=small_font,anchor="ms",fill=accent+(255,))
            proc.stdin.write(im.convert("RGB").tobytes())
        proc.stdin.close(); err=proc.stderr.read().decode("utf-8",errors="ignore"); code=proc.wait()
        if code!=0: raise RuntimeError(f"ffmpeg failed: {err[-2500:]}")
    finally:
        if proc.poll() is None: proc.kill()


def write_metadata(path:Path,pack:dict,video:Path)->None:
    path.write_text(json.dumps({"title":pack["title"],"video":str(video),"audio":"locally generated Hindi narration using eSpeak","copyright_mode":"self_generated_audio_and_visuals"},ensure_ascii=False,indent=2),encoding="utf-8")


def main()->None:
    OUT.mkdir(parents=True,exist_ok=True); VIDEOS.mkdir(parents=True,exist_ok=True)
    for old in VIDEOS.glob("*.mp4"): old.unlink()
    trends=fetch_trends(); packs=choose_packs(trends); print(f"TREND_ITEMS={len(trends)} SELECTED={len(packs)}")
    for t in trends[:20]: print("TREND:",t["title"],t.get("traffic",""))
    results=[]
    for index,pack in enumerate(packs,1):
        print(f"PREPARING {index}/{len(packs)}: {pack['deity']}")
        audio=OUT/f"{pack['slug']}_narration.wav"; ensure_voice(pack["narration"],audio,SECONDS)
        video=VIDEOS/f"{datetime.now(timezone.utc):%Y%m%d}_{index}_{pack['slug']}.mp4"; meta=OUT/f"{pack['slug']}.json"
        print(f"RENDERING {index}/{len(packs)}: {pack['title']}"); make_video(video,audio,pack,SECONDS); write_metadata(meta,pack,video); audio.unlink(missing_ok=True)
        results.append({"topic":pack["deity"],"slug":pack["slug"],"video":str(video),"duration_sec":SECONDS,"mode":"zero_cost_self_generated_visuals_plus_hindi_narration"})
    state={"channel":"Bhajan Aabha","generated_at_utc":datetime.now(timezone.utc).isoformat(),"trend_source":"Google Trends RSS with deterministic fallback","trend_count":len(trends),"videos":results,"copyright_mode":"self-generated visuals and local Hindi narration","gpu":False,"paid_services":False,"kaggle":False,"human_intervention_after_setup":False,"quality_gate":"real spoken WAV + AAC video + 720x1280","publish_status":"READY_FOR_GITHUB_RELEASE"}
    (OUT/"run_state.json").write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding="utf-8"); (OUT/"manifest.json").write_text(json.dumps({"videos":results,"trends":trends[:20]},ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps(state,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
