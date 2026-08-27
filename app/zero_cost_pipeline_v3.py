from __future__ import annotations

import html
import json
import math
import os
import shutil
import struct
import subprocess
import wave
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

from PIL import Image, ImageDraw, ImageFont

OUT = Path(os.getenv("OUTPUT_DIR", "output"))
VIDEOS = OUT / "videos"
MAX_VIDEOS = max(1, min(3, int(os.getenv("MAX_VIDEOS", "3"))))
SECONDS = max(30, min(60, int(os.getenv("VIDEO_SECONDS", "45"))))
FPS, W, H = 8, 720, 1280

PACKS = [
    {"slug":"ram","deity":"श्री राम","title":"राम नाम की भक्ति","mantra":"श्री राम जय राम जय जय राम","bg":(72,24,18),"accent":(222,157,72),"captions":["राम नाम में मन को शांति मिले","भक्ति की ज्योति हर हृदय में जले","श्री राम का स्मरण जीवन को उजला करे","हर सांस में राम, हर धड़कन में राम"],"narration":"श्री राम। राम नाम में मन को शांति मिले। भक्ति की ज्योति हर हृदय में जले। श्री राम का स्मरण जीवन को उजला करे। हर सांस में राम, हर धड़कन में राम। श्री राम जय राम जय जय राम।"},
    {"slug":"krishna","deity":"श्री कृष्ण","title":"कृष्ण भक्ति की मधुर धुन","mantra":"राधे कृष्ण, राधे कृष्ण","bg":(16,39,72),"accent":(86,158,218),"captions":["मुरली की मधुर धुन मन को छू जाए","श्याम नाम से हर चिंता दूर हो जाए","राधे कृष्ण की भक्ति मन में बस जाए","हर पल प्रेम, हर पल कृष्ण स्मरण"],"narration":"श्री कृष्ण। मुरली की मधुर धुन मन को छू जाए। श्याम नाम से हर चिंता दूर हो जाए। राधे कृष्ण की भक्ति मन में बस जाए। हर पल प्रेम, हर पल कृष्ण स्मरण। राधे कृष्ण, राधे कृष्ण।"},
    {"slug":"bhakti","deity":"भक्ति संध्या","title":"भक्ति की मधुर प्रार्थना","mantra":"ॐ शांति शांति शांति","bg":(43,24,54),"accent":(198,116,68),"captions":["भक्ति में मन को ठहरने दो","दीप की लौ में शांति को महसूस करो","प्रार्थना के इन पलों को अपने नाम करो","मन शांत हो, हृदय भक्ति से भर जाए"],"narration":"भक्ति संध्या। भक्ति में मन को ठहरने दो। दीप की लौ में शांति को महसूस करो। प्रार्थना के इन पलों को अपने नाम करो। मन शांत हो, हृदय भक्ति से भर जाए। ॐ शांति शांति शांति।"},
]
KEYS={"ram":["ram","राम","ayodhya","अयोध्या","sita","सीता"],"krishna":["krishna","कृष्ण","radha","राधा","vrindavan","वृंदावन"],"bhakti":["bhajan","भजन","aarti","आरती","mantra","मंत्र","bhakti","भक्ति"]}


def get_trends():
    for u in ("https://trends.google.com/trending/rss?geo=IN","https://trends.google.co.in/trends/trendingsearches/daily/rss?geo=IN"):
        try:
            root=ET.fromstring(urlopen(Request(u,headers={"User-Agent":"BhajanAabha/5.0"}),timeout=12).read())
            out=[]
            for x in root.findall(".//item"):
                t=html.unescape(x.findtext("title","").strip())
                if t: out.append({"title":t,"traffic":x.findtext("{*}approx_traffic","")})
            if out: return out
        except Exception as e: print("TREND_SOURCE_FAILED",type(e).__name__,str(e)[:160])
    return [{"title":p["deity"],"traffic":"fallback"} for p in PACKS]


def choose(items):
    chosen=[]
    for item in items:
        text=item["title"].lower()
        for p in PACKS:
            if p in chosen: continue
            if any(k.lower() in text for k in KEYS[p["slug"]]): chosen.append(p); break
        if len(chosen)>=MAX_VIDEOS: return chosen
    for p in PACKS:
        if p not in chosen: chosen.append(p)
        if len(chosen)>=MAX_VIDEOS: break
    return chosen


def espeak():
    for n in ("espeak-ng","espeak"):
        p=shutil.which(n)
        if p: return p
    raise RuntimeError("VOICE_FATAL: eSpeak executable missing")


def pcm_stats(path):
    with wave.open(str(path),"rb") as w:
        rate=w.getframerate(); n=w.getnframes(); raw=w.readframes(n)
        vals=struct.unpack("<%dh"%(len(raw)//2),raw) if raw else ()
        rms=(sum(v*v for v in vals)/len(vals))**0.5/32768 if vals else 0
        return n/max(rate,1),rms


def ensure_voice(text,path):
    exe=espeak()
    path.unlink(missing_ok=True)
    # IMPORTANT: use stdout instead of eSpeak's -w file mode. On the hosted
    # runner, eSpeak can emit a valid WAV while still returning a non-zero code.
    variants=[
        [exe,"-q","-v","hi","-s","138","-p","48","-a","150","--stdout",text],
        [exe,"-q","-v","hi+f2","-s","138","-p","48","-a","150","--stdout",text],
    ]
    errors=[]
    for cmd in variants:
        try:
            r=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=60,check=False)
            data=r.stdout or b""
            # Treat the generated audio as authoritative: a valid RIFF/WAV
            # with real duration/RMS is usable even when eSpeak reports rc != 0.
            if len(data)>=1000 and data[:4]==b"RIFF":
                path.write_bytes(data)
                dur,rms=pcm_stats(path)
                print(f"VOICE_TEST executable={exe} rc={r.returncode} bytes={len(data)} duration={dur:.2f}s rms={rms:.5f}")
                if dur>=2 and rms>=0.002:
                    print("VOICE_OK Hindi spoken WAV generated")
                    return
                path.unlink(missing_ok=True)
            errors.append(f"rc={r.returncode} bytes={len(data)} stderr={r.stderr.decode(errors='ignore')[-180:]}")
        except Exception as e: errors.append(repr(e))
    raise RuntimeError("VOICE_FATAL: no valid spoken Hindi WAV: "+" | ".join(errors))


def font(size,bold=False):
    names=["/usr/share/fonts/truetype/noto/NotoSansDevanagari-Bold.ttf" if bold else "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf","/usr/share/fonts/opentype/noto/NotoSansDevanagari-Bold.ttf" if bold else "/usr/share/fonts/opentype/noto/NotoSansDevanagari-Regular.ttf"]
    for n in names:
        if Path(n).exists(): return ImageFont.truetype(n,size)
    return ImageFont.load_default()


def frame(p,n,tf,bf,sf):
    t=n/FPS; im=Image.new("RGB",(W,H),p["bg"]); d=ImageDraw.Draw(im,"RGBA"); cx,cy=W//2,int(H*.45)
    for rr in range(330,40,-24): d.ellipse((cx-rr,cy-rr,cx+rr,cy+rr),outline=p["accent"]+(max(12,75-rr//6),),width=2)
    for i in range(28):
        a=t*.22+i*math.pi/14; x=cx+int(math.cos(a)*260); y=cy+int(math.sin(a)*260); d.ellipse((x-4,y-4,x+4,y+4),fill=p["accent"]+(150,))
    d.ellipse((cx-110,cy+35,cx+110,cy+105),fill=(100,52,22,240)); d.ellipse((cx-12,cy-8,cx+12,cy+65),fill=(255,190,45,255)); d.ellipse((cx-24,cy+8,cx+24,cy+54),fill=(255,245,190,245))
    d.rounded_rectangle((24,24,W-24,145),radius=26,fill=(5,5,9,205),outline=p["accent"]+(230,),width=2)
    d.text((W//2,46),"BHAJAN AABHA",font=sf,anchor="ma",fill=(255,240,210,255)); d.text((W//2,78),p["deity"],font=tf,anchor="ma",fill=(255,250,235,255))
    idx=min(3,int(t/SECONDS*4)); y=H-230; d.rounded_rectangle((24,y-42,W-24,H-68),radius=25,fill=(5,5,9,220)); d.text((W//2,y),p["captions"][idx],font=bf,anchor="ma",fill=(255,255,255,255)); d.text((W//2,H-43),p["mantra"],font=sf,anchor="ms",fill=p["accent"]+(255,))
    return im.tobytes()


def render(video,audio,p):
    cmd=["ffmpeg","-y","-f","rawvideo","-pix_fmt","rgb24","-s",f"{W}x{H}","-r",str(FPS),"-i","-","-i",str(audio),"-t",str(SECONDS),"-map","0:v:0","-map","1:a:0","-vf","format=yuv420p","-c:v","libx264","-preset","veryfast","-crf","21","-c:a","aac","-b:a","160k","-ar","44100","-af","apad","-t",str(SECONDS),"-movflags","+faststart",str(video)]
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    try:
        tf,bf,sf=font(42,True),font(30),font(20)
        for n in range(SECONDS*FPS): proc.stdin.write(frame(p,n,tf,bf,sf))
        proc.stdin.close(); err=proc.stderr.read().decode(errors="ignore"); rc=proc.wait()
        if rc: raise RuntimeError("FFMPEG_FATAL: "+err[-1800:])
    finally:
        if proc.poll() is None: proc.kill()


def validate(video):
    data=json.loads(subprocess.check_output(["ffprobe","-v","error","-show_entries","stream=codec_type,codec_name,width,height,duration","-of","json",str(video)],text=True))
    vs=[x for x in data["streams"] if x.get("codec_type")=="video"]; aa=[x for x in data["streams"] if x.get("codec_type")=="audio"]
    if not vs or not aa or vs[0].get("width")!=W or vs[0].get("height")!=H or aa[0].get("codec_name")!="aac": raise RuntimeError("OUTPUT_FATAL: invalid MP4 streams")
    if float(aa[0].get("duration",0))<1: raise RuntimeError("OUTPUT_FATAL: audio duration missing")


def main():
    OUT.mkdir(parents=True,exist_ok=True); VIDEOS.mkdir(parents=True,exist_ok=True)
    for x in VIDEOS.glob("*.mp4"): x.unlink()
    items=get_trends(); packs=choose(items); print("ARCHITECTURE=v3 ZERO_COST=true KAGGLE=false PAID_SERVICES=false"); print(f"TREND_ITEMS={len(items)} SELECTED={len(packs)}")
    results=[]
    for i,p in enumerate(packs,1):
        print(f"PREPARING {i}/{len(packs)}: {p['deity']}"); audio=OUT/f"{p['slug']}_narration.wav"; ensure_voice(p["narration"],audio); video=VIDEOS/f"{datetime.now(timezone.utc):%Y%m%d}_{i}_{p['slug']}.mp4"; render(video,audio,p); validate(video); audio.unlink(missing_ok=True); print("VIDEO_OK",video); results.append({"topic":p["deity"],"title":p["title"],"video":str(video),"duration_sec":SECONDS})
    state={"channel":"Bhajan Aabha","architecture":"github-runner-only-v3","videos":results,"paid_services":False,"paid_gpu":False,"kaggle":False,"external_media_downloads":False,"voice":"eSpeak Hindi via stdout with WAV/RMS validation","status":"READY_FOR_RELEASE","generated_at_utc":datetime.now(timezone.utc).isoformat()}
    (OUT/"run_state.json").write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding="utf-8"); (OUT/"manifest.json").write_text(json.dumps({"trends":items[:20],"videos":results},ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps(state,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
