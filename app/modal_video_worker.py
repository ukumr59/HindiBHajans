from __future__ import annotations
import subprocess,tempfile
from pathlib import Path
import modal
app=modal.App("hindibhajans-video")
VOL=modal.Volume.from_name("hindibhajans-model-cache",create_if_missing=True); MD="/models"
image=(modal.Image.debian_slim(python_version="3.11").apt_install("git","ffmpeg").pip_install("torch","torchvision","huggingface_hub","opencv-python-headless","numpy","pyloudnorm","eva-decord"))
PROMPT="A single Indian devotional singer, the same person as the reference image, singing a Hindi bhajan before a Hindu deity in a serene temple, wearing traditional Indian devotional clothing, natural singing mouth movement synchronized to the audio, subtle devotional head and upper-body movement, stable facial identity, only one person visible."
NEG="blurry, distorted face, identity drift, extra person, duplicate person, malformed hands, fused fingers, deformed mouth, jitter, flicker, camera cut, text, watermark"
def run(*a): subprocess.run([str(x) for x in a],check=True)
def first(p):
 x=sorted(Path(p).rglob("*.mp4"),key=lambda q:q.stat().st_mtime,reverse=True)
 if not x: raise RuntimeError("ECHOMIMIC_RETURNED_NO_MP4")
 return x[0]
@app.function(image=image,gpu="T4",timeout=7200,volumes={MD:VOL},env={"HF_HOME":f"{MD}/hf"})
def generate(image_bytes:bytes,audio_bytes:bytes,seconds:int=180)->bytes:
 if seconds<180 or seconds>300 or seconds%15: raise ValueError("seconds must be 180-300 and divisible by 15")
 r=Path(tempfile.mkdtemp(prefix="bhajan-video-")); ref=r/"singer.png"; audio=r/"bhajan.mp3"; ref.write_bytes(image_bytes); audio.write_bytes(audio_bytes)
 repo=r/"echomimic_v3"; run("git","clone","--depth","1","https://github.com/antgroup/echomimic_v3.git",repo); run("python","-m","pip","install","-q","-r",repo/"requirements.txt"); run("python","-m","pip","install","-q","transformers==4.48.3","diffusers==0.32.2")
 # T4-safe memory profile: offload before CUDA placement, then use smaller 65-frame/512x512 chunks.
 infer=repo/"infer_flash.py"; src=infer.read_text(); old='pipeline.to(device=device)'; count=src.count(old)
 if count != 2: raise RuntimeError(f"ECHOMIMIC_MEMORY_LAYOUT_CHANGED: expected 2 pipeline.to calls, found {count}")
 src=src.replace(old,'pipeline.enable_sequential_cpu_offload()',2)
 # Force the Wav2Vec path to return all hidden states even if the installed Transformers config defaults differ.
 call_old='output_hidden_states=True)'; call_new='output_hidden_states=True, return_dict=True)'; call_count=src.count(call_old)
 if call_count != 1: raise RuntimeError(f"ECHOMIMIC_WAV2VEC_CALL_LAYOUT_CHANGED: expected 1 audio encoder call, found {call_count}")
 src=src.replace(call_old,call_new,1)
 compile(src,str(infer),"exec"); infer.write_text(src)
 wavsrc=repo/"src/wav2vec2.py"; w=wavsrc.read_text()
 marker='self.config.output_attentions = False\n\n        output_hidden_states = ('
 if marker not in w: raise RuntimeError("ECHOMIMIC_WAV2VEC_SOURCE_LAYOUT_CHANGED")
 w=w.replace(marker,'self.config.output_attentions = False\n        self.config.output_hidden_states = True\n        self.config.return_dict = True\n\n        output_hidden_states = (',1)
 marker2='self.config.output_attentions = True\n\n        output_hidden_states = ('
 if marker2 not in w: raise RuntimeError("ECHOMIMIC_WAV2VEC_ENCODE_LAYOUT_CHANGED")
 w=w.replace(marker2,'self.config.output_attentions = True\n        self.config.output_hidden_states = True\n        self.config.return_dict = True\n\n        output_hidden_states = (',1)
 compile(w,str(wavsrc),"exec"); wavsrc.write_text(w)
 print("ECHOMIMIC_T4_MEMORY_PATCH=PASS sequential_cpu_offload",flush=True)
 print("ECHOMIMIC_T4_PROFILE=PASS frames=65 sample=512x512",flush=True)
 print("ECHOMIMIC_RUNTIME_PIN=PASS transformers=4.48.3 diffusers=0.32.2",flush=True)
 print("ECHOMIMIC_WAV2VEC_HIDDEN_STATES_PATCH=PASS",flush=True)
 from huggingface_hub import snapshot_download
 base=Path(MD)/"Wan2.1-Fun-V1.1-1.3B-InP"; wav=Path(MD)/"chinese-wav2vec2-base"; flash=Path(MD)/"echomimicv3-flash-pro"
 snapshot_download("alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP",local_dir=base); snapshot_download("TencentGameMate/chinese-wav2vec2-base",local_dir=wav); snapshot_download("BadToBest/EchoMimicV3",local_dir=flash,allow_patterns=["echomimicv3-flash-pro/*"]); VOL.commit()
 transformer=flash/"echomimicv3-flash-pro/diffusion_pytorch_model.safetensors"
 if not transformer.is_file(): raise RuntimeError(f"ECHOMIMIC_FLASH_CHECKPOINT_MISSING: expected {transformer}")
 print(f"ECHOMIMIC_FLASH_CHECKPOINT=PASS path={transformer}",flush=True)
 norm=r/"audio16k.wav"; run("ffmpeg","-y","-v","error","-i",audio,"-ac","1","-ar","16000","-c:a","pcm_s16le",norm); seg=r/"segments"; seg.mkdir(); fps=25; frames=65; ss=frames/fps
 for i in range(int((seconds+ss-1)//ss)):
  start=i*ss; remain=min(ss,seconds-start)
  if remain<.5: break
  aw=seg/f"a{i:04}.wav"; run("ffmpeg","-y","-v","error","-ss",f"{start:.3f}","-i",norm,"-t",f"{remain:.3f}","-ar","16000","-ac","1",aw); raw=seg/f"r{i:04}"; raw.mkdir()
  run("python",infer,"--image_path",ref,"--audio_path",aw,"--prompt",PROMPT,"--num_inference_steps","8","--config_path",repo/"config/config.yaml","--model_name",base,"--ckpt_idx","50000","--transformer_path",transformer,"--save_path",raw,"--wav2vec_model_dir",wav,"--sampler_name","Flow_Unipc","--video_length",frames,"--guidance_scale","5.0","--audio_guidance_scale","2.5","--audio_scale","1.0","--neg_scale","1.0","--neg_steps","0","--seed",4300+i,"--enable_teacache","--teacache_threshold","0.1","--num_skip_start_steps","5","--weight_dtype","float16","--sample_size","512","512","--fps",fps,"--negative_prompt",NEG)
  run("ffmpeg","-y","-v","error","-i",first(raw),"-an","-c:v","libx264","-preset","veryfast","-crf","20","-pix_fmt","yuv420p",seg/f"v{i:04}.mp4")
 files=sorted(seg.glob("v*.mp4")); concat=seg/"concat.txt"; concat.write_text("".join(f"file '{p.resolve()}'\n" for p in files)); visual=r/"visual.mp4"; final=r/"master.mp4"; run("ffmpeg","-y","-v","error","-f","concat","-safe","0","-i",concat,"-c","copy",visual); run("ffmpeg","-y","-v","error","-i",visual,"-i",audio,"-map","0:v:0","-map","1:a:0","-t",seconds,"-c:v","copy","-c:a","aac","-b:a","192k","-ar","48000","-movflags","+faststart",final)
 if final.stat().st_size<500000: raise RuntimeError("MASTER_NOT_CREATED")
 return final.read_bytes()
@app.local_entrypoint()
def main(seconds:int=180):
 out=Path("output"); out.mkdir(exist_ok=True); audio=out/"bhajan_source.mp3"; image=Path("assets/singer_image.png")
 if not audio.exists(): raise RuntimeError("output/bhajan_source.mp3 required")
 (out/"master.mp4").write_bytes(generate.remote(image.read_bytes(),audio.read_bytes(),seconds)); print("MODAL_VIDEO_READY",flush=True)
