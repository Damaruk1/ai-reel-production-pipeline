#!/usr/bin/env python3
"""
Model 3 - Production Single-File (Hybrid Mode B, Hybrid I/O)

Features
- Input: YouTube URL or local file
- Transcript: YouTubeTranscriptApi -> yt-dlp VTT -> Whisper (local) fallback
- Summarization & prompt engineering: Gemini (google.generativeai) when available, otherwise smart extractive fallback
- Voice: ElevenLabs API for TTS, fallback synthetic silence
- Visuals: Imagine.Art video endpoint, fallback to Imagine images -> slideshow, fallback to source clip cropping
- Compose: ffmpeg to combine visuals + narration into vertical 9:16 output
- Subtitles: YouTube-style bottom SRT aligned to TTS segments
- Tone/Duration/Voice/Watermark asked interactively if flags omitted (hybrid mode)
- Robust fallbacks and zero-crash design; always try to output something
- One file; no front-end; ready to be wrapped via API

Required system tools
- ffmpeg on PATH
- yt-dlp on PATH (for YouTube downloads and subtitle fetch)
- Optional: whisper (pip install -U openai-whisper), opencv-python-headless, sentence-transformers, scikit-learn

Environment variables (optional but recommended)
- GEMINI_API_KEY
- ELEVEN_API_KEY
- IMAGINE_API_KEY
- ELEVEN_VOICE_ID

Install approximate Python deps
pip install yt-dlp youtube-transcript-api google-generativeai sentence-transformers scikit-learn numpy requests opencv-python-headless Pillow openai-whisper
"""

import os
import re
import sys
import json
import time
import base64
import math
import shutil
import tempfile
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import requests

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY", "")
IMAGINE_API_KEY = os.getenv("IMAGINE_API_KEY", "")
ELEVEN_VOICE_ID = os.getenv("ELEVEN_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")

FPS = 12
READING_WPS = 3.2

try:
    from youtube_transcript_api import YouTubeTranscriptApi
except Exception:
    YouTubeTranscriptApi = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    from sklearn.cluster import AgglomerativeClustering
except Exception:
    AgglomerativeClustering = None

try:
    import numpy as np
except Exception:
    np = None

try:
    import cv2
except Exception:
    cv2 = None

def which(bin_name: str) -> Optional[str]:
    from shutil import which as _which
    return _which(bin_name)

def run_cmd(cmd: List[str], check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=capture, text=True)
    if check and proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{proc.stderr.strip()}")
    return proc

def ffprobe_duration(path: str) -> float:
    if which("ffprobe") is None:
        return 0.0
    p = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path)
    ], capture_output=True, text=True)
    try:
        return float(p.stdout.strip())
    except Exception:
        return 0.0

def extract_video_id(url: str) -> str:
    if "v=" in url:
        return url.split("v=")[-1].split("&")[0]
    if "youtu.be/" in url:
        return url.split("youtu.be/")[-1].split("?")[0]
    if "/shorts/" in url:
        return url.split("/shorts/")[-1].split("?")[0]
    cleaned = re.sub(r"[^A-Za-z0-9\-_]", "", url)
    return (cleaned[:11] or "video")

def safe_mkdir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)

def download_youtube(url: str, out_path: str) -> Optional[str]:
    try:
        Path(Path(out_path).parent).mkdir(parents=True, exist_ok=True)
        cmd = ["yt-dlp", "-f", "bestvideo[ext=mp4]+bestaudio/best", "-o", out_path, url]
        run_cmd(cmd, check=True)
        if Path(out_path).exists():
            return out_path
        return None
    except Exception:
        return None

def fetch_transcript_from_youtube_api(url: str) -> Tuple[str, List[Dict]]:
    vid = extract_video_id(url)
    if YouTubeTranscriptApi is None:
        return "No transcript", []
    try:
        lts = YouTubeTranscriptApi.list_transcripts(vid)
        try:
            tr = lts.find_transcript(["en", "en-US", "en-IN", "hi"])
        except Exception:
            tr = lts.find_generated_transcript(["en", "en-US", "en-IN", "hi"])
        raw = tr.fetch()
        segs = []
        for it in raw:
            start = float(it.get("start", 0.0))
            dur = float(it.get("duration", 0.0))
            text = (it.get("text") or "").strip()
            if not text:
                continue
            segs.append({"start": start, "end": start + dur, "text": text})
        full = " ".join(s["text"] for s in segs)
        return full, segs
    except Exception:
        return "No transcript", []

def fetch_vtt_with_ytdlp(url: str, outdir: str) -> Optional[str]:
    outdirp = Path(outdir)
    outdirp.mkdir(parents=True, exist_ok=True)
    template = str(outdirp / "%(id)s")
    cmd = [
        "yt-dlp", "--skip-download", "--no-warnings", "--no-playlist",
        "--write-auto-sub", "--write-subs",
        "--sub-langs", "en.*,en,en-US,en-IN,hi",
        "--sub-format", "vtt", "--convert-subs", "vtt",
        "-o", template, url
    ]
    try:
        run_cmd(cmd, check=False)
    except Exception:
        pass
    vtts = list(outdirp.glob("*.vtt"))
    return str(vtts[0]) if vtts else None

def parse_vtt(vtt_path: str) -> List[Dict]:
    text = Path(vtt_path).read_text(encoding="utf-8")
    blocks = [b.strip() for b in re.split(r"\n{2,}", text) if "-->" in b]
    segs = []
    for b in blocks:
        lines = [l for l in b.splitlines() if l.strip()]
        times = None
        texts = []
        for ln in lines:
            if "-->" in ln:
                times = ln.strip()
            elif not ln.strip().isdigit():
                texts.append(ln.strip())
        if not times:
            continue
        start_s, end_s = [t.strip() for t in times.split("-->")]
        def to_seconds(x: str) -> float:
            p = x.split(":")
            if len(p) == 3:
                h = int(p[0]); m = int(p[1]); s = float(p[2])
                return h * 3600 + m * 60 + s
            return 0.0
        segs.append({"start": to_seconds(start_s), "end": to_seconds(end_s), "text": " ".join(texts)})
    return segs

def fetch_transcript(url: str, tmpdir: str) -> Tuple[str, List[Dict]]:
    full, segs = fetch_transcript_from_youtube_api(url)
    if segs:
        return full, segs
    vtt = fetch_vtt_with_ytdlp(url, tmpdir)
    if vtt:
        try:
            segs2 = parse_vtt(vtt)
            full2 = " ".join(s["text"] for s in segs2)
            if segs2:
                return full2, segs2
        except Exception:
            pass
    try:
        import whisper
        model = whisper.load_model("base")
        vidtmp = str(Path(tmpdir) / "yt_download_forsub.mp4")
        dl = download_youtube(url, vidtmp)
        if dl:
            res = model.transcribe(vidtmp, verbose=False)
            sgs = []
            for seg in res.get("segments", []):
                sgs.append({"start": float(seg["start"]), "end": float(seg["end"]), "text": seg["text"].strip()})
            full3 = " ".join(s["text"] for s in sgs)
            if sgs:
                return full3, sgs
    except Exception:
        pass
    return "No transcript available.", [{"start": 0.0, "end": 5.0, "text": "No transcript available."}]

def embed_cluster_segments(segs: List[Dict]) -> List[Dict]:
    texts = [s.get("text", "") for s in segs]
    if SentenceTransformer is None or AgglomerativeClustering is None or np is None:
        merged = []
        cur = None
        for s in segs:
            if cur is None:
                cur = dict(s)
            else:
                cur["end"] = s["end"]
                cur["text"] += " " + s["text"]
            if cur and (cur["end"] - cur["start"]) >= 12:
                merged.append(cur)
                cur = None
        if cur:
            merged.append(cur)
        return merged
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    nc = max(2, min(12, max(2, len(texts) // 6)))
    cl = AgglomerativeClustering(n_clusters=nc, affinity="cosine", linkage="average")
    labels = cl.fit_predict(embeddings)
    groups = {}
    for idx, lab in enumerate(labels):
        groups.setdefault(int(lab), []).append(segs[idx])
    merged = []
    for g in groups.values():
        g = sorted(g, key=lambda x: x["start"])
        merged.append({"start": float(g[0]["start"]), "end": float(g[-1]["end"]), "duration": float(g[-1]["end"] - g[0]["start"]), "text": " ".join(i["text"] for i in g)})
    merged = sorted(merged, key=lambda x: x["start"])
    filtered = [m for m in merged if m.get("duration", 0) >= 6.0]
    return filtered or merged

def score_for_summary(seg: Dict) -> float:
    txt = (seg.get("text") or "").strip()
    words = len(re.findall(r"\w+", txt))
    dur = max(0.01, seg.get("duration", seg.get("end", 0.0) - seg.get("start", 0.0)))
    density = words / dur
    hooks = 0
    phrases = ["you must", "here's why", "three things", "the reason", "in this video", "don't", "never", "always", "best"]
    low = txt.lower()
    for p in phrases:
        if p in low:
            hooks += 1
    punct = sum(1 for ch in txt if ch in "!?") / max(1, len(txt))
    score = density * (1 + 0.45 * hooks) * (1 + punct)
    if np is not None:
        sentence_count = max(1, len(re.split(r"[.!?]+", txt)))
        score *= (1 + np.log1p(sentence_count))
    return float(score)

def pick_best_segments(segs: List[Dict], top_k: int = 1) -> List[Dict]:
    candidates = []
    for s in segs:
        s2 = dict(s)
        s2["duration"] = s2.get("duration", s2.get("end", s2.get("start", 0.0)) - s2.get("start", 0.0))
        if s2["duration"] < 6.0:
            continue
        s2["score"] = score_for_summary(s2)
        candidates.append(s2)
    candidates_sorted = sorted(candidates, key=lambda x: x["score"], reverse=True)
    return candidates_sorted[:top_k] if candidates_sorted else (segs[:top_k] if segs else [])

def gemini_generate_summary_and_prompts(transcript: str, target_seconds: int, tone: str, voice_style: str) -> Tuple[str, List[str]]:
    fallback = transcript[:800] if transcript else "Generated summary."
    prompt_fallback = [fallback[:120]]
    if not GEMINI_API_KEY:
        return fallback, prompt_fallback
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        instr = f"""You are a short-form creator assistant. Produce a concise narration for a {target_seconds}-second reel in a {tone} tone and {voice_style} voice style. Start with a strong hook, include 2 short points, and finish with a memorable closing line. Keep sentences short and punchy."""
        resp = model.generate_content(instr + "\n\nTranscript:\n\n" + transcript[:9000])
        summary = getattr(resp, "text", None) or fallback
        prompt_req = "From this narration produce 3 concise cinematic visual prompts suitable for text-to-video/image generation, one per line. Use mood words and visual anchors."
        resp2 = model.generate_content(prompt_req + "\n\n" + summary)
        ptxt = getattr(resp2, "text", "") or ""
        prompts = [ln.strip().lstrip("-• ") for ln in ptxt.splitlines() if ln.strip()]
        if not prompts:
            prompts = prompt_fallback
        return summary.strip(), prompts[:3]
    except Exception:
        return fallback, prompt_fallback

def clean_summary_for_tts(summary: str, max_words: int) -> str:
    words = summary.split()
    if len(words) <= max_words:
        return summary
    return " ".join(words[:max_words])

def eleven_tts_text(text: str, out_wav: str, voice_id: str = ELEVEN_VOICE_ID) -> float:
    if not ELEVEN_API_KEY:
        words = max(1, len(re.findall(r"\w+", text)))
        est = max(0.5, float(words) / READING_WPS)
        if which("ffmpeg"):
            run_cmd([
                "ffmpeg", "-y", "-f", "lavfi",
                "-i", "anullsrc=channel_layout=mono:sample_rate=44100",
                "-t", f"{est:.2f}", "-c:a", "pcm_s16le", out_wav
            ], check=False)
            return ffprobe_duration(out_wav)
        return est
    headers = {"xi-api-key": ELEVEN_API_KEY, "Content-Type": "application/json"}
    payload = {"text": text, "voice_settings": {"stability": 0.5, "similarity_boost": 0.9}}
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=60)
        r.raise_for_status()
        Path(out_wav).write_bytes(r.content)
        d = ffprobe_duration(out_wav)
        return d if d > 0.01 else float(max(0.5, len(text.split()) / READING_WPS))
    except Exception:
        words = max(1, len(re.findall(r"\w+", text)))
        est = max(0.5, float(words) / READING_WPS)
        if which("ffmpeg"):
            run_cmd([
                "ffmpeg", "-y", "-f", "lavfi",
                "-i", "anullsrc=channel_layout=mono:sample_rate=44100",
                "-t", f"{est:.2f}", "-c:a", "pcm_s16le", out_wav
            ], check=False)
            return ffprobe_duration(out_wav)
        return est

def split_into_short_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    parts = [p.strip() for p in parts if p.strip()]
    if not parts:
        return [text.strip()]
    return parts

def build_tts_segments(sentences: List[str], tmpdir: str, voice_id: str) -> Tuple[str, List[Dict]]:
    concat_list = Path(tmpdir) / "concat.txt"
    segs = []
    cur = 0.0
    with open(concat_list, "w", encoding="utf-8") as f:
        for i, s in enumerate(sentences):
            sfile = str(Path(tmpdir) / f"seg_{i:03d}.wav")
            dur = eleven_tts_text(s, sfile, voice_id)
            f.write(f"file '{Path(sfile).resolve()}'\n")
            segs.append({"text": s, "file": sfile, "duration": float(dur), "start": float(cur)})
            cur += float(dur)
    final = str(Path(tmpdir) / "narration.wav")
    if which("ffmpeg"):
        try:
            run_cmd(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list), "-c:a", "pcm_s16le", final], check=False)
            if Path(final).exists() and Path(final).stat().st_size > 0:
                return final, segs
        except Exception:
            pass
    if segs:
        return segs[0]["file"], segs
    silent = str(Path(tmpdir) / "silent.wav")
    eleven_tts_text(" ", silent, voice_id)
    return silent, [{"text": "", "file": silent, "duration": 5.0, "start": 0.0}]

def imagine_try_video(prompts: List[str], tmpdir: str, duration: float, fps: int = FPS) -> Optional[str]:
    if not IMAGINE_API_KEY:
        return None
    headers = {"Authorization": f"Bearer {IMAGINE_API_KEY}", "Content-Type": "application/json"}
    for i, p in enumerate(prompts):
        payload = {"prompt": p, "duration": int(max(4, min(30, math.ceil(duration)))), "fps": int(fps), "format": "mp4"}
        try:
            r = requests.post("https://api.imagine.art/v1/videos/generate", json=payload, headers=headers, timeout=300)
            if r.status_code != 200:
                continue
            data = r.json()
            dl = data.get("download_url") or data.get("result_url") or data.get("url")
            if dl:
                r2 = requests.get(dl, timeout=300)
                if r2.status_code == 200:
                    outp = str(Path(tmpdir) / f"imagined_vid_{i}.mp4")
                    Path(outp).write_bytes(r2.content)
                    return outp
            images = data.get("images") or []
            if images:
                img_files = []
                for j, b64 in enumerate(images):
                    img = base64.b64decode(b64)
                    pth = Path(tmpdir) / f"img_{i}_{j}.png"
                    pth.write_bytes(img)
                    img_files.append(str(pth))
                if img_files:
                    slideshow = str(Path(tmpdir) / f"imagined_slide_{i}.mp4")
                    make_slideshow_from_images(img_files, duration, slideshow, fps=fps)
                    return slideshow
        except Exception:
            continue
    return None

def imagine_try_images(prompts: List[str], tmpdir: str) -> List[str]:
    if not IMAGINE_API_KEY:
        return []
    headers = {"Authorization": f"Bearer {IMAGINE_API_KEY}", "Content-Type": "application/json"}
    saved = []
    for i, p in enumerate(prompts):
        payload = {"prompt": p, "size": "1024x1024", "n": 1}
        try:
            r = requests.post("https://api.imagine.art/v1/images/generate", json=payload, headers=headers, timeout=120)
            if r.status_code != 200:
                continue
            data = r.json()
            images = data.get("images") or []
            for j, b64 in enumerate(images):
                img = base64.b64decode(b64)
                pth = Path(tmpdir) / f"img_{i}_{j}.png"
                pth.write_bytes(img)
                saved.append(str(pth))
        except Exception:
            continue
    return saved

def make_slideshow_from_images(images: List[str], duration: float, out_mp4: str, fps: int = FPS):
    if which("ffmpeg") is None:
        raise RuntimeError("ffmpeg required for slideshow")
    n = len(images)
    if n == 0:
        raise RuntimeError("No images")
    per = max(0.5, duration / n)
    tmpdir = tempfile.mkdtemp(prefix="sl_")
    try:
        filelist = Path(tmpdir) / "imgs.txt"
        with open(filelist, "w", encoding="utf-8") as f:
            for img in images:
                f.write(f"file '{Path(img).resolve()}'\n")
                f.write(f"duration {per:.3f}\n")
            f.write(f"file '{Path(images[-1]).resolve()}'\n")
        cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(filelist), "-vsync", "vfr", "-pix_fmt", "yuv420p", "-r", str(fps), out_mp4]
        run_cmd(cmd, check=True)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

def detect_face_box_frame(video_path: str, sec: float = 0.5) -> Optional[tuple]:
    if cv2 is None:
        return None
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_no = int(max(0.0, sec) * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cascade = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        return None
    largest = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)[0]
    x, y, w, h = largest
    return (int(x), int(y), int(x + w), int(y + h))

def build_crop_filter_9_16(video_file: str, face_box: Optional[tuple] = None) -> str:
    p = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", video_file], capture_output=True, text=True)
    try:
        w, h = [int(x) for x in p.stdout.strip().split(",")]
    except Exception:
        w, h = 1280, 720
    target_aspect = 9.0 / 16.0
    cur_aspect = float(w) / float(h)
    if cur_aspect > target_aspect:
        new_h = h
        new_w = int(h * target_aspect)
    else:
        new_w = w
        new_h = int(w / target_aspect)
    if face_box:
        fx1, fy1, fx2, fy2 = face_box
        face_cx = (fx1 + fx2) // 2
        face_cy = (fy1 + fy2) // 2
        x = max(0, int(face_cx - new_w // 2))
        y = max(0, int(face_cy - new_h // 2))
        x = min(max(0, x), max(0, w - new_w))
        y = min(max(0, y), max(0, h - new_h))
    else:
        x = max(0, (w - new_w) // 2)
        y = max(0, (h - new_h) // 2)
    return f"crop={new_w}:{new_h}:{x}:{y},scale=1080:1920"

def crop_source_clip(source_video: str, start: float, end: float, out_path: str, prioritize_face: bool = True) -> str:
    if which("ffmpeg") is None:
        raise RuntimeError("ffmpeg required")
    tmp = tempfile.mkdtemp(prefix="cut_")
    try:
        clip_tmp = str(Path(tmp) / "clip_part.mp4")
        cmd = ["ffmpeg", "-y", "-ss", str(start), "-to", str(end), "-i", source_video, "-c", "copy", clip_tmp]
        try:
            run_cmd(cmd, check=True)
        except Exception:
            cmd2 = ["ffmpeg", "-y", "-i", source_video, "-ss", str(start), "-to", str(end), "-c", "copy", clip_tmp]
            run_cmd(cmd2, check=True)
        face_box = None
        if prioritize_face:
            try:
                face_box = detect_face_box_frame(clip_tmp, sec=max(0.2, (end - start) / 4.0))
            except Exception:
                face_box = None
        vf = build_crop_filter_9_16(clip_tmp, face_box)
        cmd_crop = [
            "ffmpeg", "-y", "-i", clip_tmp, "-vf", vf, "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-c:a", "aac", "-movflags", "+faststart", out_path
        ]
        run_cmd(cmd_crop, check=True)
        return out_path
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def compose_visual_and_audio(visual: str, audio: str, duration: float, out_file: str, watermark: Optional[str] = None):
    if which("ffmpeg") is None:
        raise RuntimeError("ffmpeg required")
    vf = "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2"
    cmd = ["ffmpeg", "-y", "-i", visual, "-i", audio, "-t", f"{duration:.2f}", "-vf", vf]
    if watermark:
        wm = f"drawtext=text='{watermark}':fontcolor=white:fontsize=24:x=w-tw-10:y=h-th-10:box=1:boxcolor=black@0.4"
        cmd[-1] = vf + "," + wm
    cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", "20", "-c:a", "aac", "-shortest", out_file]
    run_cmd(cmd, check=True)

def make_srt_from_tts_segs(segs: List[Dict], out_srt: str):
    with open(out_srt, "w", encoding="utf-8") as f:
        for i, s in enumerate(segs, start=1):
            start = float(s.get("start", 0.0))
            dur = float(s.get("duration", 0.0))
            end = start + dur
            def fmt(t):
                h = int(t // 3600); m = int((t % 3600) // 60); sec = int(t % 60); ms = int((t - int(t)) * 1000)
                return f"{h:02}:{m:02}:{sec:02},{ms:03}"
            text = (s.get("text") or "").strip()
            f.write(f"{i}\n{fmt(start)} --> {fmt(end)}\n{text}\n\n")

def normalize_and_compress_audio(in_wav: str, out_wav: str):
    if which("ffmpeg") is None:
        shutil.copy(in_wav, out_wav)
        return out_wav
    cmd = ["ffmpeg", "-y", "-i", in_wav, "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-c:a", "aac", "-b:a", "128k", out_wav]
    run_cmd(cmd, check=True)
    return out_wav

def pipeline_main(source: str, out_dir: str, duration_target: int, tone: str, voice_style: str, watermark: Optional[str], voice_id: str, fps: int = FPS) -> Dict:
    safe_mkdir(out_dir)
    tmp = tempfile.mkdtemp(prefix="m3work_")
    try:
        downloaded = None
        if str(source).startswith("http"):
            transcript_text, segs = fetch_transcript(source, tmp)
            downloaded = download_youtube(source, str(Path(tmp)/f"{extract_video_id(source)}.mp4"))
        else:
            if not Path(source).exists():
                raise RuntimeError("Local file not found")
            try:
                transcript_text, segs = fetch_transcript_from_youtube_api(extract_video_id(source))
                if not segs:
                    transcript_text, segs = ("", [{"start":0.0,"end":ffprobe_duration(source),"text":""}])
            except Exception:
                transcript_text, segs = ("", [{"start":0.0,"end":ffprobe_duration(source),"text":""}])
            downloaded = source
        if not transcript_text or transcript_text.strip() == "No transcript available.":
            try:
                import whisper
                model = whisper.load_model("base")
                audio_source = downloaded if downloaded else None
                if audio_source is None and str(source).startswith("http"):
                    try:
                        audio_dl = str(Path(tmp)/"dl_video.mp4")
                        download_youtube(source, audio_dl)
                        audio_source = audio_dl
                    except Exception:
                        audio_source = None
                if audio_source:
                    res = model.transcribe(audio_source, verbose=False)
                    sgs = []
                    for seg in res.get("segments", []):
                        sgs.append({"start": float(seg["start"]), "end": float(seg["end"]), "text": seg["text"].strip()})
                    transcript_text = " ".join(s["text"] for s in sgs)
                    segs = sgs or segs
            except Exception:
                pass
        clusters = embed_cluster_segments(segs)
        best = pick_best_segments(clusters, top_k=1)
        seed_seg = best[0] if best else (clusters[0] if clusters else {"start":0,"end":min(30, ffprobe_duration(downloaded) if downloaded else 30),"text":transcript_text})
        summary, prompts = gemini_generate_summary_and_prompts(transcript_text or seed_seg.get("text",""), duration_target, tone, voice_style)
        max_words = int(max(10, duration_target * READING_WPS))
        narr_text = clean_summary_for_tts(summary, max_words)
        sentences = split_into_short_sentences(narr_text)
        narration_wav, tts_segs = build_tts_segments(sentences, tmp, voice_id)
        norm_wav = str(Path(tmp)/"narration_norm.m4a")
        normalize_and_compress_audio(narration_wav, norm_wav)
        audio_duration = ffprobe_duration(norm_wav)
        visual = None
        try:
            visual = imagine_try_video(prompts, tmp, audio_duration, fps=fps)
        except Exception:
            visual = None
        if not visual:
            imgs = imagine_try_images(prompts, tmp)
            if imgs:
                slide = str(Path(tmp)/"slide.mp4")
                try:
                    make_slideshow_from_images(imgs, audio_duration, slide, fps=fps)
                    visual = slide
                except Exception:
                    visual = None
        if not visual and downloaded:
            try:
                clip_out = str(Path(tmp)/"source_clip.mp4")
                sstart = float(seed_seg.get("start", 0.0))
                send = float(seed_seg.get("end", sstart + min(duration_target, 30)))
                crop_source_clip(downloaded, sstart, send, clip_out, prioritize_face=True)
                visual = clip_out
            except Exception:
                visual = None
        if not visual:
            raise RuntimeError("Unable to generate visuals or source clip")
        final_mp4 = str(Path(out_dir)/"summary_reel_final.mp4")
        compose_visual_and_audio(visual, norm_wav, audio_duration, final_mp4, watermark=watermark)
        srt_out = str(Path(out_dir)/"summary_reel.srt")
        make_srt_from_tts_segs(tts_segs, srt_out)
        metadata = {
            "source": source,
            "video": final_mp4,
            "srt": srt_out,
            "duration": audio_duration,
            "summary": summary,
            "prompts": prompts,
            "tone": tone,
            "voice_style": voice_style
        }
        Path(str(Path(out_dir)/"metadata.json")).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return metadata
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def parse_cli_or_interactive():
    p = argparse.ArgumentParser()
    p.add_argument("--url", help="YouTube URL")
    p.add_argument("--file", help="Local video file")
    p.add_argument("--out", default="outputs/model3", help="Output directory")
    p.add_argument("--duration", type=int, help="Target duration in seconds (30-90)")
    p.add_argument("--tone", help="Tone: cinematic / educational / news / ted / story")
    p.add_argument("--voice", help="Voice style: energetic / smooth / deep / soft")
    p.add_argument("--watermark", help="Watermark text or 'no'", default="no")
    p.add_argument("--voice_id", help="ElevenLabs voice id", default=ELEVEN_VOICE_ID)
    p.add_argument("--fps", type=int, default=FPS, help="FPS for generated video")
    args = p.parse_args()
    source = args.file if args.file else args.url
    if not source:
        source = input("Enter YouTube URL or local file path: ").strip()
    duration = args.duration
    if duration is None:
        while True:
            try:
                s = input("Target reel duration in seconds (30-90, default 40): ").strip() or "40"
                duration = int(s)
                if 20 <= duration <= 120:
                    break
            except Exception:
                continue
    tone = args.tone or input("Tone (cinematic / educational / news / ted / story) [cinematic]: ").strip() or "cinematic"
    voice_style = args.voice or input("Voice style (energetic / smooth / deep / soft) [energetic]: ").strip() or "energetic"
    watermark = None
    if args.watermark and args.watermark.lower() != "no":
        watermark = args.watermark
    elif args.watermark.lower() == "no":
        watermark = None
    else:
        w = input("Watermark text (enter to skip): ").strip()
        watermark = w if w else None
    out = args.out
    safe_mkdir(out)
    return source, out, duration, tone, voice_style, watermark, args.voice_id, args.fps

def main():
    source, out, duration, tone, voice_style, watermark, voice_id, fps = parse_cli_or_interactive()
    start = time.time()
    try:
        meta = pipeline_main(source, out, duration, tone, voice_style, watermark, voice_id, fps=fps)
        print("Success. Metadata written to", str(Path(out)/"metadata.json"))
        print(json.dumps(meta, indent=2))
    except Exception as e:
        print("Pipeline failed:", str(e))
        sys.exit(1)
    finally:
        print("Elapsed:", time.time() - start)

if __name__ == "__main__":
    main()
