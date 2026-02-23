# AI Reel Production Pipeline

## Overview
End-to-end AI system that converts long-form videos into short-form vertical reels with narration, subtitles, and visuals.

## Pipeline Stages
1. Transcript extraction (YouTube API → VTT → Whisper fallback)
2. Semantic clustering of transcript segments
3. Hook-based segment scoring
4. AI narration generation
5. Text-to-speech (ElevenLabs or silent fallback)
6. Visual generation (Imagine API → slideshow → source clip fallback)
7. 9:16 video composition with FFmpeg
8. Subtitle alignment (SRT)

## Features
- Fully automated reel generation
- Hybrid AI + rule-based fallback design
- Face-aware vertical cropping
- API-optional architecture (still produces output without keys)

## Requirements
- Python 3.9+
- FFmpeg installed and added to PATH
- yt-dlp installed

## Environment Variables
Copy `.env.example` and add your keys if available:
- GEMINI_API_KEY
- ELEVEN_API_KEY
- IMAGINE_API_KEY
- ELEVEN_VOICE_ID

## How to Run

Local video:
python model3_pipeline.py --file input.mp4 --duration 40

YouTube:
python model3_pipeline.py --url "https://youtube.com/..." --duration 40

## Output
Generated files are stored in the `outputs/` directory:
- summary_reel_final.mp4
- summary_reel.srt
- metadata.json

## Note
Outputs and large media files are not included in this repository.