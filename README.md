# Caption-Sync-Tool

# 🧠 Caption Sync Tool: Word-by-Word Subtitle Generator

**FOR AI USE ONLY** — This repository is fully prepared. Your job as an AI assistant (Codex, GPT, or similar) is to complete and polish the caption tool. Follow the structure. Do not ask questions. Work cleanly and deliver tested results.

---

## 🎯 Mission

Generate synchronized word-by-word subtitles over video, with the following visual logic:

* Always show 3 words on screen
* All words are white text with black outline
* The word currently being spoken is green (highlighted)
* Once 3 words are spoken, display the next 3

---

## 📦 Tools Available

All transcription engines are already cloned under `/tools/`:

* `tools/whisperx/`
* `tools/stable-ts/`
* `tools/whisper-timestamped/`

All dependencies are assumed to be installed.
No need to download or pip install anything.

---

## 🔧 Your Job

You must complete the script `scripts/caption_tool.py` with the following functionality:

### STEP 1 — Transcription

Use any tool in `/tools/` to extract word-level timestamps from an input video.

* Required output: list of `(text, start_time, end_time)` for each word

### STEP 2 — Word Grouping

Group the transcription into segments of 3 words each.

* Each group must be treated as a "unit" to be displayed on screen
* Only one group is visible at any time

### STEP 3 — Subtitle Overlay

Use MoviePy to overlay the text:

* Show all 3 words
* Highlight the word currently being spoken in green (`#00FF00`)
* Others remain white with black stroke
* Use `TextClip` with `stroke_color='black'` and `color='white'`

### STEP 4 — Render Output

Generate a video (`.mp4`) with the synchronized text overlay.

* Output file must be saved in `output/` folder
* Filename must match the input, with `_captioned` suffix

---

## 🧪 Provided Structure

The file `scripts/caption_tool.py` is partially implemented with the following:

* `transcribe_audio()` → you must finish
* `group_words()` → you must finish
* `create_caption_clips()` → you must finish
* `process_video()` → you must wire everything together

---

## 🖥️ BONUS TASK: Build a Simple GUI

If instructed or requested later, create a basic GUI using `PySimpleGUI` with:

* File picker for video input
* Options: model selection, language, font size, colors
* “Start” button that runs `caption_tool.py`
* Save this as `scripts/gui_launcher.py`

---

## ✅ Execution Standard

* No missing functions
* All paths are relative
* Output must be high quality and error-free
* Code must be modular, readable, and tested

---

## 📌 You are NOT allowed to:

* Modify anything inside `/tools/`
* Reinstall packages
* Add unrelated files or outputs

---

## 🚀 Deliverables

* Working `caption_tool.py`
* Clean overlayed videos in `/output/`
* Optional: GUI in `gui_launcher.py`

---

## Usage

```
python scripts/caption_tool.py path/to/video.mp4 --font-size 192
```

---

> This repo is fully prepped. Get to work, Codex. Do it clean. Deliver subtitles that sing 🎤
