# Kite

**Kite** is a tool to analyze your session recordings.

Note: this is a crawling out of the slime version. Will add integrations with real session providers. The idea is to go from session recording to bug fix.

## Features

- Extracts frames from your video session recordings
- Detects UI elements in each frame using YOLO
- Embeds frames with CLIP for semantic analysis
- Summarizes and analyzes user experience with GPT-4 or GPT-4V (multimodal)
- Helps you identify where users encounter problems in your product

## Usage

1. Place your session video in the `videos/` directory.
2. Update the `VIDEO_PATH` in `session_analysis.py` if needed.
3. Run the analysis:

   ```bash
   python session_analysis.py
   ```

4. Review the output for UX insights and problem areas.

## Requirements

- Python 3.8+
- [See `requirements.txt` for dependencies]

## About

Kite helps you understand user behavior and improve your product by analyzing session recordings with state-of-the-art AI models.
