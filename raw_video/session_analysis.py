import os
import cv2
import torch
import clip  # Change this line
import ffmpeg
import numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import GPT2TokenizerFast
import openai
from dotenv import load_dotenv

load_dotenv()

# ========== Config ==========
VIDEO_PATH = "videos/test1.mp4"
FRAME_DIR = "frames"
YOLO_MODEL = "yolov8n.pt"
FRAME_RATE = 1  # Extract 1 frame per second
CLIP_MODEL_NAME = "ViT-B/32"
openai.api_key = os.getenv("OPENAI_API_KEY")
QUERY = "Where does the user encounter a problem?"

# ========== Step 1: Extract frames ==========
def extract_frames(video_path, output_dir, fps=1):
    os.makedirs(output_dir, exist_ok=True)
    
    # Use OpenCV instead of ffmpeg
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps)  # Extract every nth frame
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            output_path = os.path.join(output_dir, f"frame_{saved_count:04d}.png")
            cv2.imwrite(output_path, frame)
            saved_count += 1
            
        frame_count += 1
    
    cap.release()
    print(f"Extracted {saved_count} frames from video")

# ========== Step 2: Run YOLO detection ==========
def detect_objects_on_frames(frame_dir):
    model = YOLO(YOLO_MODEL)
    results = []
    for frame_file in sorted(os.listdir(frame_dir)):
        path = os.path.join(frame_dir, frame_file)
        result = model(path)[0]
        results.append((frame_file, result))
    return results

# ========== Step 3: Load CLIP ==========
def load_clip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(CLIP_MODEL_NAME, device=device)
    return model, preprocess, device

# ========== Step 4: Embed with CLIP ==========
def embed_frame_caption_pairs(results, frame_dir, clip_model, preprocess, device):
    frame_embeddings = []
    for frame_file, result in results:
        path = os.path.join(frame_dir, frame_file)
        image = preprocess(Image.open(path)).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = clip_model.encode_image(image).cpu().numpy()
        frame_embeddings.append((frame_file, image_features))
    return frame_embeddings

# ========== Step 5: Analyze frames with GPT-4V ==========
def analyze_with_gpt4v(frame_files, frame_dir, max_frames=10):
    """Analyze frames directly with GPT-4V for better UX insights"""
    
    # Limit frames to avoid token limits and costs
    frame_files = frame_files[:max_frames]
    
    # Prepare images for GPT-4V
    images = []
    for frame_file in frame_files:
        image_path = os.path.join(frame_dir, frame_file)
        # Convert to base64 for API
        import base64
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        images.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{encoded_image}"
            }
        })
    
    # Create content with images and text
    content = [
        {
            "type": "text",
            "text": f"You are an expert UX analyst. Analyze these {len(images)} frames from a user session and answer: {QUERY}\n\nPlease provide detailed insights about user behavior, potential issues, and UX improvements."
        }
    ]
    content.extend(images)
    
    # Send to GPT-4V
    client = openai.OpenAI()
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": content
            }
        ],
        max_tokens=4000
    )
    return completion.choices[0].message.content

# ========== Run Pipeline ==========
def main():
    print("📦 Extracting frames...")
    extract_frames(VIDEO_PATH, FRAME_DIR, fps=FRAME_RATE)

    print("🔎 Running YOLO detection...")
    yolo_results = detect_objects_on_frames(FRAME_DIR)

    print("🧠 Embedding frames with CLIP...")
    clip_model, preprocess, device = load_clip()
    _ = embed_frame_caption_pairs(yolo_results, FRAME_DIR, clip_model, preprocess, device)

    print("📈 Analyzing frames with GPT-4V...")
    frame_files = [f for f, _ in yolo_results]
    gpt_output = analyze_with_gpt4v(frame_files, FRAME_DIR, max_frames=20)

    print("📝 GPT-4V Analysis:\n", gpt_output)

if __name__ == "__main__":
    main()
