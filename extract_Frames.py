import cv2, os

video_path = "YouTubeClips/sample.avi"
out_dir = "frames"
os.makedirs(out_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
interval = int(fps * 5)

count, idx = 0, 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if count % interval == 0:
        cv2.imwrite(f"{out_dir}/frame_{idx}.jpg", frame)
        idx += 1

    count += 1

cap.release()
print("Frames extracted")

# import torch
# import cv2
# import os
# from PIL import Image
# from transformers import Blip2Processor, Blip2ForConditionalGeneration

# device = "cuda" if torch.cuda.is_available() else "cpu"

# processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
# model = Blip2ForConditionalGeneration.from_pretrained(
#     "Salesforce/blip2-opt-2.7b",
#     torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
# )
# model.to(device)

# import torch
# from PIL import Image
# from transformers import Blip2Processor, Blip2ForConditionalGeneration

# # Check GPU
# device = "cuda" if torch.cuda.is_available() else "cpu"

# # Load BLIP-2 model + processor
# processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
# model = Blip2ForConditionalGeneration.from_pretrained(
#     "Salesforce/blip2-opt-2.7b",
#     torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
# )
# model.to(device)

# # Function to generate caption for a single image
# def caption_image(image_path):
#     image = Image.open(image_path).convert("RGB")
#     inputs = processor(images=image, return_tensors="pt").to(device)
#     out = model.generate(**inputs, max_new_tokens=30)
#     caption = processor.decode(out[0], skip_special_tokens=True)
#     return caption

import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# Check GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load BLIP-2 model + processor
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
model.to(device)

# Function to generate caption for a single image
def caption_image(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=30)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption



# def caption_image(image_path):
#     image = Image.open(image_path).convert("RGB")
#     inputs = processor(image, return_tensors="pt").to(device)
#     out = model.generate(**inputs, max_new_tokens=30)
#     caption = processor.decode(out[0], skip_special_tokens=True)
#     return caption

# import cv2, os, pickle
# from blip2_video_caption import caption_image

# # -------------------------
# # CONFIG
# # -------------------------
# video_path = "MSVD/YouTubeClips/sample.avi"
# frames_dir = "frames"
# os.makedirs(frames_dir, exist_ok=True)

# pkl_file = "video_captions.pkl"  # file to save/load captions
# fps_interval_sec = 5  # generate caption every 5 seconds

# # -------------------------
# # CHECK IF PICKLE EXISTS
# # -------------------------
# if os.path.exists(pkl_file):
#     with open(pkl_file, "rb") as f:
#         results = pickle.load(f)
#     print(f"[INFO] Loaded captions from pickle ({pkl_file})")
# else:
#     # -------------------------
#     # EXTRACT FRAMES AND GENERATE CAPTIONS
#     # -------------------------
#     cap = cv2.VideoCapture(video_path)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     interval = int(fps * fps_interval_sec)

#     frame_id = 0
#     segment = 0
#     results = []

#     print("[INFO] Extracting frames and generating captions...")
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         if frame_id % interval == 0:
#             # Save frame
#             img_path = os.path.join(frames_dir, f"frame_{segment}.jpg")
#             cv2.imwrite(img_path, frame)

#             # Generate caption using BLIP-2
#             caption = caption_image(img_path)
#             results.append((segment*fps_interval_sec, caption))
#             print(f"[{segment*fps_interval_sec}s] {caption}")

#             segment += 1

#         frame_id += 1

#     cap.release()

#     # -------------------------
#     # SAVE CAPTIONS TO PICKLE
#     # -------------------------
#     with open(pkl_file, "wb") as f:
#         pickle.dump(results, f)
#     print(f"[INFO] Captions saved to {pkl_file}")

# # -------------------------
# # DISPLAY RESULTS
# # -------------------------
# print("\n=== VIDEO CAPTIONS ===")
# for t, cap_text in results:
#     print(f"{t}s : {cap_text}")

# # -------------------------
# # SAVE TO TXT FILE
# # -------------------------
# txt_file = "video_captions.txt"
# with open(txt_file, "w") as f:
#     for t, cap_text in results:
#         f.write(f"{t}s : {cap_text}\n")

# print(f"[INFO] Captions saved to {txt_file}")

# import os
# import cv2
# import pickle
# from PIL import Image
# import torch
# from transformers import Blip2Processor, Blip2ForConditionalGeneration

# # -------------------------
# # CONFIG
# # -------------------------
# video_path = "MSVD/YouTubeClips/sample.avi"
# frames_dir = "frames"
# os.makedirs(frames_dir, exist_ok=True)

# pkl_file = "video_captions.pkl"
# fps_interval_sec = 5

# device = "cuda" if torch.cuda.is_available() else "cpu"
# processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
# model = Blip2ForConditionalGeneration.from_pretrained(
#     "Salesforce/blip2-opt-2.7b",
#     torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
# )
# model.to(device)

# def caption_image(image_path):
#     image = Image.open(image_path).convert("RGB")
#     inputs = processor(images=image, return_tensors="pt").to(device)
#     out = model.generate(**inputs, max_new_tokens=30)
#     caption = processor.decode(out[0], skip_special_tokens=True)
#     return caption

# # -------------------------
# # LOAD PICKLE IF EXISTS
# # -------------------------
# if os.path.exists(pkl_file):
#     with open(pkl_file, "rb") as f:
#         results = pickle.load(f)
#     print(f"[INFO] Loaded captions from pickle ({pkl_file})")
# else:
#     # -------------------------
#     # EXTRACT FRAMES AND GENERATE CAPTIONS
#     # -------------------------
#     cap = cv2.VideoCapture(video_path)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     interval = int(fps * fps_interval_sec)

#     frame_id = 0
#     segment = 0
#     results = []

#     print("[INFO] Extracting frames and generating captions...")
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         if frame_id % interval == 0:
#             img_path = os.path.join(frames_dir, f"frame_{segment}.jpg")
#             cv2.imwrite(img_path, frame)

#             caption = caption_image(img_path)
#             results.append((segment*fps_interval_sec, caption))
#             print(f"[{segment*fps_interval_sec}s] {caption}")

#             segment += 1

#         frame_id += 1

#     cap.release()

#     # Save captions to pickle
#     with open(pkl_file, "wb") as f:
#         pickle.dump(results, f)
#     print(f"[INFO] Captions saved to {pkl_file}")

# # -------------------------
# # DISPLAY RESULTS
# # -------------------------
# print("\n=== VIDEO CAPTIONS ===")
# for t, cap_text in results:
#     print(f"{t}s : {cap_text}")

# # -------------------------
# # SAVE TO TXT
# # -------------------------
# txt_file = "video_captions.txt"
# with open(txt_file, "w") as f:
#     for t, cap_text in results:
#         f.write(f"{t}s : {cap_text}\n")
# print(f"[INFO] Captions saved to {txt_file}")

import os
import cv2
import pickle
from PIL import Image
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# -------------------------
# CONFIG
# -------------------------
video_path = "MSVD/YouTubeClips/sample.avi"
frames_dir = "frames"
os.makedirs(frames_dir, exist_ok=True)

pkl_file = "video_captions.pkl"
fps_interval_sec = 5

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
model.to(device)

def caption_image(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=30)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# -------------------------
# LOAD PICKLE IF EXISTS
# -------------------------
if os.path.exists(pkl_file):
    with open(pkl_file, "rb") as f:
        results = pickle.load(f)
    print(f"[INFO] Loaded captions from pickle ({pkl_file})")
else:
    # -------------------------
    # EXTRACT FRAMES AND GENERATE CAPTIONS
    # -------------------------
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps * fps_interval_sec)

    frame_id = 0
    segment = 0
    results = []

    print("[INFO] Extracting frames and generating captions...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % interval == 0:
            img_path = os.path.join(frames_dir, f"frame_{segment}.jpg")
            cv2.imwrite(img_path, frame)

            caption = caption_image(img_path)
            results.append((segment*fps_interval_sec, caption))
            print(f"[{segment*fps_interval_sec}s] {caption}")

            segment += 1

        frame_id += 1

    cap.release()

    # Save captions to pickle
    with open(pkl_file, "wb") as f:
        pickle.dump(results, f)
    print(f"[INFO] Captions saved to {pkl_file}")

# -------------------------
# DISPLAY RESULTS
# -------------------------
print("\n=== VIDEO CAPTIONS ===")
for t, cap_text in results:
    print(f"{t}s : {cap_text}")

# -------------------------
# SAVE TO TXT
# -------------------------
txt_file = "video_captions.txt"
with open(txt_file, "w") as f:
    for t, cap_text in results:
        f.write(f"{t}s : {cap_text}\n")
print(f"[INFO] Captions saved to {txt_file}")



