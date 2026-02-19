# import os
# import cv2
# import pickle
# import torch
# from PIL import Image
# from transformers import Blip2Processor, Blip2ForConditionalGeneration

# # =========================
# # CONFIG
# # =========================
# video_path = "MSVD/YouTubeClips/sample.avi"
# frames_dir = "frames"
# os.makedirs(frames_dir, exist_ok=True)

# pkl_file = "video_captions.pkl"
# txt_file = "video_captions.txt"

# fps_interval_sec = 5

# # 🔴 IMPORTANT FLAG
# force_regen = True   # set False later when pickle is correct

# # =========================
# # DEVICE
# # =========================
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print("Using device:", device)

# # =========================
# # LOAD BLIP-2 (CPU SAFE MODEL)
# # =========================
# processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
# model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl")
# model.to(device)

# # =========================
# # FUNCTION
# # =========================
# def caption_image(image_path):
#     image = Image.open(image_path).convert("RGB")
#     inputs = processor(images=image, return_tensors="pt").to(device)
#     out = model.generate(**inputs, max_new_tokens=50)
#     caption = processor.decode(out[0], skip_special_tokens=True)

#     if not caption.strip():
#         caption = "No caption generated"

#     return caption

# # =========================
# # LOAD PICKLE IF EXISTS
# # =========================
# if os.path.exists(pkl_file) and not force_regen:
#     with open(pkl_file, "rb") as f:
#         results = pickle.load(f)
#     print("[INFO] Loaded captions from pickle")
# else:
#     print("[INFO] Generating new captions...")

#     cap = cv2.VideoCapture(video_path)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     if fps <= 0:
#         fps = 25

#     interval = int(fps * fps_interval_sec)
#     frame_id = 0
#     segment = 0
#     results = []

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         if frame_id % interval == 0:
#             # Convert BGR → RGB
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             img_path = os.path.join(frames_dir, f"frame_{segment}.png")
#             Image.fromarray(frame_rgb).save(img_path)

#             caption = caption_image(img_path)
#             timestamp = segment * fps_interval_sec

#             results.append((timestamp, caption))
#             print(f"[{timestamp}s] {caption}")

#             segment += 1

#         frame_id += 1

#     cap.release()

#     # Save pickle
#     with open(pkl_file, "wb") as f:
#         pickle.dump(results, f)

#     print("[INFO] Pickle saved:", pkl_file)

# # =========================
# # SAVE TXT FILE
# # =========================
# with open(txt_file, "w") as f:
#     for t, cap_text in results:
#         f.write(f"{t}s : {cap_text}\n")

# print("[INFO] Captions saved to:", txt_file)

# # =========================
# # DISPLAY RESULTS
# # =========================
# print("\n=== VIDEO CAPTIONS ===")
# for t, cap_text in results:
#     print(f"{t}s : {cap_text}")


# import os
# import cv2
# import pickle
# import torch
# from PIL import Image
# from transformers import BlipProcessor, BlipForConditionalGeneration

# # =========================
# # CONFIG
# # =========================
# video_path = "YouTubeClips/sample.avi"
# frames_dir = "frames"
# os.makedirs(frames_dir, exist_ok=True)
# print(os.path.exists(video_path))

# pkl_file = "video_captions.pkl"
# txt_file = "video_captions.txt"

# fps_interval_sec = 5

# # 🔴 IMPORTANT FLAG
# force_regen = True   # first time True, later False

# # =========================
# # DEVICE
# # =========================
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print("Using device:", device)

# # =========================
# # LOAD SMALL BLIP MODEL
# # =========================
# processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
# model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
# model.to(device)

# # =========================
# # FUNCTION
# # =========================
# def caption_image(image_path):
#     image = Image.open(image_path).convert("RGB")
#     inputs = processor(image, return_tensors="pt").to(device)
#     out = model.generate(**inputs, max_new_tokens=40)
#     caption = processor.decode(out[0], skip_special_tokens=True)

#     if not caption.strip():
#         caption = "No caption generated"

#     return caption

# # =========================
# # LOAD PICKLE IF EXISTS
# # =========================
# if os.path.exists(pkl_file) and not force_regen:
#     with open(pkl_file, "rb") as f:
#         results = pickle.load(f)
#     print("[INFO] Loaded captions from pickle")
# else:
#     print("[INFO] Generating new captions...")

#     cap = cv2.VideoCapture(video_path)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     if fps <= 0:
#         fps = 25

#     interval = int(fps * fps_interval_sec)
#     frame_id = 0
#     segment = 0
#     results = []

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         if frame_id % interval == 0:
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             img_path = os.path.join(frames_dir, f"frame_{segment}.png")
#             Image.fromarray(frame_rgb).save(img_path)

#             caption = caption_image(img_path)
#             timestamp = segment * fps_interval_sec

#             results.append((timestamp, caption))
#             print(f"[{timestamp}s] {caption}")

#             segment += 1

#         frame_id += 1

#     cap.release()

#     with open(pkl_file, "wb") as f:
#         pickle.dump(results, f)

#     print("[INFO] Pickle saved:", pkl_file)

# # =========================
# # SAVE TXT FILE
# # =========================
# with open(txt_file, "w") as f:
#     for t, cap_text in results:
#         f.write(f"{t}s : {cap_text}\n")

# print("[INFO] Captions saved to:", txt_file)

# # =========================
# # DISPLAY RESULTS
# # =========================
# print("\n=== VIDEO CAPTIONS ===")
# for t, cap_text in results:
#     print(f"{t}s : {cap_text}")

# import os
# import cv2
# import pickle
# import torch
# from PIL import Image
# from transformers import BlipProcessor, BlipForConditionalGeneration

# # =========================
# # CONFIG
# # =========================
# video_path = "YouTubeClips/sample.avi"
# frames_dir = "frames"
# os.makedirs(frames_dir, exist_ok=True)
# print(os.path.exists(video_path))

# video_dir = "YouTubeClips"
# video_files = [f for f in os.listdir(video_dir) if f.endswith(".avi")]

# if len(video_files) == 0:
#     print("❌ No videos found in folder")
#     exit()

# video_path = os.path.join(video_dir, video_files[0])
# print("✅ Using video:", video_path)


# pkl_file = "video_captions.pkl"
# txt_file = "video_captions.txt"

# fps_interval_sec = 2   # safer for short videos
# force_regen = True

# # =========================
# # DEVICE
# # =========================
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print("Using device:", device)

# # =========================
# # LOAD MODEL
# # =========================
# processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
# model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
# model.to(device)

# def caption_image(image_path):
#     image = Image.open(image_path).convert("RGB")
#     inputs = processor(image, return_tensors="pt").to(device)
#     out = model.generate(**inputs, max_new_tokens=40)
#     caption = processor.decode(out[0], skip_special_tokens=True)
#     return caption

# # =========================
# # OPEN VIDEO
# # =========================
# cap = cv2.VideoCapture(video_path)

# if not cap.isOpened():
#     print("❌ ERROR: Cannot open video:", video_path)
#     exit()
# else:
#     print("✅ Video opened successfully")

# fps = cap.get(cv2.CAP_PROP_FPS)
# if fps <= 0:
#     fps = 25
#     print("⚠ FPS was 0, using default 25")

# interval = int(fps * fps_interval_sec)
# frame_id = 0
# segment = 0
# results = []

# print("⏳ Processing video...")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     if frame_id % interval == 0:
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         img_path = os.path.join(frames_dir, f"frame_{segment}.png")
#         Image.fromarray(frame_rgb).save(img_path)

#         caption = caption_image(img_path)
#         timestamp = segment * fps_interval_sec

#         print(f"[{timestamp}s] {caption}")
#         results.append((timestamp, caption))
#         segment += 1

#     frame_id += 1

# cap.release()

# # =========================
# # SAVE PICKLE
# # =========================
# with open(pkl_file, "wb") as f:
#     pickle.dump(results, f)

# # =========================
# # SAVE TXT
# # =========================
# with open(txt_file, "w") as f:
#     for t, c in results:
#         f.write(f"{t}s : {c}\n")

# print("\n✅ DONE")
# print("Saved:", pkl_file, "and", txt_file)

# print("\n=== VIDEO CAPTIONS ===")
# for t, c in results:
#     print(f"{t}s : {c}")

import os
import cv2
import pickle
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# =========================
# CONFIG
# =========================
video_dir = "YouTubeClips"   # folder containing .avi videos
frames_dir = "frames"
os.makedirs(frames_dir, exist_ok=True)

pkl_file = "video_captionsBlip.pkl"
txt_file = "video_captionsBlip.txt"

fps_interval_sec = 2   # generate caption every 2 seconds
force_regen = True

# =========================
# GET VIDEO
# =========================
video_files = [f for f in os.listdir(video_dir) if f.endswith(".avi")]

if len(video_files) == 0:
    print("No videos found in folder")
    exit()

video_path = os.path.join(video_dir, video_files[7])
print("Using video:", video_path)

# =========================
# DEVICE
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# =========================
# LOAD MODEL
# =========================
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
model.to(device)

def caption_image(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=40)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# =========================
# OPEN VIDEO
# =========================
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR: Cannot open video:", video_path)
    exit()
else:
    print("Video opened successfully")

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 25
    print("⚠ FPS was 0, using default 25")

interval = int(fps * fps_interval_sec)
frame_id = 0
segment = 0
results = []

print("Processing video... (press Q to quit)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Generate caption every N seconds
    if frame_id % interval == 0:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_path = os.path.join(frames_dir, f"frame_{segment}.png")
        Image.fromarray(frame_rgb).save(img_path)

        caption = caption_image(img_path)
        timestamp = segment * fps_interval_sec

        print(f"[{timestamp}s] {caption}")
        results.append((timestamp, caption))
        segment += 1

    # Display last caption
    if len(results) > 0:
        display_text = results[-1][1]
    else:
        display_text = "Starting..."

    cv2.putText(frame, display_text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Video Captioning", frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    frame_id += 1

cap.release()
cv2.destroyAllWindows()

# =========================
# SAVE PICKLE
# =========================
with open(pkl_file, "wb") as f:
    pickle.dump(results, f)

# =========================
# SAVE TXT
# =========================
with open(txt_file, "w") as f:
    for t, c in results:
        f.write(f"{t}s : {c}\n")

print("\n✅ DONE")
print("Saved:", pkl_file, "and", txt_file)

print("\n=== VIDEO CAPTIONS ===")
for t, c in results:
    print(f"{t}s : {c}")
