import os, cv2, torch, pickle
from PIL import Image
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    GitProcessor, GitForCausalLM,
    VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
)

# =========================
# CONFIG
# =========================
video_dir = "YouTubeClips"
fps_interval_sec = 2

pickle_files = {
    "BLIP": "blip_cap.pkl",
    "GIT": "git_cap.pkl",
    "VITGPT2": "vitgpt2_cap.pkl"
}

txt_files = {
    "BLIP": "blip_cap.txt",
    "GIT": "git_cap.txt",
    "VITGPT2": "vitgpt2_cap.txt"
}

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# =========================
# LOAD VIDEOS
# =========================
videos = [v for v in os.listdir(video_dir) if v.endswith(".avi")]
if not videos:
    print("❌ No videos found")
    exit()

# =========================
# FRAME EXTRACTOR
# =========================
def extract_frames(video_path, interval_sec=2):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    interval = int(fps * interval_sec)

    frames = []
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % interval == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))

        frame_id += 1

    cap.release()
    return frames

# =========================
# BLIP
# =========================
def run_blip():
    if os.path.exists(pickle_files["BLIP"]):
        print("Loading BLIP from pickle")
        return pickle.load(open(pickle_files["BLIP"], "rb"))

    print(" Running BLIP...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(device)

    results = []
    for video in videos:
        frames = extract_frames(os.path.join(video_dir, video), fps_interval_sec)
        if not frames:
            continue
        inputs = processor(frames[0], return_tensors="pt").to(device)
        out = model.generate(**inputs, max_new_tokens=40)
        caption = processor.decode(out[0], skip_special_tokens=True)
        results.append((video, caption))
        print("BLIP:", video, "->", caption)

    pickle.dump(results, open(pickle_files["BLIP"], "wb"))
    return results

# =========================
# GIT
# =========================
def run_git():
    if os.path.exists(pickle_files["GIT"]):
        print("Loading GIT from pickle")
        return pickle.load(open(pickle_files["GIT"], "rb"))

    print("Running GIT...")
    processor = GitProcessor.from_pretrained("microsoft/git-base")
    model = GitForCausalLM.from_pretrained("microsoft/git-base").to(device)

    results = []
    for video in videos:
        frames = extract_frames(os.path.join(video_dir, video), fps_interval_sec)
        if not frames:
            continue
        inputs = processor(images=frames[0], return_tensors="pt").to(device)
        out = model.generate(**inputs, max_new_tokens=40)
        caption = processor.batch_decode(out, skip_special_tokens=True)[0]
        results.append((video, caption))
        print("GIT:", video, "->", caption)

    pickle.dump(results, open(pickle_files["GIT"], "wb"))
    return results

# =========================
# ViT-GPT2
# =========================
def run_vitgpt2():
    if os.path.exists(pickle_files["VITGPT2"]):
        print("Loading ViT-GPT2 from pickle")
        return pickle.load(open(pickle_files["VITGPT2"], "rb"))

    print(" Running ViT-GPT2...")
    model = VisionEncoderDecoderModel.from_pretrained(
        "nlpconnect/vit-gpt2-image-captioning"
    ).to(device)
    processor = ViTImageProcessor.from_pretrained(
        "nlpconnect/vit-gpt2-image-captioning"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "nlpconnect/vit-gpt2-image-captioning"
    )

    results = []
    for video in videos:
        frames = extract_frames(os.path.join(video_dir, video), fps_interval_sec)
        if not frames:
            continue
        pixel_values = processor(frames[0], return_tensors="pt").pixel_values.to(device)
        out = model.generate(pixel_values, max_length=40)
        caption = tokenizer.decode(out[0], skip_special_tokens=True)
        results.append((video, caption))
        print("ViT-GPT2:", video, "->", caption)

    pickle.dump(results, open(pickle_files["VITGPT2"], "wb"))
    return results

# =========================
# RUN ALL MODELS
# =========================
blip_results = run_blip()
git_results = run_git()
vit_results = run_vitgpt2()

# =========================
# SAVE TXT FILES
# =========================
for name, data in zip(
    ["BLIP", "GIT", "VITGPT2"],
    [blip_results, git_results, vit_results]
):
    with open(txt_files[name], "w", encoding="utf-8") as f:
        for video, caption in data:
            f.write(f"{video}: {caption}\n")

print("\n ALL MODELS COMPLETED")
print("Saved:")
print("BLIP ->", txt_files["BLIP"])
print("GIT ->", txt_files["GIT"])
print("ViT-GPT2 ->", txt_files["VITGPT2"])
