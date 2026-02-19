import os, cv2, torch, pickle
import nltk
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    GitProcessor, GitForCausalLM,
    VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
)
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

# =========================
# DOWNLOAD NLTK DATA (run once)
# =========================
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")

# =========================
# CONFIG
# =========================
video_path = "new_video.avi"          # 🔹 NEW VIDEO
reference_caption = "a man is playing football"  # 🔹 TRUE CAPTION

fps_interval_sec = 2
device = "cuda" if torch.cuda.is_available() else "cpu"

pickle_files = {
    "BLIP": "blip_new.pkl",
    "GIT": "git_new.pkl",
    "VITGPT2": "vitgpt2_new.pkl"
}

# =========================
# FRAME EXTRACTION
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

frames = extract_frames(video_path, fps_interval_sec)
if not frames:
    print("❌ No frames extracted")
    exit()

# =========================
# BLIP
# =========================
def run_blip(frames):
    if os.path.exists(pickle_files["BLIP"]):
        return pickle.load(open(pickle_files["BLIP"], "rb"))

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(device)

    inputs = processor(frames[0], return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=40)
    caption = processor.decode(out[0], skip_special_tokens=True)

    pickle.dump(caption, open(pickle_files["BLIP"], "wb"))
    return caption

# =========================
# GIT
# =========================
def run_git(frames):
    if os.path.exists(pickle_files["GIT"]):
        return pickle.load(open(pickle_files["GIT"], "rb"))

    processor = GitProcessor.from_pretrained("microsoft/git-base")
    model = GitForCausalLM.from_pretrained("microsoft/git-base").to(device)

    inputs = processor(images=frames[0], return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=40)
    caption = processor.batch_decode(out, skip_special_tokens=True)[0]

    pickle.dump(caption, open(pickle_files["GIT"], "wb"))
    return caption

# =========================
# ViT-GPT2
# =========================
def run_vitgpt2(frames):
    if os.path.exists(pickle_files["VITGPT2"]):
        return pickle.load(open(pickle_files["VITGPT2"], "rb"))

    model = VisionEncoderDecoderModel.from_pretrained(
        "nlpconnect/vit-gpt2-image-captioning"
    ).to(device)
    processor = ViTImageProcessor.from_pretrained(
        "nlpconnect/vit-gpt2-image-captioning"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "nlpconnect/vit-gpt2-image-captioning"
    )

    pixel_values = processor(frames[0], return_tensors="pt").pixel_values.to(device)
    out = model.generate(pixel_values, max_length=40)
    caption = tokenizer.decode(out[0], skip_special_tokens=True)

    pickle.dump(caption, open(pickle_files["VITGPT2"], "wb"))
    return caption

# =========================
# RUN MODELS
# =========================
blip_caption = run_blip(frames)
git_caption = run_git(frames)
vit_caption = run_vitgpt2(frames)

print("\n=== GENERATED CAPTIONS ===")
print("BLIP:", blip_caption)
print("GIT:", git_caption)
print("ViT-GPT2:", vit_caption)

# =========================
# EVALUATION
# =========================
smoothie = SmoothingFunction().method4
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def compute_scores(pred, ref):
    ref_tokens = nltk.word_tokenize(ref.lower())
    pred_tokens = nltk.word_tokenize(pred.lower())

    bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothie)
    meteor = meteor_score([ref], pred)
    rouge = scorer.score(ref, pred)['rougeL'].fmeasure

    return bleu, meteor, rouge

results = []

for name, caption in zip(
    ["BLIP", "GIT", "ViT-GPT2"],
    [blip_caption, git_caption, vit_caption]
):
    bleu, meteor, rouge = compute_scores(caption, reference_caption)
    results.append([name, bleu, meteor, rouge])

df = pd.DataFrame(results, columns=["Model", "BLEU", "METEOR", "ROUGE-L"])

print("\n=== EVALUATION RESULTS ===")
print(df)

# =========================
# GRAPH
# =========================
df.set_index("Model").plot(kind="bar", figsize=(8,5))
plt.title("Captioning Comparison on New Video")
plt.ylabel("Score")
plt.ylim(0,1)
plt.grid(True)
plt.show()