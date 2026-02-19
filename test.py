########################################### captions for every 2 s ########################################
# import os, cv2, torch
# from PIL import Image
# from transformers import (
#     BlipProcessor, BlipForConditionalGeneration,
#     GitProcessor, GitForCausalLM,
#     VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
# )

# # =========================
# # CONFIG
# # =========================
# new_video_path = "YouTubeClips/_6OTzzK7t9Y_158_170.avi"  # your new video
# device = "cuda" if torch.cuda.is_available() else "cpu"
# interval_sec = 2  # generate caption every 2 seconds

# # =========================
# # LOAD VIDEO AND INFO
# # =========================
# cap = cv2.VideoCapture(new_video_path)
# if not cap.isOpened():
#     raise RuntimeError("Could not open video")

# fps = cap.get(cv2.CAP_PROP_FPS) or 25
# frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# video_duration = int(frame_count / fps)

# # Extract one frame every 2 seconds
# frames = {}
# for sec in range(0, video_duration + 1, interval_sec):
#     cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)  # jump to exact second
#     ret, frame = cap.read()
#     if not ret:
#         continue
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     frames[sec] = Image.fromarray(frame_rgb)
# cap.release()

# if not frames:
#     raise RuntimeError("No frames extracted")

# # =========================
# # LOAD MODELS
# # =========================
# print("Loading models...")

# blip_processor = BlipProcessor.from_pretrained(
#     "Salesforce/blip-image-captioning-base", use_fast=False
# )
# blip_model = BlipForConditionalGeneration.from_pretrained(
#     "Salesforce/blip-image-captioning-base"
# ).to(device)

# git_processor = GitProcessor.from_pretrained("microsoft/git-base")
# git_model = GitForCausalLM.from_pretrained("microsoft/git-base").to(device)

# vit_model = VisionEncoderDecoderModel.from_pretrained(
#     "nlpconnect/vit-gpt2-image-captioning"
# ).to(device)
# vit_processor = ViTImageProcessor.from_pretrained(
#     "nlpconnect/vit-gpt2-image-captioning", use_fast=False
# )
# vit_tokenizer = AutoTokenizer.from_pretrained(
#     "nlpconnect/vit-gpt2-image-captioning"
# )

# # =========================
# # GENERATE CAPTIONS
# # =========================
# blip_captions, git_captions, vit_captions = [], [], []

# for sec in sorted(frames.keys()):
#     frame = frames[sec]

#     # BLIP
#     inputs = blip_processor(frame, return_tensors="pt").to(device)
#     out = blip_model.generate(**inputs, max_new_tokens=40)
#     blip_caption = blip_processor.decode(out[0], skip_special_tokens=True)
#     blip_captions.append((sec, blip_caption))

#     # GIT
#     inputs = git_processor(images=frame, return_tensors="pt").to(device)
#     out = git_model.generate(**inputs, max_new_tokens=40)
#     git_caption = git_processor.batch_decode(out, skip_special_tokens=True)[0]
#     git_captions.append((sec, git_caption))

#     # ViT-GPT2
#     pixel_values = vit_processor(frame, return_tensors="pt").pixel_values.to(device)
#     out = vit_model.generate(pixel_values, max_length=40)
#     vit_caption = vit_tokenizer.decode(out[0], skip_special_tokens=True)
#     vit_captions.append((sec, vit_caption))

# # =========================
# # PRINT RESULTS
# # =========================
# print("\n====== CAPTIONS FOR NEW VIDEO ======")
# print(f"Video duration: {video_duration}s, FPS: {fps:.2f}, Total frames: {frame_count}\n")

# print("--- BLIP ---")
# for t, cap in blip_captions:
#     print(f"{t}s: {cap}")

# print("\n--- GIT ---")
# for t, cap in git_captions:
#     print(f"{t}s: {cap}")

# print("\n--- ViT-GPT2 ---")
# for t, cap in vit_captions:
#     print(f"{t}s: {cap}")


################################################### Random frames ################################################################

# import os, cv2, torch
# from PIL import Image
# from transformers import (
#     BlipProcessor, BlipForConditionalGeneration,
#     GitProcessor, GitForCausalLM,
#     VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
# )
# import random

# # =========================
# # CONFIG
# # =========================
# new_video_path = "YouTubeClips/_6OTzzK7t9Y_158_170.avi"
# device = "cuda" if torch.cuda.is_available() else "cpu"
# interval_sec = 2

# # =========================
# # LOAD VIDEO AND INFO
# # =========================
# cap = cv2.VideoCapture(new_video_path)
# if not cap.isOpened():
#     raise RuntimeError("Could not open video")

# fps = cap.get(cv2.CAP_PROP_FPS) or 25
# frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# video_duration = int(frame_count / fps)

# frames = {}
# for sec in range(0, video_duration + 1, interval_sec):
#     cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
#     ret, frame = cap.read()
#     if not ret:
#         continue
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     frames[sec] = Image.fromarray(frame_rgb)
# cap.release()

# if not frames:
#     raise RuntimeError("No frames extracted")

# # =========================
# # LOAD MODELS
# # =========================
# print("Loading models...")

# blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=False)
# blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# git_processor = GitProcessor.from_pretrained("microsoft/git-base")
# git_model = GitForCausalLM.from_pretrained("microsoft/git-base").to(device)

# vit_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning").to(device)
# vit_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning", use_fast=False)
# vit_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# # =========================
# # FUNCTION TO GENERATE CAPTIONS
# # =========================
# def generate_captions(frame_order):
#     blip_captions, git_captions, vit_captions = [], [], []
#     for sec in frame_order:
#         frame = frames[sec]

#         # BLIP
#         inputs = blip_processor(frame, return_tensors="pt").to(device)
#         out = blip_model.generate(**inputs, max_new_tokens=40)
#         blip_caption = blip_processor.decode(out[0], skip_special_tokens=True)
#         blip_captions.append((sec, blip_caption))

#         # GIT
#         inputs = git_processor(images=frame, return_tensors="pt").to(device)
#         out = git_model.generate(**inputs, max_new_tokens=40)
#         git_caption = git_processor.batch_decode(out, skip_special_tokens=True)[0]
#         git_captions.append((sec, git_caption))

#         # ViT-GPT2
#         pixel_values = vit_processor(frame, return_tensors="pt").pixel_values.to(device)
#         out = vit_model.generate(pixel_values, max_length=40)
#         vit_caption = vit_tokenizer.decode(out[0], skip_special_tokens=True)
#         vit_captions.append((sec, vit_caption))

#     return blip_captions, git_captions, vit_captions

# # =========================
# # ORIGINAL ORDER
# # =========================
# original_order = sorted(frames.keys())
# blip_orig, git_orig, vit_orig = generate_captions(original_order)

# print("\n====== CAPTIONS (ORIGINAL ORDER) ======")
# print(f"Video duration: {video_duration}s, FPS: {fps:.2f}, Total frames: {frame_count}\n")

# print("--- BLIP ---")
# for t, cap in blip_orig:
#     print(f"{t}s: {cap}")

# print("\n--- GIT ---")
# for t, cap in git_orig:
#     print(f"{t}s: {cap}")

# print("\n--- ViT-GPT2 ---")
# for t, cap in vit_orig:
#     print(f"{t}s: {cap}")

# # =========================
# # RANDOM ORDER
# # =========================
# random_order = original_order.copy()
# random.shuffle(random_order)
# blip_rand, git_rand, vit_rand = generate_captions(random_order)

# print("\n====== CAPTIONS (RANDOM ORDER) ======")
# print(f"Video duration: {video_duration}s, FPS: {fps:.2f}, Total frames: {frame_count}\n")

# print("--- BLIP ---")
# for t, cap in blip_rand:
#     print(f"{t}s: {cap}")

# print("\n--- GIT ---")
# for t, cap in git_rand:
#     print(f"{t}s: {cap}")

# print("\n--- ViT-GPT2 ---")
# for t, cap in vit_rand:
#     print(f"{t}s: {cap}")

######################################### Reverse captions, Negative captions  #########################################

# import os, cv2, torch
# from PIL import Image
# from transformers import (
#     BlipProcessor, BlipForConditionalGeneration,
#     GitProcessor, GitForCausalLM,
#     VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
# )
# import random
# import nltk

# # =========================
# # CONFIG
# # =========================
# new_video_path = "YouTubeClips/_6OTzzK7t9Y_158_170.avi"
# device = "cuda" if torch.cuda.is_available() else "cpu"
# interval_sec = 2

# # =========================
# # LOAD VIDEO AND INFO
# # =========================
# cap = cv2.VideoCapture(new_video_path)
# if not cap.isOpened():
#     raise RuntimeError("Could not open video")

# fps = cap.get(cv2.CAP_PROP_FPS) or 25
# frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# video_duration = int(frame_count / fps)

# frames = {}
# for sec in range(0, video_duration + 1, interval_sec):
#     cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
#     ret, frame = cap.read()
#     if not ret:
#         continue
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     frames[sec] = Image.fromarray(frame_rgb)
# cap.release()

# if not frames:
#     raise RuntimeError("No frames extracted")

# # =========================
# # LOAD MODELS
# # =========================
# print("Loading models...")

# blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=False)
# blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# git_processor = GitProcessor.from_pretrained("microsoft/git-base")
# git_model = GitForCausalLM.from_pretrained("microsoft/git-base").to(device)

# vit_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning").to(device)
# vit_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning", use_fast=False)
# vit_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# # =========================
# # SIMPLE NLP NEGATION FUNCTION
# # (basic verb/adjective inversion)
# # =========================
# def negate_caption(caption):
#     # This is a simple template-based negation
#     # Can be replaced with more advanced NLP later
#     negation_words = ["not", "never", "no"]
#     tokens = nltk.word_tokenize(caption)
#     if len(tokens) > 1:
#         tokens.insert(1, "not")  # Insert "not" after first word for simple negation
#     return " ".join(tokens)

# # =========================
# # CAPTION GENERATION FUNCTION
# # =========================
# def generate_captions(frame_order, negate=False):
#     blip_caps, git_caps, vit_caps = [], [], []
#     for sec in frame_order:
#         frame = frames[sec]

#         # BLIP
#         inputs = blip_processor(frame, return_tensors="pt").to(device)
#         out = blip_model.generate(**inputs, max_new_tokens=40)
#         blip_caption = blip_processor.decode(out[0], skip_special_tokens=True)
#         if negate:
#             blip_caption = negate_caption(blip_caption)
#         blip_caps.append((sec, blip_caption))

#         # GIT
#         inputs = git_processor(images=frame, return_tensors="pt").to(device)
#         out = git_model.generate(**inputs, max_new_tokens=40)
#         git_caption = git_processor.batch_decode(out, skip_special_tokens=True)[0]
#         if negate:
#             git_caption = negate_caption(git_caption)
#         git_caps.append((sec, git_caption))

#         # ViT-GPT2
#         pixel_values = vit_processor(frame, return_tensors="pt").pixel_values.to(device)
#         out = vit_model.generate(pixel_values, max_length=40)
#         vit_caption = vit_tokenizer.decode(out[0], skip_special_tokens=True)
#         if negate:
#             vit_caption = negate_caption(vit_caption)
#         vit_caps.append((sec, vit_caption))

#     return blip_caps, git_caps, vit_caps

# # =========================
# # FRAME ORDERS
# # =========================
# original_order = sorted(frames.keys())
# reverse_order = sorted(frames.keys(), reverse=True)

# # =========================
# # GENERATE CAPTIONS
# # =========================
# blip_orig, git_orig, vit_orig = generate_captions(original_order)
# blip_rev, git_rev, vit_rev = generate_captions(reverse_order)
# blip_neg, git_neg, vit_neg = generate_captions(original_order, negate=True)

# # =========================
# # PRINT RESULTS
# # =========================
# def print_captions(title, blip, git, vit):
#     print(f"\n====== {title} ======")
#     print(f"Video duration: {video_duration}s, FPS: {fps:.2f}, Total frames: {frame_count}\n")

#     print("--- BLIP ---")
#     for t, cap in blip:
#         print(f"{t}s: {cap}")

#     print("\n--- GIT ---")
#     for t, cap in git:
#         print(f"{t}s: {cap}")

#     print("\n--- ViT-GPT2 ---")
#     for t, cap in vit:
#         print(f"{t}s: {cap}")

# # Original captions
# print_captions("CAPTIONS (ORIGINAL ORDER)", blip_orig, git_orig, vit_orig)

# # Reverse captions
# print_captions("CAPTIONS (REVERSE ORDER)", blip_rev, git_rev, vit_rev)

# # Negated captions
# print_captions("CAPTIONS (NEGATED)", blip_neg, git_neg, vit_neg)



############################################################ Sentence Coherence #######################################

# import os, cv2, torch, random
# from PIL import Image
# from transformers import (
#     BlipProcessor, BlipForConditionalGeneration,
#     GitProcessor, GitForCausalLM,
#     VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
# )
# from sentence_transformers import SentenceTransformer, util

# # =========================
# # CONFIG
# # =========================
# new_video_path = "VideoTest.mp4"
# device = "cuda" if torch.cuda.is_available() else "cpu"
# interval_sec = 2  # extract frame every 2 seconds

# # =========================
# # LOAD VIDEO AND EXTRACT FRAMES
# # =========================
# cap = cv2.VideoCapture(new_video_path)
# if not cap.isOpened():
#     raise RuntimeError("Could not open video")

# fps = cap.get(cv2.CAP_PROP_FPS) or 25
# frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# video_duration = int(frame_count / fps)

# frames = {}
# for sec in range(0, video_duration + 1, interval_sec):
#     cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
#     ret, frame = cap.read()
#     if not ret:
#         continue
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     frames[sec] = Image.fromarray(frame_rgb)
# cap.release()

# if not frames:
#     raise RuntimeError("No frames extracted")

# # =========================
# # LOAD MODELS
# # =========================
# print("Loading models...")

# # BLIP
# blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=False)
# blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# # GIT
# git_processor = GitProcessor.from_pretrained("microsoft/git-base")
# git_model = GitForCausalLM.from_pretrained("microsoft/git-base").to(device)

# # ViT-GPT2
# vit_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning").to(device)
# vit_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning", use_fast=False)
# vit_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# # Sentence embedding model for story coherence
# embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# # =========================
# # FUNCTION TO GENERATE CAPTIONS
# # =========================
# def generate_captions(frame_order):
#     blip_captions, git_captions, vit_captions = [], [], []
#     for sec in frame_order:
#         frame = frames[sec]

#         # BLIP
#         inputs = blip_processor(frame, return_tensors="pt").to(device)
#         out = blip_model.generate(**inputs, max_new_tokens=40)
#         blip_caption = blip_processor.decode(out[0], skip_special_tokens=True)
#         blip_captions.append((sec, blip_caption))

#         # GIT
#         inputs = git_processor(images=frame, return_tensors="pt").to(device)
#         out = git_model.generate(**inputs, max_new_tokens=40)
#         git_caption = git_processor.batch_decode(out, skip_special_tokens=True)[0]
#         git_captions.append((sec, git_caption))

#         # ViT-GPT2
#         pixel_values = vit_processor(frame, return_tensors="pt").pixel_values.to(device)
#         out = vit_model.generate(pixel_values, max_length=40)
#         vit_caption = vit_tokenizer.decode(out[0], skip_special_tokens=True)
#         vit_captions.append((sec, vit_caption))

#     return blip_captions, git_captions, vit_captions

# # =========================
# # FUNCTION TO MAKE COHERENT STORY
# # =========================
# def captions_to_coherent_story(captions, model_name="Model", similarity_threshold=0.9):
#     # Sort captions by time
#     captions_sorted = sorted(captions, key=lambda x: x[0])
    
#     story_lines = []
#     prev_embedding = None

#     for t, cap in captions_sorted:
#         embedding = embed_model.encode(cap, convert_to_tensor=True)
#         # Skip if semantically similar to previous caption
#         if prev_embedding is not None:
#             sim = util.cos_sim(embedding, prev_embedding).item()
#             if sim >= similarity_threshold:
#                 continue
#         story_lines.append(cap)
#         prev_embedding = embedding

#     # Add natural transitions
#     story_text = " ".join([f"Then, {line.lower()}" for line in story_lines])
#     story_text = story_text[0].upper() + story_text[1:]  # capitalize first letter

#     print(f"\n--- {model_name} Coherent Story ---")
#     print(story_text)
#     return story_text

# # =========================
# # ORIGINAL ORDER CAPTIONS
# # =========================
# original_order = sorted(frames.keys())
# blip_orig, git_orig, vit_orig = generate_captions(original_order)

# print("\n====== CAPTIONS (ORIGINAL ORDER) ======")
# print(f"Video duration: {video_duration}s, FPS: {fps:.2f}, Total frames: {frame_count}\n")

# print("--- BLIP ---")
# for t, cap in blip_orig:
#     print(f"{t}s: {cap}")

# print("\n--- GIT ---")
# for t, cap in git_orig:
#     print(f"{t}s: {cap}")

# print("\n--- ViT-GPT2 ---")
# for t, cap in vit_orig:
#     print(f"{t}s: {cap}")

# # =========================
# # CREATE COHERENT STORIES
# # =========================
# blip_story = captions_to_coherent_story(blip_orig, "BLIP")
# git_story = captions_to_coherent_story(git_orig, "GIT")
# vit_story = captions_to_coherent_story(vit_orig, "ViT-GPT2")

# # =========================
# # RANDOM ORDER CAPTIONS
# # =========================
# random_order = original_order.copy()
# random.shuffle(random_order)
# blip_rand, git_rand, vit_rand = generate_captions(random_order)

# print("\n====== CAPTIONS (RANDOM ORDER) ======")
# print(f"Video duration: {video_duration}s, FPS: {fps:.2f}, Total frames: {frame_count}\n")

# print("--- BLIP ---")
# for t, cap in blip_rand:
#     print(f"{t}s: {cap}")

# print("\n--- GIT ---")
# for t, cap in git_rand:
#     print(f"{t}s: {cap}")

# print("--- ViT-GPT2 ---")
# for t, cap in vit_rand:
#     print(f"{t}s: {cap}")


############################################################ Different video Captioning ##################################################

import os, cv2, torch
from PIL import Image
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    GitProcessor, GitForCausalLM,
    VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer,
    T5Tokenizer, T5ForConditionalGeneration
)

# =========================
# CONFIG
# =========================
video_path = "VideoTest.mp4"
interval_sec = 2
device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# LOAD VIDEO
# =========================
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError("Could not open video")

fps = cap.get(cv2.CAP_PROP_FPS) or 25
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_duration = int(frame_count / fps)

frames = {}
for sec in range(0, video_duration + 1, interval_sec):
    cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
    ret, frame = cap.read()
    if not ret:
        continue
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames[sec] = Image.fromarray(frame_rgb)

cap.release()
if not frames:
    raise RuntimeError("No frames extracted")

print(f"\nVideo duration: {video_duration}s | FPS: {fps:.2f} | Frames sampled: {len(frames)}")

# =========================
# LOAD MODELS
# =========================
print("\nLoading captioning models...")

blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=False)
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

git_processor = GitProcessor.from_pretrained("microsoft/git-base")
git_model = GitForCausalLM.from_pretrained("microsoft/git-base").to(device)

vit_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning").to(device)
vit_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning", use_fast=False)
vit_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

print("Loading story model (T5)...")
story_tokenizer = T5Tokenizer.from_pretrained("t5-small")
story_model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)

# =========================
# GENERATE CAPTIONS
# =========================
blip_captions, git_captions, vit_captions = [], [], []

print("\nGenerating captions...")
for sec in sorted(frames.keys()):
    frame = frames[sec]

    # BLIP
    inputs = blip_processor(frame, return_tensors="pt").to(device)
    out = blip_model.generate(**inputs, max_new_tokens=40)
    blip_caption = blip_processor.decode(out[0], skip_special_tokens=True)
    blip_captions.append((sec, blip_caption))

    # GIT
    inputs = git_processor(images=frame, return_tensors="pt").to(device)
    out = git_model.generate(**inputs, max_new_tokens=40)
    git_caption = git_processor.batch_decode(out, skip_special_tokens=True)[0]
    git_captions.append((sec, git_caption))

    # ViT-GPT2
    pixel_values = vit_processor(frame, return_tensors="pt").pixel_values.to(device)
    out = vit_model.generate(pixel_values, max_length=40)
    vit_caption = vit_tokenizer.decode(out[0], skip_special_tokens=True)
    vit_captions.append((sec, vit_caption))

# =========================
# STORY GENERATOR
# =========================
def generate_story(captions):
    captions = sorted(captions, key=lambda x: x[0])
    text = " ".join([cap for _, cap in captions])

    prompt = f"{text}"

    inputs = story_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

    outputs = story_model.generate(
        **inputs,
        max_length=250,
        num_beams=5,
        temperature=0.8,
        top_p=0.9,
        repetition_penalty=2.0
    )

    story = story_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return story

# =========================
# PRINT RESULTS
# =========================
print("\n====== FRAME-LEVEL CAPTIONS (ViT-GPT2) ======")
for t, cap in vit_captions:
    print(f"{t}s: {cap}")

print("\n====== GENERATED STORIES ======")

print("\n--- BLIP STORY ---")
print(generate_story(blip_captions))

print("\n--- GIT STORY ---")
print(generate_story(git_captions))

print("\n--- ViT-GPT2 STORY ---")
print(generate_story(vit_captions))




