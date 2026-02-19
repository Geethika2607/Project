# import pickle
# import nltk
# import pandas as pd
# import matplotlib.pyplot as plt
# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# from nltk.translate.meteor_score import meteor_score
# from rouge_score import rouge_scorer

# nltk.download("punkt")
# nltk.download("wordnet")
# nltk.download("omw-1.4")

# # =========================
# # FILES
# # =========================
# reference_file = "captions.txt"

# pickle_files = {
#     "BLIP": "blip_cap.pkl",
#     "GIT": "git_cap.pkl",
#     "VITGPT2": "vitgpt2_cap.pkl"
# }

# # =========================
# # LOAD REFERENCE CAPTIONS
# # =========================
# def load_reference(file):
#     refs = {}
#     with open(file, "r", encoding="utf-8") as f:
#         for line in f:
#             if ":" in line:
#                 vid, cap = line.strip().split(":", 1)
#                 refs[vid.strip()] = cap.strip()
#     return refs

# references = load_reference(reference_file)

# # =========================
# # LOAD PKL CAPTIONS
# # =========================
# def load_pickle(file):
#     data = pickle.load(open(file, "rb"))  # list of (video, caption)
#     return dict(data)

# # =========================
# # METRICS
# # =========================
# smoothie = SmoothingFunction().method4
# scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

# def compute_scores(preds, refs):
#     bleu_list, meteor_list, rouge_list = [], [], []

#     for vid in preds:
#         if vid not in refs:
#             continue

#         pred = preds[vid]
#         ref = refs[vid]

#         ref_tokens = nltk.word_tokenize(ref.lower())
#         pred_tokens = nltk.word_tokenize(pred.lower())

#         bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothie)
#         meteor = meteor_score([ref], pred)
#         rouge = scorer.score(ref, pred)["rougeL"].fmeasure

#         bleu_list.append(bleu)
#         meteor_list.append(meteor)
#         rouge_list.append(rouge)

#     return (
#         sum(bleu_list)/len(bleu_list),
#         sum(meteor_list)/len(meteor_list),
#         sum(rouge_list)/len(rouge_list)
#     )

# # =========================
# # RUN EVALUATION
# # =========================
# results = []

# for model, pkl_file in pickle_files.items():
#     predictions = load_pickle(pkl_file)
#     bleu, meteor, rouge = compute_scores(predictions, references)
#     results.append([model, bleu, meteor, rouge])

# df = pd.DataFrame(results, columns=["Model", "BLEU", "METEOR", "ROUGE-L"])

# print("\n=== FINAL EVALUATION ===")
# print(df)

# # =========================
# # SAVE CSV
# # =========================
# df.to_csv("evaluation_results.csv", index=False)

# # =========================
# # PLOT GRAPH
# # =========================
# df.set_index("Model").plot(kind="bar", figsize=(8,5))
# plt.title("Video Captioning Model Comparison")
# plt.ylabel("Score")
# plt.ylim(0,1)
# plt.grid(True)
# plt.show()

# import os

# video_dir = "YouTubeClips"
# caption_file = "captions.txt"

# videos = set()

# for v in os.listdir(video_dir):
#     if v.endswith(".avi"):
#         vid = os.path.splitext(v)[0]  # remove .avi
#         videos.add(vid)

# matched = 0
# total_caps = 0

# with open(caption_file, "r", encoding="utf-8") as f:
#     for line in f:
#         parts = line.strip().split()
#         if len(parts) > 1:
#             total_caps += 1
#             cap_id = parts[0]
#             if cap_id in videos:
#                 matched += 1

# print("Videos in folder:", len(videos))
# print("Captions in file:", total_caps)
# print("Matched videos:", matched)

# import nltk
# import pandas as pd
# import matplotlib.pyplot as plt
# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# from nltk.translate.meteor_score import meteor_score
# from rouge_score import rouge_scorer

# # =========================
# # DOWNLOAD NLTK DATA
# # =========================
# nltk.download("punkt")
# nltk.download("wordnet")
# nltk.download("omw-1.4")

# # =========================
# # FILE PATHS
# # =========================
# reference_file = "captions.txt"

# model_files = {
#     "BLIP": "blip_cap.txt",
#     "GIT": "git_cap.txt",
#     "ViT-GPT2": "vitgpt2_cap.txt"
# }

# # =========================
# # LOAD REFERENCE FILE
# # Format: videoID caption
# # =========================
# def load_reference(file):
#     data = {}
#     with open(file, "r", encoding="utf-8") as f:
#         for line in f:
#             parts = line.strip().split(" ", 1)
#             if len(parts) == 2:
#                 vid, cap = parts
#                 data[vid.strip()] = cap.strip()
#     return data

# # =========================
# # LOAD PREDICTION FILE
# # Format: videoID.avi: caption
# # =========================
# def load_predictions(file):
#     data = {}
#     with open(file, "r", encoding="utf-8") as f:
#         for line in f:
#             if ":" in line:
#                 vid, cap = line.strip().split(":", 1)
#                 vid = vid.replace(".avi", "").strip()
#                 data[vid] = cap.strip()
#     return data

# references = load_reference(reference_file)
# print("Reference captions:", len(references))

# # =========================
# # METRICS
# # =========================
# smoothie = SmoothingFunction().method4
# rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

# def compute_scores(pred_dict, ref_dict, model_name):
#     bleu_list, meteor_list, rouge_list = [], [], []

#     common_ids = set(pred_dict.keys()) & set(ref_dict.keys())
#     print(f"{model_name} matched videos:", len(common_ids))

#     if len(common_ids) == 0:
#         return 0, 0, 0, 0

#     for vid in common_ids:
#         pred = pred_dict[vid].lower()
#         ref = ref_dict[vid].lower()

#         ref_tokens = nltk.word_tokenize(ref)
#         pred_tokens = nltk.word_tokenize(pred)

#         bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothie)
#         meteor = meteor_score([ref], pred)
#         rouge_l = rouge.score(ref, pred)["rougeL"].fmeasure

#         bleu_list.append(bleu)
#         meteor_list.append(meteor)
#         rouge_list.append(rouge_l)

#     return (
#         sum(bleu_list) / len(bleu_list),
#         sum(meteor_list) / len(meteor_list),
#         sum(rouge_list) / len(rouge_list),
#         len(common_ids)
#     )

# # =========================
# # RUN EVALUATION
# # =========================
# results = []

# for model_name, file in model_files.items():
#     preds = load_predictions(file)
#     print(f"\nLoaded {model_name} captions:", len(preds))

#     bleu, meteor, rouge_l, matched = compute_scores(preds, references, model_name)
#     results.append([model_name, bleu, meteor, rouge_l, matched])

# # =========================
# # RESULTS TABLE
# # =========================
# df = pd.DataFrame(results, columns=["Model", "BLEU", "METEOR", "ROUGE-L", "Matched Videos"])
# print("\n=== FINAL RESULTS ===")
# print(df)

# df.to_csv("final_model_comparison.csv", index=False)

# # =========================
# # PLOT GRAPH
# # =========================
# df.plot(x="Model", y=["BLEU", "METEOR", "ROUGE-L"], kind="bar", figsize=(9,6))
# plt.title("Captioning Model Comparison (Matched Videos Only)")
# plt.ylabel("Score")
# plt.ylim(0,1)
# plt.grid(True)
# plt.tight_layout()
# plt.show()


# import nltk
# import pandas as pd
# import matplotlib.pyplot as plt
# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# from nltk.translate.meteor_score import meteor_score
# from rouge_score import rouge_scorer

# # =========================
# # DOWNLOAD REQUIRED NLTK DATA
# # =========================
# nltk.download("punkt")
# nltk.download("punkt_tab")
# nltk.download("wordnet")
# nltk.download("omw-1.4")

# # =========================
# # FILE PATHS
# # =========================
# reference_file = "captions.txt"

# model_files = {
#     "BLIP": "blip_cap.txt",
#     "GIT": "git_cap.txt",
#     "ViT-GPT2": "vitgpt2_cap.txt"
# }

# # =========================
# # LOAD REFERENCE FILE
# # Format: videoID caption
# # =========================
# def load_reference(file):
#     data = {}
#     with open(file, "r", encoding="utf-8") as f:
#         for line in f:
#             parts = line.strip().split(" ", 1)
#             if len(parts) == 2:
#                 vid, cap = parts
#                 data[vid.strip()] = cap.strip()
#     return data

# # =========================
# # LOAD MODEL FILE
# # Format: videoID.avi: caption
# # =========================
# def load_predictions(file):
#     data = {}
#     with open(file, "r", encoding="utf-8") as f:
#         for line in f:
#             if ":" in line:
#                 vid, cap = line.strip().split(":", 1)
#                 vid = vid.replace(".avi", "").strip()
#                 data[vid] = cap.strip()
#     return data

# references = load_reference(reference_file)
# print("Reference captions:", len(references))

# smoothie = SmoothingFunction().method4
# rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

# # =========================
# # COMPUTE METRICS SAFELY
# # =========================
# def compute_scores(pred_dict, ref_dict, model_name):
#     bleu_list, meteor_list, rouge_list = [], [], []

#     common_ids = set(pred_dict.keys()) & set(ref_dict.keys())
#     print(f"{model_name} matched videos:", len(common_ids))

#     if len(common_ids) == 0:
#         print(f"⚠ No matching videos for {model_name}, skipping...")
#         return 0, 0, 0, 0

#     for vid in common_ids:
#         pred = pred_dict[vid].lower()
#         ref = ref_dict[vid].lower()

#         ref_tokens = nltk.word_tokenize(ref)
#         pred_tokens = nltk.word_tokenize(pred)

#         bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothie)
#         meteor = meteor_score([ref], pred)
#         rouge_l = rouge.score(ref, pred)["rougeL"].fmeasure

#         bleu_list.append(bleu)
#         meteor_list.append(meteor)
#         rouge_list.append(rouge_l)

#     return (
#         sum(bleu_list) / len(bleu_list),
#         sum(meteor_list) / len(meteor_list),
#         sum(rouge_list) / len(rouge_list),
#         len(common_ids)
#     )

# # =========================
# # RUN EVALUATION
# # =========================
# results = []

# for model_name, file in model_files.items():
#     print(f"\nEvaluating {model_name}...")
#     preds = load_predictions(file)

#     bleu, meteor, rouge_l, matched = compute_scores(preds, references, model_name)

#     results.append([model_name, bleu, meteor, rouge_l, matched])

# # =========================
# # RESULTS TABLE
# # =========================
# df = pd.DataFrame(results, columns=["Model", "BLEU", "METEOR", "ROUGE-L", "Matched Videos"])
# print("\n=== FINAL RESULTS ===")
# print(df)

# df.to_csv("final_model_comparison.csv", index=False)

# # =========================
# # PLOT GRAPH
# # =========================
# df.plot(x="Model", y=["BLEU", "METEOR", "ROUGE-L"], kind="bar", figsize=(9,6))
# plt.title("Captioning Model Comparison (Matched Videos Only)")
# plt.ylabel("Score")
# plt.ylim(0,1)
# plt.grid(True)
# plt.tight_layout()
# plt.show()


# import nltk
# import pandas as pd
# import matplotlib.pyplot as plt
# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# from nltk.translate.meteor_score import meteor_score
# from rouge_score import rouge_scorer
# import os

# # =========================
# # DOWNLOAD REQUIRED NLTK DATA
# # =========================
# nltk.download("punkt")
# nltk.download("wordnet")
# nltk.download("omw-1.4")

# # =========================
# # FILE PATHS
# # =========================
# reference_file = "captions.txt"

# model_files = {
#     "BLIP": "blip_cap.txt",
#     "GIT": "git_cap.txt",
#     "ViT-GPT2": "vitgpt2_cap.txt"
# }

# # =========================
# # LOAD REFERENCE FILE
# # Format: videoID caption
# # =========================
# def load_reference(file):
#     data = {}
#     with open(file, "r", encoding="utf-8") as f:
#         for line in f:
#             parts = line.strip().split(" ", 1)
#             if len(parts) == 2:
#                 vid, cap = parts
#                 data[vid.strip()] = cap.strip()
#     return data

# # =========================
# # LOAD MODEL FILE
# # Format: videoID.avi: caption
# # =========================
# def load_predictions(file):
#     data = {}
#     with open(file, "r", encoding="utf-8") as f:
#         for line in f:
#             if ":" in line:
#                 vid, cap = line.strip().split(":", 1)
#                 vid = os.path.splitext(vid)[0].strip()  # remove .avi or other extensions
#                 data[vid] = cap.strip()
#     return data

# # =========================
# # COMPUTE METRICS
# # =========================
# def compute_scores(pred_dict, ref_dict, model_name):
#     smoothie = SmoothingFunction().method4
#     rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

#     bleu_list, meteor_list, rouge_list = [], [], []

#     # Only compare videos present in both sets
#     common_ids = set(pred_dict.keys()) & set(ref_dict.keys())
#     print(f"{model_name} matched videos:", len(common_ids))

#     if len(common_ids) == 0:
#         print(f"⚠ No matching videos for {model_name}, skipping...")
#         return 0, 0, 0, 0

#     for vid in common_ids:
#         pred = pred_dict[vid].lower()
#         ref = ref_dict[vid].lower()

#         # Skip empty captions
#         if not pred or not ref:
#             continue

#         # Tokenize for BLEU & METEOR
#         pred_tokens = nltk.word_tokenize(pred)
#         ref_tokens = nltk.word_tokenize(ref)

#         # Compute metrics
#         bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothie)
#         meteor = meteor_score([ref_tokens], pred_tokens)
#         rouge_l = rouge.score(ref, pred)["rougeL"].fmeasure

#         bleu_list.append(bleu)
#         meteor_list.append(meteor)
#         rouge_list.append(rouge_l)

#     # Avoid division by zero
#     n = max(len(bleu_list), 1)
#     return (
#         sum(bleu_list) / n,
#         sum(meteor_list) / n,
#         sum(rouge_list) / n,
#         len(common_ids)
#     )

# # =========================
# # MAIN EVALUATION
# # =========================
# def main():
#     references = load_reference(reference_file)
#     print("Reference captions loaded:", len(references))

#     results = []

#     for model_name, file in model_files.items():
#         if not os.path.exists(file):
#             print(f"⚠ File not found for {model_name}: {file}, skipping...")
#             continue

#         print(f"\nEvaluating {model_name}...")
#         preds = load_predictions(file)
#         bleu, meteor, rouge_l, matched = compute_scores(preds, references, model_name)
#         results.append([model_name, bleu, meteor, rouge_l, matched])

#     # =========================
#     # RESULTS TABLE
#     # =========================
#     df = pd.DataFrame(results, columns=["Model", "BLEU", "METEOR", "ROUGE-L", "Matched Videos"])
#     print("\n=== FINAL RESULTS ===")
#     print(df)

#     # Save CSV
#     df.to_csv("final_model_comparison.csv", index=False)
#     print("\n✅ Saved results to final_model_comparison.csv")

#     # =========================
#     # PLOT GRAPH
#     # =========================
#     df_plot = df.set_index("Model")[["BLEU", "METEOR", "ROUGE-L"]]
#     df_plot.plot(kind="bar", figsize=(10,6))
#     plt.title("Captioning Model Comparison (Matched Videos Only)")
#     plt.ylabel("Score")
#     plt.ylim(0,1)
#     plt.grid(axis="y")
#     plt.tight_layout()
#     plt.show()

# if __name__ == "__main__":
#     main()

# import nltk
# import pandas as pd
# import matplotlib.pyplot as plt
# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# from nltk.translate.meteor_score import meteor_score
# from rouge_score import rouge_scorer
# import os

# # For CIDEr & SPICE
# from pycocoevalcap.cider.cider import Cider
# from pycocoevalcap.spice.spice import Spice

# # =========================
# # DOWNLOAD REQUIRED NLTK DATA
# # =========================
# nltk.download("punkt")
# nltk.download("wordnet")
# nltk.download("omw-1.4")

# # =========================
# # FILE PATHS
# # =========================
# reference_file = "captions.txt"

# model_files = {
#     "BLIP": "blip_cap.txt",
#     "GIT": "git_cap.txt",
#     "ViT-GPT2": "vitgpt2_cap.txt"
# }

# # =========================
# # LOAD REFERENCE FILE
# # Format: videoID caption
# # =========================
# def load_reference(file):
#     data = {}
#     with open(file, "r", encoding="utf-8") as f:
#         for line in f:
#             parts = line.strip().split(" ", 1)
#             if len(parts) == 2:
#                 vid, cap = parts
#                 data[vid.strip()] = cap.strip()
#     return data

# # =========================
# # LOAD MODEL FILE
# # Format: videoID.avi: caption
# # =========================
# def load_predictions(file):
#     data = {}
#     with open(file, "r", encoding="utf-8") as f:
#         for line in f:
#             if ":" in line:
#                 vid, cap = line.strip().split(":", 1)
#                 vid = os.path.splitext(vid)[0].strip()  # remove .avi or other extensions
#                 data[vid] = cap.strip()
#     return data

# # =========================
# # COMPUTE METRICS
# # =========================
# def compute_scores(pred_dict, ref_dict, model_name):
#     smoothie = SmoothingFunction().method4
#     rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

#     bleu_list, meteor_list, rouge_list = [], [], []

#     # Prepare for CIDEr/SPICE
#     preds_for_cider = {}
#     refs_for_cider = {}

#     # Only compare videos present in both sets
#     common_ids = set(pred_dict.keys()) & set(ref_dict.keys())
#     print(f"{model_name} matched videos:", len(common_ids))

#     if len(common_ids) == 0:
#         print(f"⚠ No matching videos for {model_name}, skipping...")
#         return 0,0,0,0,0,0

#     for vid in common_ids:
#         pred = pred_dict[vid].lower()
#         ref = ref_dict[vid].lower()

#         if not pred or not ref:
#             continue

#         # Tokenize for BLEU & METEOR
#         pred_tokens = nltk.word_tokenize(pred)
#         ref_tokens = nltk.word_tokenize(ref)

#         bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothie)
#         meteor = meteor_score([ref_tokens], pred_tokens)
#         rouge_l = rouge.score(ref, pred)["rougeL"].fmeasure

#         bleu_list.append(bleu)
#         meteor_list.append(meteor)
#         rouge_list.append(rouge_l)

#         # Wrap in list for pycocoevalcap
#         preds_for_cider[vid] = [pred]
#         refs_for_cider[vid] = [ref]

#     # =========================
#     # CIDEr & SPICE
#     # =========================
#     cider_score, spice_score = 0, 0
#     try:
#         cider_scorer = Cider()
#         cider_score, _ = cider_scorer.compute_score(refs_for_cider, preds_for_cider)
#     except Exception as e:
#         print(f"⚠ CIDEr error for {model_name}: {e}")

#     try:
#         spice_scorer = Spice()
#         spice_score, _ = spice_scorer.compute_score(refs_for_cider, preds_for_cider)
#     except Exception as e:
#         print(f"⚠ SPICE error for {model_name}: {e}")

#     n = max(len(bleu_list), 1)
#     return (
#         sum(bleu_list)/n,
#         sum(meteor_list)/n,
#         sum(rouge_list)/n,
#         cider_score,
#         spice_score,
#         len(common_ids)
#     )

# # =========================
# # MAIN EVALUATION
# # =========================
# def main():
#     references = load_reference(reference_file)
#     print("Reference captions loaded:", len(references))

#     results = []

#     for model_name, file in model_files.items():
#         if not os.path.exists(file):
#             print(f"⚠ File not found for {model_name}: {file}, skipping...")
#             continue

#         print(f"\nEvaluating {model_name}...")
#         preds = load_predictions(file)
#         bleu, meteor, rouge_l, cider, spice, matched = compute_scores(preds, references, model_name)
#         results.append([model_name, bleu, meteor, rouge_l, cider, spice, matched])

#     # =========================
#     # RESULTS TABLE
#     # =========================
#     df = pd.DataFrame(results, columns=["Model","BLEU","METEOR","ROUGE-L","CIDEr","SPICE","Matched Videos"])
#     print("\n=== FINAL RESULTS ===")
#     print(df)

#     # Save CSV
#     df.to_csv("final_model_comparison.csv", index=False)
#     print("\n✅ Saved results to final_model_comparison.csv")

#     # =========================
#     # PLOT GRAPH
#     # =========================
#     df_plot = df.set_index("Model")[["BLEU","METEOR","ROUGE-L","CIDEr","SPICE"]]
#     df_plot.plot(kind="bar", figsize=(12,6))
#     plt.title("Captioning Model Comparison (Matched Videos Only)")
#     plt.ylabel("Score")
#     plt.ylim(0,1)
#     plt.grid(axis="y")
#     plt.tight_layout()
#     plt.show()

# if __name__ == "__main__":
#     main()


import nltk
import pandas as pd
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import os

# For CIDEr
from pycocoevalcap.cider.cider import Cider

# =========================
# DOWNLOAD REQUIRED NLTK DATA
# =========================
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")

# =========================
# FILE PATHS
# =========================
reference_file = "captions.txt"

model_files = {
    "BLIP": "blip_cap.txt",
    "GIT": "git_cap.txt",
    "ViT-GPT2": "vitgpt2_cap.txt"
}

# =========================
# LOAD REFERENCE FILE
# Format: videoID caption
# =========================
def load_reference(file):
    data = {}
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                vid, cap = parts
                data[vid.strip()] = cap.strip()
    return data

# =========================
# LOAD MODEL FILE
# Format: videoID.avi: caption
# =========================
def load_predictions(file):
    data = {}
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            if ":" in line:
                vid, cap = line.strip().split(":", 1)
                vid = os.path.splitext(vid)[0].strip()  # remove .avi or other extensions
                data[vid] = cap.strip()
    return data

# =========================
# COMPUTE METRICS
# =========================
def compute_scores(pred_dict, ref_dict, model_name):
    smoothie = SmoothingFunction().method4
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    bleu_list, meteor_list, rouge_list = [], [], []

    # Prepare for CIDEr
    preds_for_cider = {}
    refs_for_cider = {}

    # Only compare videos present in both sets
    common_ids = set(pred_dict.keys()) & set(ref_dict.keys())
    print(f"{model_name} matched videos:", len(common_ids))

    if len(common_ids) == 0:
        print(f"⚠ No matching videos for {model_name}, skipping...")
        return 0,0,0,0,0

    for vid in common_ids:
        pred = pred_dict[vid].lower()
        ref = ref_dict[vid].lower()

        if not pred or not ref:
            continue

        # Tokenize for BLEU & METEOR
        pred_tokens = nltk.word_tokenize(pred)
        ref_tokens = nltk.word_tokenize(ref)

        bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothie)
        meteor = meteor_score([ref_tokens], pred_tokens)
        rouge_l = rouge.score(ref, pred)["rougeL"].fmeasure

        bleu_list.append(bleu)
        meteor_list.append(meteor)
        rouge_list.append(rouge_l)

        # Wrap in list for pycocoevalcap
        preds_for_cider[vid] = [pred]
        refs_for_cider[vid] = [ref]

    # =========================
    # CIDEr
    # =========================
    cider_score = 0
    try:
        cider_scorer = Cider()
        cider_score, _ = cider_scorer.compute_score(refs_for_cider, preds_for_cider)
    except Exception as e:
        print(f"⚠ CIDEr error for {model_name}: {e}")

    n = max(len(bleu_list), 1)
    return (
        sum(bleu_list)/n,
        sum(meteor_list)/n,
        sum(rouge_list)/n,
        cider_score,
        len(common_ids)
    )

# =========================
# MAIN EVALUATION
# =========================
def main():
    references = load_reference(reference_file)
    print("Reference captions loaded:", len(references))

    results = []

    for model_name, file in model_files.items():
        if not os.path.exists(file):
            print(f"⚠ File not found for {model_name}: {file}, skipping...")
            continue

        print(f"\nEvaluating {model_name}...")
        preds = load_predictions(file)
        bleu, meteor, rouge_l, cider, matched = compute_scores(preds, references, model_name)
        results.append([model_name, bleu, meteor, rouge_l, cider, matched])

    # =========================
    # RESULTS TABLE
    # =========================
    df = pd.DataFrame(results, columns=["Model","BLEU","METEOR","ROUGE-L","CIDEr","Matched Videos"])
    print("\n=== FINAL RESULTS ===")
    print(df)

    # Save CSV
    df.to_csv("final_model_comparison.csv", index=False)
    print("\n Saved results to final_model_comparison.csv")

    # =========================
    # PLOT GRAPH
    # =========================
    df_plot = df.set_index("Model")[["BLEU","METEOR","ROUGE-L","CIDEr"]]
    df_plot.plot(kind="bar", figsize=(12,6))
    plt.title("Captioning Model Comparison (Matched Videos Only)")
    plt.ylabel("Score")
    plt.ylim(0,1)
    plt.grid(axis="y")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

