
# import nltk
# import pandas as pd
# import matplotlib.pyplot as plt
# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# from nltk.translate.meteor_score import meteor_score
# from rouge_score import rouge_scorer
# import os

# # For CIDEr
# from pycocoevalcap.cider.cider import Cider

# # =========================
# # DOWNLOAD REQUIRED NLTK DATA
# # =========================
# nltk.download("punkt")
# nltk.download("wordnet")
# nltk.download("omw-1.4")

# # =========================
# # FILE PATHS
# # =========================
# reference_file = "msvd_references.txt"

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

#     # Prepare for CIDEr
#     preds_for_cider = {}
#     refs_for_cider = {}

#     # Only compare videos present in both sets
#     common_ids = set(pred_dict.keys()) & set(ref_dict.keys())
#     print(f"{model_name} matched videos:", len(common_ids))

#     if len(common_ids) == 0:
#         print(f"⚠ No matching videos for {model_name}, skipping...")
#         return 0,0,0,0,0

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
#     # CIDEr
#     # =========================
#     cider_score = 0
#     try:
#         cider_scorer = Cider()
#         cider_score, _ = cider_scorer.compute_score(refs_for_cider, preds_for_cider)
#     except Exception as e:
#         print(f"⚠ CIDEr error for {model_name}: {e}")

#     n = max(len(bleu_list), 1)
#     return (
#         sum(bleu_list)/n,
#         sum(meteor_list)/n,
#         sum(rouge_list)/n,
#         cider_score,
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
#         bleu, meteor, rouge_l, cider, matched = compute_scores(preds, references, model_name)
#         results.append([model_name, bleu, meteor, rouge_l, cider, matched])

#     # =========================
#     # RESULTS TABLE
#     # =========================
#     df = pd.DataFrame(results, columns=["Model","BLEU","METEOR","ROUGE-L","CIDEr","Matched Videos"])
#     print("\n=== FINAL RESULTS ===")
#     print(df)

#     # Save CSV
#     df.to_csv("final_model_comparison.csv", index=False)
#     print("\n Saved results to final_model_comparison.csv")

#     # =========================
#     # PLOT GRAPH
#     # =========================
#     df_plot = df.set_index("Model")[["BLEU","METEOR","ROUGE-L","CIDEr"]]
#     df_plot.plot(kind="bar", figsize=(12,6))
#     plt.title("Captioning Model Comparison")
#     plt.ylabel("Score")
#     plt.ylim(0,1)
#     plt.grid(axis="y")
#     plt.tight_layout()
#     plt.show()

# if __name__ == "__main__":
#     main()

# import os
# import nltk
# import torch
# import pandas as pd
# from bert_score import score as bert_score
# from pycocoevalcap.spice.spice import Spice
# from sentence_transformers import SentenceTransformer, util

# nltk.download("punkt")

# # =========================
# # FILE PATHS
# # =========================
# reference_file = "msvd_references.txt"

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
#                 vid = os.path.splitext(vid)[0].strip()
#                 data[vid] = cap.strip()
#     return data

# # =========================
# # COMPUTE NEW METRICS
# # =========================
# def compute_new_metrics(pred_dict, ref_dict, model_name):
#     common_ids = set(pred_dict.keys()) & set(ref_dict.keys())
#     print(f"{model_name} matched videos:", len(common_ids))

#     if len(common_ids) == 0:
#         return 0,0,0,0

#     preds = []
#     refs = []

#     for vid in common_ids:
#         preds.append(pred_dict[vid])
#         refs.append(ref_dict[vid])

#     # =========================
#     # BERTScore
#     # =========================
#     P, R, F1 = bert_score(preds, refs, lang="en", verbose=False)
#     bert_f1 = F1.mean().item()

#     # =========================
#     # SPICE
#     # =========================
#     spice_scorer = Spice()
#     spice_refs = {i: [refs[i]] for i in range(len(refs))}
#     spice_preds = {i: [preds[i]] for i in range(len(preds))}
#     spice_score, _ = spice_scorer.compute_score(spice_refs, spice_preds)

#     # =========================
#     # CLIPScore (text-text similarity)
#     # =========================
#     model = SentenceTransformer("all-MiniLM-L6-v2")
#     ref_emb = model.encode(refs, convert_to_tensor=True)
#     pred_emb = model.encode(preds, convert_to_tensor=True)
#     clip_score = util.cos_sim(pred_emb, ref_emb).mean().item()

#     return bert_f1, spice_score, clip_score, len(common_ids)

# # =========================
# # MAIN
# # =========================
# def main():
#     references = load_reference(reference_file)
#     print("Reference captions loaded:", len(references))

#     results = []

#     for model_name, file in model_files.items():
#         if not os.path.exists(file):
#             print(f"File not found: {file}")
#             continue

#         preds = load_predictions(file)
#         bert, spice, clipscore, matched = compute_new_metrics(preds, references, model_name)

#         results.append([model_name, bert, spice, clipscore, matched])

#     df = pd.DataFrame(results, columns=["Model","BERTScore","SPICE","CLIPScore","Matched Videos"])
#     print("\n=== NEW METRIC RESULTS ===")
#     print(df)

#     df.to_csv("new_metrics_results.csv", index=False)
#     print("Saved to new_metrics_results.csv")

# if __name__ == "__main__":
#     main()

import os
import nltk
import torch
import pandas as pd
import matplotlib.pyplot as plt

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from pycocoevalcap.cider.cider import Cider

from bert_score import score as bert_score
from sentence_transformers import SentenceTransformer, util

# =========================
# DOWNLOAD NLTK DATA
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

device = "cuda" if torch.cuda.is_available() else "cpu"

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
                vid = os.path.splitext(vid)[0].strip()
                data[vid] = cap.strip()
    return data

# =========================
# METRIC COMPUTATION
# =========================
def compute_scores(pred_dict, ref_dict, model_name):
    smoothie = SmoothingFunction().method4
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    bleu_list, meteor_list, rouge_list = [], [], []

    preds_for_cider = {}
    refs_for_cider = {}

    common_ids = set(pred_dict.keys()) & set(ref_dict.keys())
    print(f"{model_name} matched videos:", len(common_ids))

    if len(common_ids) == 0:
        return 0,0,0,0,0,0

    pred_sentences = []
    ref_sentences = []

    for vid in common_ids:
        pred = pred_dict[vid].lower()
        ref = ref_dict[vid].lower()

        if not pred or not ref:
            continue

        pred_tokens = nltk.word_tokenize(pred)
        ref_tokens = nltk.word_tokenize(ref)

        bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothie)
        meteor = meteor_score([ref_tokens], pred_tokens)
        rouge_l = rouge.score(ref, pred)["rougeL"].fmeasure

        bleu_list.append(bleu)
        meteor_list.append(meteor)
        rouge_list.append(rouge_l)

        preds_for_cider[vid] = [pred]
        refs_for_cider[vid] = [ref]

        pred_sentences.append(pred)
        ref_sentences.append(ref)

    # =========================
    # CIDEr
    # =========================
    cider_scorer = Cider()
    cider_score, _ = cider_scorer.compute_score(refs_for_cider, preds_for_cider)

    # =========================
    # BERTScore
    # =========================
    P, R, F1 = bert_score(pred_sentences, ref_sentences, lang="en", rescale_with_baseline=True)
    bert_f1 = F1.mean().item()

    # =========================
    # CLIPScore (SentenceTransformer)
    # =========================
    clip_model = SentenceTransformer("clip-ViT-B-32")
    emb_preds = clip_model.encode(pred_sentences, convert_to_tensor=True)
    emb_refs = clip_model.encode(ref_sentences, convert_to_tensor=True)
    clip_sim = util.cos_sim(emb_preds, emb_refs).diagonal().mean().item()

    n = len(bleu_list)

    return (
        sum(bleu_list)/n,
        sum(meteor_list)/n,
        sum(rouge_list)/n,
        cider_score,
        bert_f1,
        clip_sim,
        len(common_ids)
    )

# =========================
# MAIN
# =========================
def main():
    references = load_reference(reference_file)
    print("Reference captions loaded:", len(references))

    results = []

    for model_name, file in model_files.items():
        print(f"\nEvaluating {model_name}...")
        preds = load_predictions(file)
        bleu, meteor, rouge_l, cider, bert, clip, matched = compute_scores(preds, references, model_name)

        results.append([model_name, bleu, meteor, rouge_l, cider, bert, clip, matched])

    df = pd.DataFrame(results, columns=[
        "Model","BLEU","METEOR","ROUGE-L","CIDEr","BERTScore","CLIPScore","Matched Videos"
    ])

    print("\n=== FINAL RESULTS ===")
    print(df)

    df.to_csv("final_model_comparison_with_bert_clip.csv", index=False)
    print("\nSaved to final_model_comparison_with_bert_clip.csv")

    # =========================
    # PLOT
    # =========================
    df_plot = df.set_index("Model")[["BLEU","METEOR","ROUGE-L","CIDEr","BERTScore","CLIPScore"]]
    df_plot.plot(kind="bar", figsize=(14,6))
    plt.title("Captioning Model Comparison (With BERTScore & CLIPScore)")
    plt.ylabel("Score")
    plt.ylim(0,1)
    plt.grid(axis="y")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
