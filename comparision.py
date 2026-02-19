import nltk
import matplotlib.pyplot as plt
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
# =========================
# FILE PATHS
# =========================
ref_file = "msvd_references.txt"

bleu_scores = []
meteor_scores = []
rouge_scores = []


model_files = {
    "BLIP": "video_captionsBlip.txt",
    "GIT": "git_captions.txt",
    "ViT-GPT2": "vitgpt2_captions.txt"
}

# =========================
# LOAD FILE
# =========================
def load_captions(file):
    captions = []
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            if ":" in line:
                captions.append(line.strip().split(":", 1)[1].strip())
    return captions

references = load_captions(ref_file)

# =========================
# METRIC FUNCTIONS
# =========================
smoothie = SmoothingFunction().method4
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def evaluate_model(preds, refs):
    bleu_scores = []
    meteor_scores = []
    rouge_scores = []

    for pred, ref in zip(preds, refs):
        ref_tokens = nltk.word_tokenize(ref.lower())
        pred_tokens = nltk.word_tokenize(pred.lower())

        bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothie)
        meteor = meteor_score([ref], pred)
        rouge = scorer.score(ref, pred)['rougeL'].fmeasure

        bleu_scores.append(bleu)
        meteor_scores.append(meteor)
        rouge_scores.append(rouge)

    return (
        sum(bleu_scores)/len(bleu_scores),
        sum(meteor_scores)/len(meteor_scores),
        sum(rouge_scores)/len(rouge_scores)
    )

# =========================
# EVALUATE ALL MODELS
# =========================
results = []

for model_name, file in model_files.items():
    preds = load_captions(file)
    bleu, meteor, rouge = evaluate_model(preds, references)
    results.append([model_name, bleu, meteor, rouge])

df = pd.DataFrame(results, columns=["Model", "BLEU", "METEOR", "ROUGE-L"])
print("\n=== Evaluation Results ===")
print(df)

# =========================
# SAVE CSV
# =========================
df.to_csv("model_comparison.csv", index=False)

# =========================
# PLOT GRAPH
# =========================
df.plot(x="Model", kind="bar", figsize=(8,5))
plt.title("Captioning Model Comparison")
plt.ylabel("Score")
plt.ylim(0,1)
plt.grid(True)
plt.show()
