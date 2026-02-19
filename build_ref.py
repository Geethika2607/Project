import os

video_dir = "YouTubeClips"
ann_file = "captions.txt"
ref_out = "msvd_references.txt"

annotations = {}

with open(ann_file, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) == 2:
            vid, caption = parts
            annotations.setdefault(vid, []).append(caption)

with open(ref_out, "w", encoding="utf-8") as out:
    for video in os.listdir(video_dir):
        vid = os.path.splitext(video)[0]
        if vid in annotations:
            out.write(f"{vid}: {annotations[vid][0]}\n")

print("msvd_references.txt created")
