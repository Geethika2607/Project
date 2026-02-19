import os, cv2, pickle, torch
from PIL import Image
from transformers import GitProcessor, GitForCausalLM

video_dir = "YouTubeClips"
frames_dir = "frames_git"
os.makedirs(frames_dir, exist_ok=True)

txt_file = "git_captions.txt"
results = []


pkl_file = "git_captions.pkl"
txt_file = "git_captions.txt"
fps_interval_sec = 2

video_files = [f for f in os.listdir(video_dir) if f.endswith(".avi")]
video_path = os.path.join(video_dir, video_files[0])
print("Using video:", video_path)

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = GitProcessor.from_pretrained("microsoft/git-base")
model = GitForCausalLM.from_pretrained("microsoft/git-base").to(device)

def caption_image(img_path):
    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=40)
    return processor.batch_decode(out, skip_special_tokens=True)[0]

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS) or 25
interval = int(fps * fps_interval_sec)

frame_id = 0
segment = 0
results = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_id % interval == 0:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_path = f"{frames_dir}/frame_{segment}.png"
        Image.fromarray(rgb).save(img_path)

        caption = caption_image(img_path)
        results.append((segment*fps_interval_sec, caption))
        segment += 1

    display_text = results[-1][1] if results else "Starting..."
    cv2.putText(frame, display_text, (20,40), cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
    cv2.imshow("GIT Captioning", frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    frame_id += 1

cap.release()
cv2.destroyAllWindows()

with open(pkl_file,"wb") as f: pickle.dump(results,f)
with open(txt_file,"w") as f:
    for t,c in results: f.write(f"{t}s : {c}\n")

print("GIT Done")
