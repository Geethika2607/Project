import cv2

video_path = "E:\Final Project\Project\YouTubeClips\_0nX-El-ySo_83_93.avi"  # put your video file path here

cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Video Player", frame)

    # Press 'q' to quit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
