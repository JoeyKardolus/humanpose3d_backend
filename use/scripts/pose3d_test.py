import cv2
import csv
import time
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# CSV voor world-landmarks (meters, camera-onafhankelijk)
landmark_names = [l.name for l in mp_pose.PoseLandmark]
csv_path = "pose_world_landmarks.csv"
csv_file = open(csv_path, "w", newline="", encoding="utf-8")
writer = csv.writer(csv_file)
writer.writerow(["timestamp_s", "landmark", "x_m", "y_m", "z_m", "visibility"])

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Kon de webcam niet openen. Probeer een andere index (1/2) of sluit apps die de camera gebruiken.")

with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:

    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Geen frame van de camera ontvangen.")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(thickness=2)
            )

        # Schrijf world-landmarks naar CSV (als beschikbaar)
        if results.pose_world_landmarks:
            ts = time.time() - t0
            for i, lm in enumerate(results.pose_world_landmarks.landmark):
                writer.writerow([f"{ts:.3f}", landmark_names[i], f"{lm.x:.6f}", f"{lm.y:.6f}", f"{lm.z:.6f}", f"{lm.visibility:.3f}"])

        # FPS tonen
        cv2.putText(frame, "Druk 'q' om te stoppen", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.imshow("MediaPipe Pose — live + CSV", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
csv_file.close()
print(f"CSV opgeslagen als: {csv_path}")
