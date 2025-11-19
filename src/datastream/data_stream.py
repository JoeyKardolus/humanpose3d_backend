from pathlib import Path
import csv


class DataStream:
    def __init__(self, data_path: Path) -> None:
        self.data_path: Path = data_path

    def save_to_csv(self, landmarks) -> None:
        """Save landmarks to CSV file efficiently."""
        with open(self.data_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                ["Frame", "Landmark", "X", "Y", "Z", "Visibility", "Presence"]
            )

            rows = [
                [frame, landmark_idx, lm.x, lm.y, lm.z, lm.visibility, lm.presence]
                for frame, frame_landmarks in enumerate(landmarks)
                for pose_landmarks in frame_landmarks
                for landmark_idx, lm in enumerate(pose_landmarks)
            ]
            writer.writerows(rows)
