from pathlib import Path
from src.mediastream.media_stream import MediaStream
from src.posedetector.pose_detector import PoseDetector
from src.visualizedata.visualize_data import VisualizeData
from src.datastream.data_stream import DataStream


def main() -> None:
    name: str = "Max"
    input_video_path = Path("data", "input", f"{name}.mp4")
    output_csv_path = Path("data", "output", f"{name}.csv")
    model_path = Path("models", "pose_landmarker_heavy.task")

    ms = MediaStream()
    pd = PoseDetector(model_path)
    ds = DataStream(output_csv_path)
    vd = VisualizeData()

    video = ms.read_video(input_video_path)
    landmarks = pd.detect_pose_landmarks(video, True)

    ds.save_to_csv(landmarks)
    vd.plot_landmarks(landmarks)


if __name__ == "__main__":
    main()
