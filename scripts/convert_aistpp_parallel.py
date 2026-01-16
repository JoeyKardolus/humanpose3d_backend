#!/usr/bin/env python3
"""
Parallel AIST++ conversion - runs multiple workers on different sequence ranges.

Usage:
    uv run python scripts/convert_aistpp_parallel.py --workers 4
"""

import argparse
import sys
from pathlib import Path
from multiprocessing import Pool
import cv2
import numpy as np
from tqdm import tqdm
import mediapipe as mp

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.convert_aistpp_to_training import (
    load_camera_params,
    get_camera_setting,
    compute_view_angles,
    extract_mediapipe_coco,
    center_on_pelvis,
    normalize_pose_scale,
    load_keypoints3d,
    get_video_name,
)


def process_single_video(args):
    """Process a single video file."""
    video_path, gt_keypoints, output_dir, seq_name_cam, camera_pos, max_frames, frame_skip = args

    existing = list(output_dir.glob(f"{seq_name_cam}_f*.npz"))
    if len(existing) >= max_frames // 2:
        return len(existing)

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    gt_frames = len(gt_keypoints)
    n_frames = min(total_frames, gt_frames, max_frames * frame_skip)

    examples_generated = 0
    frame_idx = 0

    while frame_idx < n_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue

        gt_pose = gt_keypoints[frame_idx]

        if np.isnan(gt_pose).any() or np.allclose(gt_pose, 0):
            frame_idx += 1
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if not results.pose_world_landmarks or not results.pose_landmarks:
            frame_idx += 1
            continue

        mp_pose_data, visibility, pose_2d = extract_mediapipe_coco(results)
        azimuth, elevation = compute_view_angles(gt_pose, camera_pos)

        gt_pelvis = (gt_pose[11] + gt_pose[12]) / 2
        camera_relative = camera_pos - gt_pelvis

        mp_centered = center_on_pelvis(mp_pose_data)
        gt_centered = center_on_pelvis(gt_pose)

        mp_centered[:, 1] = -mp_centered[:, 1]
        mp_centered[:, 2] = -mp_centered[:, 2]

        mp_normalized, mp_scale = normalize_pose_scale(mp_centered, target_scale=1.0)
        gt_normalized, gt_scale = normalize_pose_scale(gt_centered, target_scale=1.0)

        if mp_scale < 0.01 or gt_scale < 0.01:
            frame_idx += 1
            continue

        if mp_normalized[0, 1] < 0.5 or gt_normalized[0, 1] < 0.5:
            frame_idx += 1
            continue

        output_path = output_dir / f"{seq_name_cam}_f{frame_idx:06d}.npz"

        if output_path.exists():
            examples_generated += 1
            frame_idx += 1
            continue

        np.savez_compressed(
            output_path,
            corrupted=mp_normalized.astype(np.float32),
            ground_truth=gt_normalized.astype(np.float32),
            visibility=visibility.astype(np.float32),
            pose_2d=pose_2d.astype(np.float32),
            azimuth=np.float32(azimuth),
            elevation=np.float32(elevation),
            camera_relative=camera_relative.astype(np.float32),
            mp_scale=np.float32(mp_scale),
            gt_scale=np.float32(gt_scale),
            sequence=seq_name_cam,
            frame_idx=frame_idx,
        )

        examples_generated += 1
        frame_idx += 1

    cap.release()
    pose.close()
    return examples_generated


def main():
    parser = argparse.ArgumentParser(description='Parallel AIST++ conversion')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--max-frames', type=int, default=180, help='Max frames per video')
    parser.add_argument('--frame-skip', type=int, default=3, help='Frame skip rate')
    args = parser.parse_args()

    print("=" * 60)
    print(f"PARALLEL AIST++ CONVERSION ({args.workers} workers)")
    print("=" * 60)

    aistpp_dir = Path("data/AIST++")
    annotations_dir = aistpp_dir / "annotations"
    videos_dir = aistpp_dir / "videos"
    cameras_dir = annotations_dir / "cameras"
    mapping_file = cameras_dir / "mapping.txt"
    output_dir = Path("data/training/aistpp_converted")
    output_dir.mkdir(parents=True, exist_ok=True)

    keypoints_dir = annotations_dir / "keypoints3d"
    kp_files = sorted(keypoints_dir.glob("*.pkl"))
    video_files = list(videos_dir.glob("*.mp4"))
    video_dict = {v.stem: v for v in video_files}

    print(f"Found {len(kp_files)} sequences, {len(video_files)} videos")

    camera_views = ['c01', 'c02', 'c03', 'c05', 'c07', 'c09']
    camera_cache = {}

    tasks = []
    print("Building task list...")
    for kp_file in tqdm(kp_files, desc="Loading sequences"):
        seq_name = kp_file.stem
        try:
            gt_keypoints = load_keypoints3d(kp_file)
        except:
            continue

        setting = get_camera_setting(seq_name, mapping_file)

        for cam_view in camera_views:
            video_name = get_video_name(seq_name, cam_view)
            video_path = video_dict.get(video_name)
            if video_path is None:
                continue

            cache_key = f"{setting}_{cam_view}"
            if cache_key not in camera_cache:
                try:
                    camera_cache[cache_key] = load_camera_params(cameras_dir, setting, cam_view)
                except:
                    continue

            camera_pos = camera_cache[cache_key]['position']
            seq_name_cam = f"{seq_name}_{cam_view}"

            tasks.append((
                video_path, gt_keypoints, output_dir, seq_name_cam,
                camera_pos, args.max_frames, args.frame_skip,
            ))

    print(f"Created {len(tasks)} tasks")
    print(f"Starting {args.workers} parallel workers...")

    total = 0
    with Pool(args.workers) as pool:
        for count in tqdm(pool.imap_unordered(process_single_video, tasks), total=len(tasks)):
            total += count

    final_count = len(list(output_dir.glob("*.npz")))
    print(f"\nDone! Total samples: {final_count}")


if __name__ == "__main__":
    main()
