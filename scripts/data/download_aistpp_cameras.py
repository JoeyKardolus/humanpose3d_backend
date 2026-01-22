#!/usr/bin/env python3
"""Download specific camera views from AIST++ dataset.

Modified from official downloader to only download requested cameras.
"""
import multiprocessing
import os
import sys
import urllib.request
from functools import partial

SOURCE_URL = 'https://aistdancedb.ongaaccel.jp/v1.0.0/video/10M/'
LIST_URL = 'https://storage.googleapis.com/aist_plusplus_public/20121228/video_list.txt'


def _download(video_url, download_folder):
    save_path = os.path.join(download_folder, os.path.basename(video_url))
    if os.path.exists(save_path):
        return  # Skip if already exists
    try:
        urllib.request.urlretrieve(video_url, save_path)
    except Exception as e:
        print(f"\nFailed to download {video_url}: {e}", file=sys.stderr)


def download_cameras(cameras, download_folder, num_processes=4):
    """Download videos for specific camera views."""
    os.makedirs(download_folder, exist_ok=True)

    # Get video list
    print("Fetching video list...")
    seq_names = urllib.request.urlopen(LIST_URL)
    seq_names = [seq_name.strip().decode('utf-8') for seq_name in seq_names]

    # Filter to requested cameras
    filtered = []
    for seq in seq_names:
        for cam in cameras:
            if f'_{cam}_' in seq:
                filtered.append(seq)
                break

    print(f"Found {len(filtered)} videos for cameras {cameras}")

    # Check which are already downloaded
    existing = set(os.listdir(download_folder)) if os.path.exists(download_folder) else set()
    to_download = [seq for seq in filtered if seq + '.mp4' not in existing]
    print(f"Need to download {len(to_download)} videos ({len(filtered) - len(to_download)} already exist)")

    if not to_download:
        print("All videos already downloaded!")
        return

    video_urls = [os.path.join(SOURCE_URL, seq + '.mp4') for seq in to_download]

    download_func = partial(_download, download_folder=download_folder)
    pool = multiprocessing.Pool(processes=num_processes)
    for i, _ in enumerate(pool.imap_unordered(download_func, video_urls)):
        sys.stderr.write(f'\rdownloading {i + 1} / {len(video_urls)}')
    sys.stderr.write('\ndone.\n')
    pool.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cameras', nargs='+', default=['c04', 'c05', 'c06', 'c07', 'c08', 'c09'],
                       help='Camera views to download')
    default_download = os.path.join(os.path.expanduser('~'), '.humanpose3d', 'training', 'AIST++', 'videos')
    parser.add_argument('--download_folder', default=default_download,
                       help='Where to store videos')
    parser.add_argument('--num_processes', type=int, default=4,
                       help='Number of parallel downloads')
    args = parser.parse_args()

    print("=" * 60)
    print("AIST++ Video Downloader")
    print("=" * 60)
    print(f"Cameras: {args.cameras}")
    print(f"Output: {args.download_folder}")
    print("=" * 60)
    print("\nBy running this script, you agree to AIST++ Terms of Use:")
    print("https://aistdancedb.ongaaccel.jp/terms_of_use/\n")

    download_cameras(args.cameras, args.download_folder, args.num_processes)
