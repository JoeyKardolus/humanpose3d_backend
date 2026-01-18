#!/usr/bin/env python3
"""
Clean MTC annotation.pkl by removing hand data (we only need body poses).

This reduces the annotation file size significantly and speeds up loading.
The original file is backed up before modification.

Usage:
    python scripts/data/clean_mtc_annotation.py
    python scripts/data/clean_mtc_annotation.py --dry-run  # Preview without changes
    python scripts/data/clean_mtc_annotation.py --stats    # Show statistics only
"""

import argparse
import pickle
import shutil
from pathlib import Path
from datetime import datetime


def get_annotation_stats(data: dict) -> dict:
    """Get statistics about the annotation data."""
    stats = {
        'training_samples': 0,
        'testing_samples': 0,
        'samples_with_body': 0,
        'samples_with_left_hand': 0,
        'samples_with_right_hand': 0,
        'unique_sequences': set(),
        'body_landmarks_total': 0,
        'left_hand_landmarks_total': 0,
        'right_hand_landmarks_total': 0,
    }

    for split in ['training_data', 'testing_data']:
        samples = data.get(split, [])
        if split == 'training_data':
            stats['training_samples'] = len(samples)
        else:
            stats['testing_samples'] = len(samples)

        for sample in samples:
            stats['unique_sequences'].add(sample.get('seqName', 'unknown'))

            if 'body' in sample and sample['body'].get('landmarks'):
                stats['samples_with_body'] += 1
                stats['body_landmarks_total'] += len(sample['body']['landmarks'])

            if 'left_hand' in sample and sample['left_hand'].get('landmarks'):
                stats['samples_with_left_hand'] += 1
                stats['left_hand_landmarks_total'] += len(sample['left_hand']['landmarks'])

            if 'right_hand' in sample and sample['right_hand'].get('landmarks'):
                stats['samples_with_right_hand'] += 1
                stats['right_hand_landmarks_total'] += len(sample['right_hand']['landmarks'])

    stats['unique_sequences'] = len(stats['unique_sequences'])
    return stats


def clean_annotation_data(data: dict) -> dict:
    """Remove hand data from annotation, keeping only body data."""
    cleaned = {}

    for split in ['training_data', 'testing_data']:
        if split not in data:
            continue

        cleaned_samples = []
        for sample in data[split]:
            cleaned_sample = {
                'seqName': sample.get('seqName'),
                'frame_str': sample.get('frame_str'),
                'id': sample.get('id'),
            }

            # Keep only body data
            if 'body' in sample:
                cleaned_sample['body'] = sample['body']

            # Explicitly remove hand data by not copying it
            # 'left_hand' and 'right_hand' are omitted

            cleaned_samples.append(cleaned_sample)

        cleaned[split] = cleaned_samples

    return cleaned


def main():
    parser = argparse.ArgumentParser(description="Clean MTC annotation by removing hand data")
    parser.add_argument("--mtc-dir", type=str,
                       default="/home/dupe/ai-test-project/humanpose3d_mediapipe/data/mtc/a4_release",
                       help="Path to MTC a4_release directory")
    parser.add_argument("--dry-run", action="store_true",
                       help="Preview changes without modifying files")
    parser.add_argument("--stats", action="store_true",
                       help="Show statistics only")
    parser.add_argument("--no-backup", action="store_true",
                       help="Don't create backup of original file")

    args = parser.parse_args()

    mtc_dir = Path(args.mtc_dir)
    annotation_file = mtc_dir / "annotation.pkl"

    if not annotation_file.exists():
        print(f"ERROR: annotation.pkl not found at {annotation_file}")
        return 1

    # Get original file size
    original_size = annotation_file.stat().st_size / (1024 * 1024)
    print(f"Loading annotation file ({original_size:.1f} MB)...")

    with open(annotation_file, 'rb') as f:
        data = pickle.load(f)

    # Show statistics
    print("\n" + "=" * 60)
    print("ORIGINAL ANNOTATION STATISTICS")
    print("=" * 60)

    stats = get_annotation_stats(data)
    print(f"  Training samples:      {stats['training_samples']:,}")
    print(f"  Testing samples:       {stats['testing_samples']:,}")
    print(f"  Unique sequences:      {stats['unique_sequences']}")
    print(f"  Samples with body:     {stats['samples_with_body']:,}")
    print(f"  Samples with L hand:   {stats['samples_with_left_hand']:,}")
    print(f"  Samples with R hand:   {stats['samples_with_right_hand']:,}")
    print(f"\n  Body landmarks:        {stats['body_landmarks_total']:,} values")
    print(f"  Left hand landmarks:   {stats['left_hand_landmarks_total']:,} values")
    print(f"  Right hand landmarks:  {stats['right_hand_landmarks_total']:,} values")

    # Calculate estimated savings
    hand_data_pct = (stats['left_hand_landmarks_total'] + stats['right_hand_landmarks_total']) / \
                    (stats['body_landmarks_total'] + stats['left_hand_landmarks_total'] + stats['right_hand_landmarks_total'] + 1) * 100
    print(f"\n  Hand data percentage:  {hand_data_pct:.1f}%")
    print(f"  Estimated new size:    {original_size * (100 - hand_data_pct) / 100:.1f} MB")

    if args.stats:
        return 0

    if args.dry_run:
        print("\n[DRY RUN] Would clean annotation and remove hand data")
        return 0

    # Clean the data
    print("\n" + "=" * 60)
    print("CLEANING ANNOTATION DATA")
    print("=" * 60)

    cleaned_data = clean_annotation_data(data)

    # Verify cleaned data
    cleaned_stats = get_annotation_stats(cleaned_data)
    print(f"  Body samples preserved: {cleaned_stats['samples_with_body']:,}")
    print(f"  Hand samples removed:   {stats['samples_with_left_hand'] + stats['samples_with_right_hand']:,}")

    # Backup original
    if not args.no_backup:
        backup_file = mtc_dir / f"annotation_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        print(f"\n  Backing up original to: {backup_file.name}")
        shutil.copy2(annotation_file, backup_file)

    # Save cleaned data
    print(f"  Writing cleaned annotation...")
    with open(annotation_file, 'wb') as f:
        pickle.dump(cleaned_data, f)

    new_size = annotation_file.stat().st_size / (1024 * 1024)
    savings = original_size - new_size
    print(f"\n  Original size:  {original_size:.1f} MB")
    print(f"  New size:       {new_size:.1f} MB")
    print(f"  Saved:          {savings:.1f} MB ({savings/original_size*100:.1f}%)")

    print("\n" + "=" * 60)
    print("DONE - Hand data removed from annotation.pkl")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
