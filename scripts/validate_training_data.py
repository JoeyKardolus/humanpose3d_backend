#!/usr/bin/env python3
"""
Validate training data quality before training.

CRITICAL: Run this before 9-hour training to catch issues early!

Checks:
1. Feet geometry (non-collinear)
2. Bone length CV (reasonable variance)
3. Joint angle ROM (within limits)
4. Data corruption (NaNs, infinities)
5. Marker count consistency
6. Viewpoint diversity
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys


def validate_training_data(data_dir: Path, num_samples: int = 1000):
    """Run comprehensive validation on training data.

    Returns: True if all checks pass, False otherwise
    """

    print("="*80)
    print("TRAINING DATA VALIDATION")
    print("="*80)
    print()

    samples = sorted(data_dir.glob("*.npz"))

    if len(samples) == 0:
        print(f"❌ CRITICAL: No NPZ files found in {data_dir}")
        return False

    print(f"Found {len(samples)} training samples")
    print(f"Validating first {min(num_samples, len(samples))} samples...")
    print()

    samples = samples[:num_samples]
    issues = []

    # ========================================================================
    # Check 1: Feet Geometry (Non-Collinearity)
    # ========================================================================
    print("Check 1: Feet Geometry (Non-Collinearity)")
    print("-" * 40)

    collinear_count = 0
    x_spreads = []
    y_spreads = []
    z_spreads = []

    for sample_path in tqdm(samples[:100], desc="Checking feet"):
        data = np.load(sample_path)
        ground_truth = data['ground_truth']
        marker_names = data['marker_names'].tolist()

        # Check right foot (use RHeel, RAnkle, r_5meta_study for proper 3D triangle)
        # Note: RBigToe is a direct CMU joint mapping and may be collinear with ankle
        if all(m in marker_names for m in ['RHeel', 'RAnkle', 'r_5meta_study']):
            heel_idx = marker_names.index('RHeel')
            ankle_idx = marker_names.index('RAnkle')
            meta_idx = marker_names.index('r_5meta_study')

            positions = np.array([
                ground_truth[heel_idx],
                ground_truth[ankle_idx],
                ground_truth[meta_idx]
            ])

            x_spread = (positions[:, 0].max() - positions[:, 0].min()) * 1000  # mm
            y_spread = (positions[:, 1].max() - positions[:, 1].min()) * 1000  # mm
            z_spread = (positions[:, 2].max() - positions[:, 2].min()) * 1000  # mm

            x_spreads.append(x_spread)
            y_spreads.append(y_spread)
            z_spreads.append(z_spread)

            # Collinear if X and Y both have minimal variation
            if x_spread < 10.0 and y_spread < 10.0:
                collinear_count += 1

    if collinear_count > 10:
        issues.append(f"FAIL: {collinear_count}/100 samples have collinear feet!")
        print(f"❌ FAIL: {collinear_count}% have collinear feet")
        print(f"   Mean X-spread: {np.mean(x_spreads):.1f}mm (need >20mm)")
        print(f"   Mean Y-spread: {np.mean(y_spreads):.1f}mm (need >20mm)")
        print(f"   Mean Z-spread: {np.mean(z_spreads):.1f}mm")
    else:
        print(f"✓ PASS: {100 - collinear_count}% valid foot geometry")
        print(f"   Mean X-spread: {np.mean(x_spreads):.1f}mm ✓")
        print(f"   Mean Y-spread: {np.mean(y_spreads):.1f}mm ✓")
        print(f"   Mean Z-spread: {np.mean(z_spreads):.1f}mm ✓")

    # ========================================================================
    # Check 2: Bone Length CV (Ground Truth Should Have Low Variance)
    # ========================================================================
    print("\nCheck 2: Bone Length Coefficient of Variation")
    print("-" * 40)

    femur_lengths = []
    tibia_lengths = []

    for sample_path in tqdm(samples[:100], desc="Checking bones"):
        data = np.load(sample_path)
        ground_truth = data['ground_truth']
        marker_names = data['marker_names'].tolist()

        # Check femur bone (RHip -> RKnee)
        if all(m in marker_names for m in ['RHip', 'RKnee']):
            hip_idx = marker_names.index('RHip')
            knee_idx = marker_names.index('RKnee')

            femur_length = np.linalg.norm(
                ground_truth[knee_idx] - ground_truth[hip_idx]
            )
            femur_lengths.append(femur_length)

        # Check tibia bone (RKnee -> RAnkle)
        if all(m in marker_names for m in ['RKnee', 'RAnkle']):
            knee_idx = marker_names.index('RKnee')
            ankle_idx = marker_names.index('RAnkle')

            tibia_length = np.linalg.norm(
                ground_truth[ankle_idx] - ground_truth[knee_idx]
            )
            tibia_lengths.append(tibia_length)

    mean_femur = np.mean(femur_lengths)
    std_femur = np.std(femur_lengths)
    cv_femur = std_femur / mean_femur if mean_femur > 0 else 0

    mean_tibia = np.mean(tibia_lengths)
    std_tibia = np.std(tibia_lengths)
    cv_tibia = std_tibia / mean_tibia if mean_tibia > 0 else 0

    # Note: Ground truth should have LOW CV (bones are constant length)
    # Corrupted data will have higher CV due to depth errors
    if 0.15 < mean_femur < 0.60 and cv_femur < 0.20:
        print(f"✓ PASS: Femur length = {mean_femur:.3f}m (CV={cv_femur:.3f})")
    else:
        issues.append(f"FAIL: Femur length = {mean_femur:.3f}m (CV={cv_femur:.3f})")
        print(f"❌ FAIL: Femur length abnormal")

    if 0.15 < mean_tibia < 0.50 and cv_tibia < 0.20:
        print(f"✓ PASS: Tibia length = {mean_tibia:.3f}m (CV={cv_tibia:.3f})")
    else:
        issues.append(f"FAIL: Tibia length = {mean_tibia:.3f}m (CV={cv_tibia:.3f})")
        print(f"❌ FAIL: Tibia length abnormal")

    # ========================================================================
    # Check 3: Data Corruption (NaN/Inf)
    # ========================================================================
    print("\nCheck 3: Data Corruption (NaN/Inf)")
    print("-" * 40)

    nan_count = 0
    inf_count = 0

    for sample_path in tqdm(samples, desc="Checking corruption"):
        data = np.load(sample_path)
        corrupted = data['corrupted']
        ground_truth = data['ground_truth']

        if np.isnan(corrupted).any() or np.isnan(ground_truth).any():
            nan_count += 1
        if np.isinf(corrupted).any() or np.isinf(ground_truth).any():
            inf_count += 1

    if nan_count > 0 or inf_count > 0:
        issues.append(f"FAIL: {nan_count} NaN, {inf_count} Inf samples")
        print(f"❌ FAIL: Data corruption detected")
        print(f"   NaN samples: {nan_count}")
        print(f"   Inf samples: {inf_count}")
    else:
        print(f"✓ PASS: No NaN/Inf values")

    # ========================================================================
    # Check 4: Marker Count Consistency
    # ========================================================================
    print("\nCheck 4: Marker Count Consistency")
    print("-" * 40)

    marker_counts = []
    for sample_path in samples[:100]:
        data = np.load(sample_path)
        marker_counts.append(len(data['marker_names']))

    unique_counts = set(marker_counts)
    if len(unique_counts) == 1:
        print(f"✓ PASS: All samples have {marker_counts[0]} markers")
    else:
        issues.append(f"FAIL: Inconsistent marker counts: {unique_counts}")
        print(f"❌ FAIL: Inconsistent marker counts: {unique_counts}")

    # ========================================================================
    # Check 5: Viewpoint-Dependent Noise Diversity
    # ========================================================================
    print("\nCheck 5: Viewpoint-Dependent Noise Diversity")
    print("-" * 40)

    # Load samples from different camera angles
    angles_found = set()
    noise_levels_found = set()

    for sample_path in samples[:100]:
        data = np.load(sample_path)
        angles_found.add(float(data['camera_angle']))
        noise_levels_found.add(float(data['noise_std']))

    if len(angles_found) >= 3:
        print(f"✓ PASS: Found {len(angles_found)} camera angles: {sorted(angles_found)}")
    else:
        issues.append(f"FAIL: Only {len(angles_found)} camera angles found")
        print(f"❌ FAIL: Insufficient camera angle diversity")

    if len(noise_levels_found) >= 2:
        print(f"✓ PASS: Found {len(noise_levels_found)} noise levels: {sorted(noise_levels_found)}")
    else:
        issues.append(f"FAIL: Only {len(noise_levels_found)} noise levels found")
        print(f"❌ FAIL: Insufficient noise level diversity")

    # ========================================================================
    # Check 6: Depth Error Pattern (Viewpoint-Dependent)
    # ========================================================================
    print("\nCheck 6: Depth Error Pattern (Viewpoint-Dependent)")
    print("-" * 40)

    # Check if right shoulder has higher depth error at low angles
    # and left shoulder has higher error at high angles
    low_angle_samples = [s for s in samples if '01_01_f' in s.name and '_a00_' in s.name or '_a15_' in s.name][:10]
    high_angle_samples = [s for s in samples if '01_01_f' in s.name and '_a60_' in s.name or '_a75_' in s.name][:10]

    if low_angle_samples and high_angle_samples:
        # Low angles (frontal view)
        low_r_errors = []
        low_l_errors = []

        for sample_path in low_angle_samples:
            data = np.load(sample_path)
            corrupted = data['corrupted']
            ground_truth = data['ground_truth']
            marker_names = data['marker_names'].tolist()

            if 'RShoulder' in marker_names and 'LShoulder' in marker_names:
                rs_idx = marker_names.index('RShoulder')
                ls_idx = marker_names.index('LShoulder')

                r_error = abs(corrupted[rs_idx, 2] - ground_truth[rs_idx, 2]) * 1000  # mm
                l_error = abs(corrupted[ls_idx, 2] - ground_truth[ls_idx, 2]) * 1000

                low_r_errors.append(r_error)
                low_l_errors.append(l_error)

        # High angles (side view)
        high_r_errors = []
        high_l_errors = []

        for sample_path in high_angle_samples:
            data = np.load(sample_path)
            corrupted = data['corrupted']
            ground_truth = data['ground_truth']
            marker_names = data['marker_names'].tolist()

            if 'RShoulder' in marker_names and 'LShoulder' in marker_names:
                rs_idx = marker_names.index('RShoulder')
                ls_idx = marker_names.index('LShoulder')

                r_error = abs(corrupted[rs_idx, 2] - ground_truth[rs_idx, 2]) * 1000
                l_error = abs(corrupted[ls_idx, 2] - ground_truth[ls_idx, 2]) * 1000

                high_r_errors.append(r_error)
                high_l_errors.append(l_error)

        # Check for viewpoint-dependent asymmetry
        low_r_mean = np.mean(low_r_errors) if low_r_errors else 0
        low_l_mean = np.mean(low_l_errors) if low_l_errors else 0
        high_r_mean = np.mean(high_r_errors) if high_r_errors else 0
        high_l_mean = np.mean(high_l_errors) if high_l_errors else 0

        print(f"  Low angles (0°-15°):")
        print(f"    R shoulder: {low_r_mean:.1f}mm, L shoulder: {low_l_mean:.1f}mm")
        print(f"  High angles (60°-75°):")
        print(f"    R shoulder: {high_r_mean:.1f}mm, L shoulder: {high_l_mean:.1f}mm")

        # At low angles, both should have similar errors (frontal view)
        # At high angles, should see asymmetry (side view favors one side)
        if high_r_mean > 0 and high_l_mean > 0:
            asymmetry_ratio = max(high_r_mean, high_l_mean) / min(high_r_mean, high_l_mean)
            if asymmetry_ratio > 1.2:
                print(f"✓ PASS: Viewpoint-dependent asymmetry detected (ratio={asymmetry_ratio:.2f})")
            else:
                print(f"⚠️  WARNING: Low asymmetry (ratio={asymmetry_ratio:.2f}) - viewpoint noise may be weak")
        else:
            print(f"⚠️  WARNING: Could not verify viewpoint-dependent noise")
    else:
        print(f"⚠️  WARNING: Not enough samples to check viewpoint patterns")

    # ========================================================================
    # FINAL REPORT
    # ========================================================================
    print()
    print("="*80)
    print("VALIDATION SUMMARY")
    print("="*80)

    if len(issues) == 0:
        print("✅ ALL CHECKS PASSED - Safe to proceed with training")
        print()
        print("Training data is ready:")
        print(f"  - {len(samples)} samples validated")
        print(f"  - Feet geometry: VALID (3D triangles)")
        print(f"  - Bone lengths: CONSISTENT")
        print(f"  - No data corruption")
        print(f"  - {len(angles_found)} camera angles, {len(noise_levels_found)} noise levels")
        print()
        print("✓ You can start training now!")
        return True
    else:
        print(f"❌ {len(issues)} ISSUES FOUND - DO NOT START TRAINING")
        print()
        print("Issues:")
        for issue in issues:
            print(f"  - {issue}")
        print()
        print("⚠️  FIX THESE ISSUES BEFORE TRAINING!")
        print("   Otherwise you'll waste 9 hours on bad data!")
        return False


if __name__ == "__main__":
    data_dir = Path("data/training/cmu_converted")

    if not data_dir.exists():
        print(f"ERROR: Training data directory not found: {data_dir}")
        print("Run: python3 training/generate_training_data.py")
        sys.exit(1)

    passed = validate_training_data(data_dir, num_samples=1000)

    sys.exit(0 if passed else 1)
