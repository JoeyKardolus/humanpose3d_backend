#!/usr/bin/env python3
"""
Diagnose POF (Part Orientation Field) prediction issues.

Analyzes:
1. Per-limb error breakdown (which limbs are hardest?)
2. Correlation between 2D foreshortening and error
3. Correlation between camera angle and error
4. Whether predictions are actually unit vectors
5. Distribution of GT vs predicted orientations
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from tqdm import tqdm

from src.depth_refinement.model import create_model, LIMBS
from src.depth_refinement.dataset import create_dataloaders
from src.depth_refinement.losses import limb_orientation_angle_error

# Limb names for reporting
LIMB_NAMES = [
    "L upper arm (5→7)",
    "L forearm (7→9)",
    "R upper arm (6→8)",
    "R forearm (8→10)",
    "L thigh (11→13)",
    "L shin (13→15)",
    "R thigh (12→14)",
    "R shin (14→16)",
    "Shoulder width (5↔6)",
    "Hip width (11↔12)",
    "L torso (5→11)",
    "R torso (6→12)",
]


def compute_2d_foreshortening(pose_2d, limb_idx):
    """Compute 2D limb length (foreshortening indicator)."""
    parent, child = LIMBS[limb_idx]
    vec_2d = pose_2d[:, child] - pose_2d[:, parent]  # (batch, 2)
    length_2d = torch.norm(vec_2d, dim=-1)  # (batch,)
    return length_2d


def analyze_predictions(model, val_loader, device, num_batches=50):
    """Analyze POF predictions in detail."""
    model.eval()

    # Per-limb statistics
    limb_errors = [[] for _ in range(12)]  # Angular error per limb
    limb_gt_z = [[] for _ in range(12)]    # GT Z component (depth direction)
    limb_pred_z = [[] for _ in range(12)]  # Predicted Z component
    limb_foreshorten = [[] for _ in range(12)]  # 2D length (foreshortening)

    # Global stats
    all_azimuths = []
    all_elevations = []
    all_errors = []

    # Check unit vector norms
    pred_norms = []
    gt_norms = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader, desc='Analyzing', total=num_batches)):
            if i >= num_batches:
                break

            corrupted = batch['corrupted'].to(device)
            visibility = batch['visibility'].to(device)
            pose_2d = batch['pose_2d'].to(device)
            azimuth = batch['azimuth'].to(device)
            elevation = batch['elevation'].to(device)
            gt_limb_orientations = batch['gt_limb_orientations'].to(device)

            # Get predictions
            output = model(corrupted, visibility, pose_2d=pose_2d,
                          azimuth=azimuth, elevation=elevation)

            pred_limb = output['pred_limb_orientations']  # (batch, 12, 3)

            # Check norms
            pred_norm = torch.norm(pred_limb, dim=-1)  # (batch, 12)
            gt_norm = torch.norm(gt_limb_orientations, dim=-1)
            pred_norms.append(pred_norm.cpu())
            gt_norms.append(gt_norm.cpu())

            # Per-limb angular error
            angle_errors = limb_orientation_angle_error(pred_limb, gt_limb_orientations)  # (batch, 12)

            for limb_idx in range(12):
                limb_errors[limb_idx].extend(angle_errors[:, limb_idx].cpu().numpy())
                limb_gt_z[limb_idx].extend(gt_limb_orientations[:, limb_idx, 2].cpu().numpy())
                limb_pred_z[limb_idx].extend(pred_limb[:, limb_idx, 2].cpu().numpy())

                # 2D foreshortening
                foreshorten = compute_2d_foreshortening(pose_2d, limb_idx)
                limb_foreshorten[limb_idx].extend(foreshorten.cpu().numpy())

            all_azimuths.extend(azimuth.cpu().numpy())
            all_elevations.extend(elevation.cpu().numpy())
            all_errors.extend(angle_errors.mean(dim=-1).cpu().numpy())

    return {
        'limb_errors': limb_errors,
        'limb_gt_z': limb_gt_z,
        'limb_pred_z': limb_pred_z,
        'limb_foreshorten': limb_foreshorten,
        'pred_norms': torch.cat(pred_norms, dim=0).numpy(),
        'gt_norms': torch.cat(gt_norms, dim=0).numpy(),
        'azimuths': np.array(all_azimuths),
        'elevations': np.array(all_elevations),
        'mean_errors': np.array(all_errors),
    }


def print_report(results):
    """Print detailed diagnostic report."""
    print("\n" + "=" * 70)
    print("POF (Part Orientation Fields) DIAGNOSTIC REPORT")
    print("=" * 70)

    # 1. Unit vector check
    print("\n1. UNIT VECTOR CHECK")
    print("-" * 40)
    pred_norms = results['pred_norms']
    gt_norms = results['gt_norms']
    print(f"   Predicted norms: mean={pred_norms.mean():.4f}, std={pred_norms.std():.4f}, "
          f"min={pred_norms.min():.4f}, max={pred_norms.max():.4f}")
    print(f"   GT norms:        mean={gt_norms.mean():.4f}, std={gt_norms.std():.4f}, "
          f"min={gt_norms.min():.4f}, max={gt_norms.max():.4f}")
    if abs(pred_norms.mean() - 1.0) > 0.01:
        print("   WARNING: Predicted vectors are not unit normalized!")

    # 2. Per-limb error breakdown
    print("\n2. PER-LIMB ERROR BREAKDOWN (sorted by error)")
    print("-" * 40)
    limb_mean_errors = [(i, np.mean(results['limb_errors'][i])) for i in range(12)]
    limb_mean_errors.sort(key=lambda x: x[1], reverse=True)

    for limb_idx, mean_err in limb_mean_errors:
        errors = results['limb_errors'][limb_idx]
        print(f"   {LIMB_NAMES[limb_idx]:25s}: {mean_err:5.1f}° ± {np.std(errors):4.1f}° "
              f"(median={np.median(errors):5.1f}°)")

    overall_mean = np.mean([np.mean(e) for e in results['limb_errors']])
    print(f"\n   OVERALL MEAN: {overall_mean:.1f}°")

    # 3. Z-component analysis (depth direction)
    print("\n3. Z-COMPONENT ANALYSIS (depth prediction)")
    print("-" * 40)
    print("   Limb                        GT_Z_mean  Pred_Z_mean  Correlation")
    for limb_idx in range(12):
        gt_z = np.array(results['limb_gt_z'][limb_idx])
        pred_z = np.array(results['limb_pred_z'][limb_idx])
        corr = np.corrcoef(gt_z, pred_z)[0, 1] if len(gt_z) > 1 else 0
        print(f"   {LIMB_NAMES[limb_idx]:25s}: {gt_z.mean():7.3f}    {pred_z.mean():7.3f}      {corr:+.3f}")

    # 4. Foreshortening correlation
    print("\n4. 2D FORESHORTENING vs ERROR (should be negative - short 2D = hard)")
    print("-" * 40)
    print("   Limb                        2D_len_mean  Corr(2D_len, error)")
    for limb_idx in range(12):
        foreshorten = np.array(results['limb_foreshorten'][limb_idx])
        errors = np.array(results['limb_errors'][limb_idx])
        corr = np.corrcoef(foreshorten, errors)[0, 1] if len(foreshorten) > 1 else 0
        print(f"   {LIMB_NAMES[limb_idx]:25s}: {foreshorten.mean():7.3f}        {corr:+.3f}")

    # 5. Camera angle analysis
    print("\n5. CAMERA ANGLE vs MEAN ERROR")
    print("-" * 40)
    azimuths = results['azimuths']
    elevations = results['elevations']
    errors = results['mean_errors']

    # Bin by azimuth quadrant
    quadrants = ['Front (315-45°)', 'Right (45-135°)', 'Back (135-225°)', 'Left (225-315°)']
    quadrant_errors = [[], [], [], []]
    for az, err in zip(azimuths, errors):
        if az >= 315 or az < 45:
            quadrant_errors[0].append(err)
        elif 45 <= az < 135:
            quadrant_errors[1].append(err)
        elif 135 <= az < 225:
            quadrant_errors[2].append(err)
        else:
            quadrant_errors[3].append(err)

    for name, errs in zip(quadrants, quadrant_errors):
        if errs:
            print(f"   {name:20s}: {np.mean(errs):5.1f}° (n={len(errs)})")

    # 6. Identify worst cases
    print("\n6. WORST LIMBS BY CATEGORY")
    print("-" * 40)
    arm_errors = [np.mean(results['limb_errors'][i]) for i in [0, 1, 2, 3]]
    leg_errors = [np.mean(results['limb_errors'][i]) for i in [4, 5, 6, 7]]
    torso_errors = [np.mean(results['limb_errors'][i]) for i in [8, 9, 10, 11]]
    print(f"   Arms mean:  {np.mean(arm_errors):.1f}°")
    print(f"   Legs mean:  {np.mean(leg_errors):.1f}°")
    print(f"   Torso mean: {np.mean(torso_errors):.1f}°")

    print("\n" + "=" * 70)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Diagnose POF prediction issues')
    parser.add_argument('--checkpoint', type=str, default='models/checkpoints/best_depth_model.pth',
                        help='Model checkpoint to analyze')
    parser.add_argument('--data', type=str, default='data/training/aistpp_converted',
                        help='Training data path')
    parser.add_argument('--num-batches', type=int, default=50,
                        help='Number of validation batches to analyze')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load model
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get('config', {})

    model = create_model(
        num_joints=config.get('num_joints', 17),
        d_model=config.get('d_model', 64),
        num_heads=config.get('num_heads', 4),
        num_layers=config.get('num_layers', 4),
        use_elepose=config.get('use_elepose', False),
        use_limb_orientations=config.get('use_limb_orientations', True),
    ).to(device)
    model.load_state_dict(checkpoint['model'])
    print(f"Model loaded (epoch {checkpoint.get('epoch', '?')})")

    # Load data
    print(f"\nLoading data from {args.data}")
    _, val_loader = create_dataloaders(args.data, batch_size=64, num_workers=4)

    # Analyze
    print(f"\nAnalyzing {args.num_batches} batches...")
    results = analyze_predictions(model, val_loader, device, args.num_batches)

    # Report
    print_report(results)


if __name__ == '__main__':
    main()
