# Pipeline Overview

## Capture & Detection

## Conversion

## Augmentation & Export

## Strict Mode

Seed set ORDER_22 (Pose2Sim-aligned order): Neck, RShoulder, LShoulder, RHip, LHip, RKnee, LKnee, RAnkle, LAnkle, RHeel, LHeel, RSmallToe, LSmallToe, RBigToe, LBigToe, RElbow, LElbow, RWrist, LWrist, Hip, Head, Nose.

Mapping MediaPipe → marker names (subset):
- LEFT/RIGHT_SHOULDER → L/RShoulder
- LEFT/RIGHT_ELBOW → L/RElbow
- LEFT/RIGHT_WRIST → L/RWrist
- LEFT/RIGHT_HIP → L/RHip
- LEFT/RIGHT_KNEE → L/RKnee
- LEFT/RIGHT_ANKLE → L/RAnkle
- LEFT/RIGHT_HEEL → L/RHeel
- LEFT/RIGHT_FOOT_INDEX → L/RBigToe
- NOSE → Nose (HEAD may be missing)

Strict rules:
- No placeholders or duplicated markers; missing landmarks remain blank.
- Derivatives require both parents: Hip = mean(LHip, RHip); Neck = mean(LShoulder, RShoulder).
- Apply a visibility threshold (default 0.5) when exporting CSV rows.
- Header-fix adjusts only metadata (NumMarkers, labels) to match collected data without inventing names.
- Each run writes artifacts into `data/output/pose-3d/<basename>/`.
