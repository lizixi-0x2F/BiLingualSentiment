# Removed Files in Simplified Version

This document lists files that have been removed from the project as part of the simplification process to remove advanced features.

## Removed Files

1. **src/core/adversarial.py**
   - This file contained the FGM (Fast Gradient Method) adversarial training implementation.
   - Removal reason: Adversarial training was one of the advanced features that needed to be simplified.

2. **src/core/boundary_weights.py**
   - This file contained the BoundarySampleWeighter class for weighted sampling of boundary samples.
   - Removal reason: Part of the advanced training techniques that were simplified.

## Modified Files

1. **src/core/__init__.py**
   - Removed import of the FGM class from the adversarial.py module.
   - Removed import of the BoundarySampleWeighter class from the boundary_weights.py module.
   - Updated header comment to indicate this is the simplified version.

2. **src/train.py**
   - Modified to remove references to FGM adversarial training.
   - Removed LabelSmoothingLoss usage.
   - Removed QuadrantClassificationLoss class definition and usage.

## Notes on Other Advanced Features

1. **Label Smoothing / Soft Labels**
   - The LabelSmoothingLoss was referenced in train.py but not defined there.
   - The simplified version (train_simple.py) uses a standard MSELoss instead.

2. **Multi-task Learning Heads / Quadrant Classification**
   - The QuadrantClassificationLoss class was defined in train.py but isn't used in the simplified version.
   - The model in train_simple.py is initialized with `use_quadrant_head=False`.

## Remaining Files

All other files have been kept to ensure the simplified project remains functional while removing the three advanced features:
1. FGM adversarial training
2. Soft labels / label smoothing
3. Multi-task learning heads

The simplified version can be run using the scripts in the `scripts/` directory that have the "simple" prefix.
