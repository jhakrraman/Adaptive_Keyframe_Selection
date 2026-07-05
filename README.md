# Adaptive Keyframe Selection

This repository contains the code for the paper **"Adaptive Keyframe Selection for
Scalable 3D Scene Reconstruction in Dynamic Environments,"** accepted at **ROBOVIS 2026**.

A lightweight, framework-agnostic front-end that decides *online* whether each incoming
RGB-D frame should become a keyframe. It warps the most recent keyframe into the current
view via depth-based reprojection, measures a hybrid photometric + structural (SSIM)
error over the co-visible region, and thresholds it against a momentum-aware moving
statistic so that redundant frames (static scenes, pure ego-motion) are skipped while
genuinely novel or dynamic content triggers a keyframe. Pure NumPy + SciPy — no deep
learning framework required, so it drops in front of pipelines such as Spann3r / CUT3R.

## Files

| File | Description |
|------|-------------|
| `adaptive_keyframe_selection.py` | Core method. Warping, hybrid error (Algorithm 1) and the momentum-aware selector (Algorithm 2). This is the only file you need to integrate. |
| `demo_keyframe_selection.py` | Self-contained synthetic RGB-D demo (static → slow motion → fast motion → dynamic object). Runs the selector and saves a diagnostic plot. |
| `keyframe_selection_demo.png` | Example output plot produced by the demo. |

## Requirements

```
numpy
scipy      # SSIM Gaussian window only
matplotlib # demo plot only
```

```bash
pip install numpy scipy matplotlib
```

## Usage

**Online (streaming) — the intended integration:**

```python
import numpy as np
from adaptive_keyframe_selection import AdaptiveKeyframeSelector

# intrinsics: 3x3 K ; pose: 4x4 camera-to-world ; depth in metric units
selector = AdaptiveKeyframeSelector(intrinsics_K)

for rgb, depth, pose in rgbd_stream:          # rgb (H,W,3) or (H,W); depth (H,W)
    if selector.update(rgb, depth, pose):     # -> bool
        reconstruction.add_keyframe(rgb, depth, pose)

print(f"KFCR = {selector.result.kfcr:.1f}%")  # compression ratio
```

**Batch (offline) over a list of frames:**

```python
result = AdaptiveKeyframeSelector(intrinsics_K).run(frames)  # frames: [(rgb, depth, pose), ...]
print(result.keyframe_indices)   # selected frame indices (first frame always kept)
print(result.errors, result.thresholds)   # per-frame logs for analysis / plotting
```

**Run the demo and the built-in geometry self-test:**

```bash
python demo_keyframe_selection.py            # writes keyframe_selection_demo.png
python adaptive_keyframe_selection.py --test # validates the reprojection/warp math
```

## Key hyperparameters

Constructor arguments map to the paper's symbols (defaults shown):

| Argument | Symbol | Default | Meaning |
|----------|:------:|:-------:|---------|
| `alpha`, `beta` | α, β | 0.7, 0.3 | photometric / structural weights (Eq. 3) |
| `window_size` | W | 5 | sliding window for the moving statistics (Eq. 4–5) |
| `sensitivity` | k | 1.5 | std multiplier in `θ = μ + k·σ` (Eq. 6) |
| `decay` | γ | 0.95 | post-selection refractory factor (Eq. 7); set `1.0` to disable |
| `base_threshold` | θ₀ | 0.05 | floor threshold — **dataset-dependent, tune to your error scale** |
| `init_threshold` | θ_init | 0.15 | warm-up threshold before the window fills |

## Notes

- `base_threshold` (θ₀) is dataset-dependent (grid-searched per dataset in the paper) and
  should be tuned to the scale of your images/depth. Defaults assume intensities in `[0, 1]`.
- **Refractory decay (Eq. 7):** taken literally, `θ ← γ·θ` (γ<1) *lowers* the threshold and
  has no lasting effect (θ is recomputed from the window each step). This implementation
  follows the described *intent* — after a selection the threshold is briefly *raised* and
  relaxes back by γ per frame, suppressing bursty selection. See the docstring for details;
  `decay=1.0` turns it off.
