import numpy as np
from scipy.ndimage import gaussian_filter

from adaptive_keyframe_selection import (
    AdaptiveKeyframeSelector,
    warp_frame,
    to_grayscale,
)


# --------------------------------------------------------------------------- #
# Synthetic scene generation
# --------------------------------------------------------------------------- #
def make_scene(H, W, seed=7):
    rng = np.random.default_rng(seed)
    # Textured intensity: blurred noise + a few sharp edges for structure.
    tex = gaussian_filter(rng.random((H, W)), sigma=2.0)
    tex += 0.4 * gaussian_filter(rng.random((H, W)), sigma=6.0)
    tex[:, W // 3] += 0.6                      # a couple of strong vertical edges
    tex[2 * H // 3, :] += 0.6
    tex = (tex - tex.min()) / (np.ptp(tex) + 1e-9)

    # Smoothly varying depth (metric); mean ~3 m, +/- ~0.8 m -> real parallax.
    depth = 3.0 + 0.8 * gaussian_filter(rng.standard_normal((H, W)), sigma=8.0)
    return tex.astype(np.float64), depth.astype(np.float64)


def yaw_pose(tx, ty, yaw):
    """camera-to-world pose from a translation and a yaw (rotation about y)."""
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, 0.0]
    return T


_NOISE_RNG = np.random.default_rng(123)


def render(tex, depth, pose, K, noise_std=0.012):
    gray, mask, wdepth = warp_frame(tex, depth, np.eye(4), pose, K,
                                    return_depth=True)
    gray[~mask] = tex[~mask]                    # continue background texture
    wdepth[~mask] = depth[~mask]                # plausible background depth
    if noise_std > 0:
        gray = np.clip(gray + _NOISE_RNG.normal(0, noise_std, gray.shape), 0, 1)
    wdepth = wdepth * (1.0 + _NOISE_RNG.normal(0, 0.02, wdepth.shape))
    return gray, wdepth


def add_moving_object(gray, depth, cx, cy, radius=15, brightness=1.0, obj_depth=1.2):
    H, W = gray.shape
    yy, xx = np.ogrid[:H, :W]
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    blob = np.exp(-d2 / (2 * radius ** 2))
    inside = d2 <= (1.6 * radius) ** 2
    gray = np.clip(gray + brightness * blob, 0, 1)
    depth = depth.copy()
    depth[inside] = obj_depth                   # object occludes background
    return gray, depth


def build_sequence(H=120, W=160):
    fx = fy = 150.0
    K = np.array([[fx, 0, W / 2.0], [0, fy, H / 2.0], [0, 0, 1.0]])
    tex, depth = make_scene(H, W)

    frames, phase_of = [], []
    p1, p2, p3, p4 = 15, 20, 20, 30          # frames per phase

    # Phase 1: stationary.
    for _ in range(p1):
        frames.append(render(tex, depth, np.eye(4), K)); phase_of.append(1)
    # Phase 2: slow lateral translation.
    for i in range(p2):
        pose = yaw_pose(0.010 * (i + 1), 0.0, 0.0)
        frames.append(render(tex, depth, pose, K)); phase_of.append(2)
    tx_end = 0.010 * p2
    # Phase 3: faster translation + small yaw.
    for i in range(p3):
        pose = yaw_pose(tx_end + 0.028 * (i + 1), 0.0, 0.006 * (i + 1))
        frames.append(render(tex, depth, pose, K)); phase_of.append(3)
    pose_hold = yaw_pose(tx_end + 0.028 * p3, 0.0, 0.006 * p3)
    # Phase 4: stationary camera, but the scene is dynamic -- an object moves
    # erratically and the illumination flickers (high-variance change).
    obj_rng = np.random.default_rng(99)
    for i in range(p4):
        g, d = render(tex, depth, pose_hold, K)
        cx = int(obj_rng.uniform(0.15, 0.85) * W)      # erratic jumps
        cy = int(obj_rng.uniform(0.25, 0.75) * H)
        g, d = add_moving_object(g, d, cx, cy)
        if i % 5 == 0:                                   # intermittent flicker
            g = np.clip(g + obj_rng.uniform(-0.12, 0.12), 0, 1)
        frames.append((g, d)); phase_of.append(4)

    poses = []  # rebuild the pose list aligned with frames for the selector
    poses += [np.eye(4)] * p1
    poses += [yaw_pose(0.010 * (i + 1), 0.0, 0.0) for i in range(p2)]
    poses += [yaw_pose(tx_end + 0.028 * (i + 1), 0.0, 0.006 * (i + 1)) for i in range(p3)]
    poses += [pose_hold] * p4

    seq = [(g, d, p) for (g, d), p in zip(frames, poses)]
    return seq, np.array(phase_of), K


# --------------------------------------------------------------------------- #
# Run + report
# --------------------------------------------------------------------------- #
def main():
    seq, phase_of, K = build_sequence()

    selector = AdaptiveKeyframeSelector(
        intrinsics=K,
        alpha=0.7, beta=0.3,       
        window_size=5,              # W
        sensitivity=1.5,            # k
        decay=0.95,                 # gamma (refractory)
        base_threshold=0.056,       # theta0  (just above static noise floor)
        init_threshold=0.110,       # theta_init (warm-up)
    )
    res = selector.run(seq)

    print("=" * 66)
    print("Adaptive Keyframe Selection — synthetic sequence")
    print("=" * 66)
    print(f"Total frames        : {res.num_frames}")
    print(f"Selected keyframes  : {res.num_keyframes}")
    print(f"KFCR (compression)  : {res.kfcr:.2f}%   (higher = fewer frames kept)")
    print(f"Keyframe indices    : {res.keyframe_indices}")
    print("-" * 66)
    print(f"{'phase':<28}{'frames':>8}{'keyframes':>12}{'kept %':>10}")
    kf_set = set(res.keyframe_indices)  # 1-based
    labels = {1: "1 stationary (redundant)",
              2: "2 slow camera motion",
              3: "3 fast camera motion+yaw",
              4: "4 dynamic object"}
    for ph in (1, 2, 3, 4):
        idxs = np.where(phase_of == ph)[0]         # 0-based
        n = len(idxs)
        nk = sum(1 for i in idxs if (i + 1) in kf_set)
        print(f"{labels[ph]:<28}{n:>8}{nk:>12}{100.0*nk/max(n,1):>9.1f}%")
    print("=" * 66)
    err = np.array(res.errors)
    print("per-phase hybrid error  (mean / max):")
    for ph in (1, 2, 3, 4):
        idxs = np.where(phase_of == ph)[0]
        e = err[idxs]
        print(f"  {labels[ph]:<28} mean={e.mean():.4f}  max={e.max():.4f}")
    print("=" * 66)

    try:
        _plot(res, phase_of)
    except Exception as e:  # plotting is optional
        print(f"(plot skipped: {e})")


def _plot(res, phase_of, path="keyframe_selection_demo.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = res.num_frames
    x = np.arange(1, n + 1)
    err = np.array(res.errors)
    thr = np.array(res.thresholds)
    kf = np.array(res.keyframe_indices)  # 1-based

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})

    # Phase shading.
    phase_colors = {1: "#e8eef7", 2: "#e7f5ea", 3: "#fdf1e3", 4: "#fbe9ec"}
    b = 0
    for ph in (1, 2, 3, 4):
        idxs = np.where(phase_of == ph)[0]
        if len(idxs):
            lo, hi = idxs.min() + 0.5, idxs.max() + 1.5
            ax1.axvspan(lo, hi, color=phase_colors[ph], zorder=0)
            ax2.axvspan(lo, hi, color=phase_colors[ph], zorder=0)

    ax1.plot(x, err, color="#1f4e79", lw=1.8, label="hybrid error $e_t$", zorder=3)
    ax1.plot(x, thr, color="#c0392b", lw=1.6, ls="--",
             label=r"dynamic threshold $\theta_t$", zorder=3)
    ax1.scatter(kf, err[kf - 1], color="#e67e22", s=55, zorder=5,
                edgecolor="k", linewidths=0.6, label="selected keyframe")
    ax1.set_ylabel("error")
    ax1.set_title("Momentum-aware adaptive keyframe selection "
                  f"(KFCR = {res.kfcr:.1f}%)")
    ax1.legend(loc="upper left", framealpha=0.9)
    ax1.grid(alpha=0.25)

    # Keyframe raster.
    ax2.vlines(kf, 0, 1, color="#e67e22", lw=2)
    ax2.set_yticks([])
    ax2.set_ylabel("keyframes")
    ax2.set_xlabel("frame index")
    ax2.set_xlim(0.5, n + 0.5)

    # Phase labels.
    names = {1: "stationary", 2: "slow motion", 3: "fast motion", 4: "dynamic object"}
    for ph in (1, 2, 3, 4):
        idxs = np.where(phase_of == ph)[0]
        if len(idxs):
            ax1.text((idxs.min() + idxs.max()) / 2 + 1, ax1.get_ylim()[1] * 0.96,
                     names[ph], ha="center", va="top", fontsize=9, color="#444")

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"Saved diagnostic plot -> {path}")


if __name__ == "__main__":
    main()
