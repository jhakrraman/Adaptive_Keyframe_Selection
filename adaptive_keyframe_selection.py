"""
The method is a lightweight, content-aware perception front-end that decides,
online and frame-by-frame, whether an incoming RGB-D frame should be kept as a
keyframe for a downstream reconstruction network (e.g. Spann3r / CUT3R).

It has two complementary parts:

  1. Hybrid error metric (paper Sec. 3.2 / Algorithm 1)
     The most recent keyframe is depth-warped into the current camera view.
     The perceptual change between the current frame and the warped keyframe is
     measured with a weighted combination of
         - photometric L1 error   e_photo  (Eq. 1)   -- fine texture / detail
         - structural error       e_ssim   (Eq. 2)   -- illumination-robust
     giving the hybrid score       e_t = alpha*e_photo + beta*e_ssim   (Eq. 3).

  2. Momentum-aware dynamic threshold (paper Sec. 3.3 / Algorithm 2)
     A frame is selected iff e_t exceeds a threshold theta_t that adapts to the
     recent statistics of the error signal (the scene's "momentum"):
         mu_t, sigma_t  over a sliding window of size W          (Eq. 4, 5)
         theta_t = max(theta0, mu_t + k*sigma_t)                 (Eq. 6)
     After a selection, a refractory period is applied so that the very next
     frames need a larger error to also fire, preventing bursty selection during
     sustained high motion (Eq. 7). See the note on `decay` below.

Dependencies: numpy, scipy (only scipy.ndimage.gaussian_filter, for SSIM).

--------------------------------------------------------------------------------
Conventions
--------------------------------------------------------------------------------
* RGB images may be passed as (H, W, 3) or as (H, W) grayscale, in either
  [0, 1] float or [0, 255] / uint8 range (auto-normalized to [0, 1]).
* Depth maps are (H, W) metric depth in the SAME camera as the RGB frame.
  Non-positive / non-finite depths are treated as invalid.
* Poses are 4x4 camera-to-world transforms (world_T_cam).
* Intrinsics K is a 3x3 pinhole matrix shared by all frames.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter


# =============================================================================
# Image utilities
# =============================================================================
def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert an RGB (H,W,3) or grayscale (H,W) image to float32 grayscale in
    [0, 1]. Inputs in [0, 255] or uint8 are auto-normalized."""
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[-1] == 3:
        gray = (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2])
    elif img.ndim == 2:
        gray = img
    else:
        raise ValueError(f"Expected (H,W,3) or (H,W) image, got shape {img.shape}")
    gray = gray.astype(np.float64)
    # Auto-normalize obviously-8-bit data.
    if img.dtype == np.uint8 or gray.max() > 1.5:
        gray = gray / 255.0
    return gray


# =============================================================================
# Depth-based frame warping  
# =============================================================================
def warp_frame(
    gray_k: np.ndarray,
    depth_k: np.ndarray,
    pose_k: np.ndarray,
    pose_t: np.ndarray,
    K: np.ndarray,
    min_depth: float = 1e-3,
    max_depth: float = 1e4,
    return_depth: bool = False,
):
   
    H, W = gray_k.shape
    gray_k = np.asarray(gray_k, dtype=np.float64)
    depth_k = np.asarray(depth_k, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)

    # Homogeneous pixel grid (u = column, v = row).
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    us = us.reshape(-1).astype(np.float64)
    vs = vs.reshape(-1).astype(np.float64)
    z = depth_k.reshape(-1)
    g = gray_k.reshape(-1)

    valid_src = np.isfinite(z) & (z > min_depth) & (z < max_depth)

    # Back-project to keyframe camera coords: X = z * K^-1 [u, v, 1]^T.
    K_inv = np.linalg.inv(K)
    pix = np.stack([us, vs, np.ones_like(us)], axis=0)          # (3, N)
    rays = K_inv @ pix                                          # (3, N)
    pts_cam_k = rays * z[None, :]                               # (3, N)
    pts_cam_k_h = np.vstack([pts_cam_k, np.ones((1, pts_cam_k.shape[1]))])

    # Keyframe cam -> world -> current cam.
    T_t_from_k = np.linalg.inv(pose_t) @ pose_k                 # (4, 4)
    pts_cam_t_h = T_t_from_k @ pts_cam_k_h                      # (4, N)
    pts_cam_t = pts_cam_t_h[:3]
    z_t = pts_cam_t[2]

    # Project into the current image plane.
    proj = K @ pts_cam_t                                        # (3, N)
    eps = 1e-12
    u_t = proj[0] / (proj[2] + eps)
    v_t = proj[1] / (proj[2] + eps)

    ui = np.round(u_t).astype(np.int64)
    vi = np.round(v_t).astype(np.int64)
    in_front = z_t > min_depth
    in_bounds = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    keep = valid_src & in_front & in_bounds

    warped = np.zeros(H * W, dtype=np.float64)
    wdepth = np.full(H * W, np.nan, dtype=np.float64)
    mask = np.zeros(H * W, dtype=bool)

    if np.any(keep):
        ui_k, vi_k = ui[keep], vi[keep]
        zt_k, g_k = z_t[keep], g[keep]
        flat = vi_k * W + ui_k

        # z-buffer: for each target pixel keep the nearest (min z_t) source.
        # lexsort's last key is primary -> sort by (flat asc, z_t asc); the
        # first occurrence of each unique flat index then holds the min depth.
        order = np.lexsort((zt_k, flat))
        flat_s = flat[order]
        g_s = g_k[order]
        z_s = zt_k[order]
        uniq, first_idx = np.unique(flat_s, return_index=True)

        warped[uniq] = g_s[first_idx]
        wdepth[uniq] = z_s[first_idx]
        mask[uniq] = True

    warped = warped.reshape(H, W)
    mask = mask.reshape(H, W)
    wdepth = wdepth.reshape(H, W)

    if return_depth:
        return warped, mask, wdepth
    return warped, mask


# =============================================================================
# Error metrics  
# =============================================================================
def photometric_error(cur: np.ndarray, warped: np.ndarray,
                      mask: Optional[np.ndarray]) -> float:
    
    diff = np.abs(cur - warped)
    if mask is not None and mask.any():
        return float(diff[mask].mean())
    return float(diff.mean())


def _ssim_map(x: np.ndarray, y: np.ndarray, sigma: float = 1.5,
             C1: float = 0.01 ** 2, C2: float = 0.03 ** 2) -> np.ndarray:
    
    # gaussian_filter is a normalized, separable Gaussian == local weighted mean.
    def gf(a):
        return gaussian_filter(a, sigma=sigma, truncate=3.5, mode="reflect")

    mu_x, mu_y = gf(x), gf(y)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sigma_x2 = gf(x * x) - mu_x2
    sigma_y2 = gf(y * y) - mu_y2
    sigma_xy = gf(x * y) - mu_xy

    num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    return num / den


def ssim_error(cur: np.ndarray, warped: np.ndarray,
              mask: Optional[np.ndarray]) -> float:

    w = warped
    if mask is not None:
        w = warped.copy()
        w[~mask] = cur[~mask]
    smap = _ssim_map(cur, w)
    if mask is not None and mask.any():
        mean_ssim = float(smap[mask].mean())
    else:
        mean_ssim = float(smap.mean())
    return 1.0 - mean_ssim


def compute_hybrid_error(
    cur_gray: np.ndarray,
    kf_gray: np.ndarray,
    kf_depth: np.ndarray,
    kf_pose: np.ndarray,
    cur_pose: np.ndarray,
    K: np.ndarray,
    alpha: float = 0.7,
    beta: float = 0.3,
) -> Tuple[float, float, float, np.ndarray]:
   
    warped, mask = warp_frame(kf_gray, kf_depth, kf_pose, cur_pose, K)
    e_photo = photometric_error(cur_gray, warped, mask)
    e_ssim = ssim_error(cur_gray, warped, mask)
    e_t = alpha * e_photo + beta * e_ssim
    return e_t, e_photo, e_ssim, mask


# =============================================================================
# Selector
# =============================================================================
@dataclass
class _KF:
    """Internal minimal keyframe record."""
    gray: np.ndarray
    depth: np.ndarray
    pose: np.ndarray
    index: int


@dataclass
class SelectionResult:
    """Per-frame diagnostics returned by the batch API."""
    keyframe_indices: List[int] = field(default_factory=list)
    is_keyframe: List[bool] = field(default_factory=list)
    errors: List[float] = field(default_factory=list)
    photo_errors: List[float] = field(default_factory=list)
    ssim_errors: List[float] = field(default_factory=list)
    thresholds: List[float] = field(default_factory=list)

    @property
    def num_frames(self) -> int:
        return len(self.is_keyframe)

    @property
    def num_keyframes(self) -> int:
        return len(self.keyframe_indices)

    @property
    def kfcr(self) -> float:
        """Keyframe Compression Ratio (%): percentage of frames discarded.
        Higher = more compression / efficiency."""
        if self.num_frames == 0:
            return 0.0
        return 100.0 * (1.0 - self.num_keyframes / self.num_frames)


class AdaptiveKeyframeSelector:

    def __init__(
        self,
        intrinsics: np.ndarray,
        alpha: float = 0.7,
        beta: float = 0.3,
        window_size: int = 5,
        sensitivity: float = 1.5,
        decay: float = 0.95,
        base_threshold: float = 0.05,
        init_threshold: float = 0.15,
        min_depth: float = 1e-3,
        max_depth: float = 1e4,
    ):
        self.K = np.asarray(intrinsics, dtype=np.float64)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.W = int(window_size)
        self.k = float(sensitivity)
        self.gamma = float(decay)
        self.theta0 = float(base_threshold)
        self.theta_init = float(init_threshold)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        if not (0.0 < self.gamma <= 1.0):
            raise ValueError("decay (gamma) must be in (0, 1]")
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        """Clear all state (call before processing a new sequence)."""
        self._t = 0                       # 1-based frame counter
        self._last_kf: Optional[_KF] = None
        self._window: deque = deque(maxlen=self.W)   # recent errors for stats
        self._refractory = 0.0            # decaying post-selection threshold
        self._error_log: List[float] = []
        self.result = SelectionResult()

    @property
    def last_keyframe_index(self) -> Optional[int]:
        return None if self._last_kf is None else self._last_kf.index

    def _adaptive_threshold(self) -> float:
        if len(self._window) >= self.W:
            arr = np.fromiter(self._window, dtype=np.float64)
            mu = float(arr.mean())
            sigma = float(arr.std())          # population std (1/W), matches Eq.5
            return max(self.theta0, mu + self.k * sigma)
        frac = min(self._t / self.W, 1.0)
        return self.theta0 * frac + self.theta_init * (1.0 - frac)

    # online step
    def update(self, rgb: np.ndarray, depth: np.ndarray,
              pose: np.ndarray) -> bool:
        
        self._t += 1
        gray = to_grayscale(rgb)
        depth = np.asarray(depth, dtype=np.float64)
        pose = np.asarray(pose, dtype=np.float64)

        # The first frame is always a keyframe.
        if self._last_kf is None:
            self._last_kf = _KF(gray, depth, pose, self._t)
            self.result.keyframe_indices.append(self._t)
            self.result.is_keyframe.append(True)
            self.result.errors.append(0.0)
            self.result.photo_errors.append(0.0)
            self.result.ssim_errors.append(0.0)
            self.result.thresholds.append(self._adaptive_threshold())
            return True

        # (1) Hybrid error vs. the most recent keyframe.
        e_t, e_photo, e_ssim, _ = compute_hybrid_error(
            gray, self._last_kf.gray, self._last_kf.depth, self._last_kf.pose,
            pose, self.K, self.alpha, self.beta,
        )
        # Append current error to the window BEFORE computing stats
        self._window.append(e_t)
        self._error_log.append(e_t)

        # (2) Dynamic threshold with refractory period.
        theta_adaptive = self._adaptive_threshold()
        theta_eff = max(theta_adaptive, self._refractory)

        # (3) Selection decision.
        selected = e_t > theta_eff

        # Relax the refractory level toward the adaptive threshold each frame.
        self._refractory *= self.gamma

        if selected:
            self._last_kf = _KF(gray, depth, pose, self._t)
            self.result.keyframe_indices.append(self._t)
            # Raise the bar transiently so the next frame(s) need a larger
            # error to also fire (refractory period; gamma < 1 => raise).
            self._refractory = theta_eff / self.gamma

        self.result.is_keyframe.append(selected)
        self.result.errors.append(e_t)
        self.result.photo_errors.append(e_photo)
        self.result.ssim_errors.append(e_ssim)
        self.result.thresholds.append(theta_eff)
        return selected

    # batch API
    def run(self, frames: Iterable[Tuple[np.ndarray, np.ndarray, np.ndarray]]
            ) -> SelectionResult:
        """Process an iterable of (rgb, depth, pose) tuples and return the full
        `SelectionResult` (does not reset automatically; call reset() first if
        reusing the selector)."""
        for rgb, depth, pose in frames:
            self.update(rgb, depth, pose)
        return self.result


# =============================================================================
# Self-tests for the geometric warp (run: python adaptive_keyframe_selection.py --test)
# =============================================================================
def _test_warp() -> None:
    rng = np.random.default_rng(0)
    H, W = 60, 80
    fx = fy = 120.0
    K = np.array([[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1]], dtype=np.float64)
    gray = rng.random((H, W))
    depth = np.full((H, W), 2.0)
    I4 = np.eye(4)

    # (a) Identity pose reproduces the input exactly on valid pixels.
    warped, mask = warp_frame(gray, depth, I4, I4, K)
    assert mask.all(), "identity warp should keep all pixels"
    assert np.allclose(warped[mask], gray[mask]), "identity warp must be exact"

    # (b) Known lateral camera translation shifts content by a predictable
    #     number of pixels: du = fx * tx / Z.
    tx, Z = 0.10, 2.0
    pose_t = np.eye(4)
    pose_t[0, 3] = tx  # move current camera +x in world
    warped2, mask2 = warp_frame(gray, depth, I4, pose_t, K)
    expected_shift = -fx * tx / Z  # image content moves opposite to camera
    # Compare an interior column band, accounting for the integer shift.
    s = int(round(expected_shift))
    x0, x1 = 25, 55
    ref = gray[20:40, x0 + (-s):x1 + (-s)]
    got = warped2[20:40, x0:x1]
    valid = mask2[20:40, x0:x1]
    err = np.abs(ref - got)[valid].mean()
    assert err < 1e-6, f"translated warp mismatch, mean err={err:.2e}, shift={s}"
    print("[warp self-test] identity + known-translation checks passed "
          f"(predicted shift = {expected_shift:.2f}px).")


if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        _test_warp()
    else:
        print("Run the demo with:  python demo_keyframe_selection.py")
        print("Run warp self-tests with:  python adaptive_keyframe_selection.py --test")
