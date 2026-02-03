import os
import argparse
from dataclasses import dataclass
from typing import Tuple, List
from pathlib import Path

import cv2
import numpy as np


# ----------------------------- Configuration Params -----------------------------
@dataclass
class AcneParams:
    """
    Configuration parameters for synthetic acne generation.
    """
    seed: int = 1234
    count: int = 32           # Total number of acne lesions to attempt generating
    r_min: int = 5            # Minimum radius of a lesion
    r_max: int = 14           # Maximum radius of a lesion
    red_strength: float = 0.46 # Intensity of the redness blending
    pustule_prob: float = 0.55 # Probability of a lesion having a visible white center (pustule)
    blur_sigma: float = 0.9   # Final blur to blend edges naturally


@dataclass
class ErythemaParams:
    """
    Configuration parameters for synthetic erythema (redness) generation.
    """
    seed: int = 4321
    strength: float = 0.52    # Intensity of the red overlay
    patchiness: float = 0.75  # How irregular the redness pattern is (vs. uniform)
    smoothness: int = 41      # Kernel size for smoothing (Larger = more natural gradients)
    fine_noise: float = 0.12  # Amount of high-frequency noise for skin texture realism


# ----------------------------- Helper Functions -----------------------------
def clamp01(x: np.ndarray) -> np.ndarray:
    """Clips array values to the range [0.0, 1.0]."""
    return np.clip(x, 0.0, 1.0)


def ensure_odd(k: int) -> int:
    """Ensures kernel size is odd (required for OpenCV functions)."""
    return k if k % 2 == 1 else k + 1


def read_bgr(path: str) -> np.ndarray:
    """Reads an image from path in BGR format."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def detect_face_rect(bgr: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Detects the largest face in the image using Haar Cascades.
    Used as a lightweight alternative to MediaPipe.
    
    Returns:
        Tuple (x, y, w, h) of the bounding box.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    face_cascade = cv2.CascadeClassifier(cascade_path)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5,
        flags=cv2.CASCADE_SCALE_IMAGE, minSize=(80, 80)
    )
    
    # If face detected, return the largest one
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        return int(x), int(y), int(w), int(h)

    # Fallback: If no face detected, assume center crop
    H, W = bgr.shape[:2]
    w = int(W * 0.55)
    h = int(H * 0.65)
    x = (W - w) // 2
    y = (H - h) // 2
    return x, y, w, h


def skin_mask_ycrcb(bgr: np.ndarray) -> np.ndarray:
    """
    Generates a binary skin mask using thresholds in the YCrCb color space.
    This helps ensure we only generate lesions on skin, not hair or background.
    """
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)

    # Classic skin detection thresholds
    mask = (
        (Cr >= 133) & (Cr <= 173) &
        (Cb >= 77) & (Cb <= 127) &
        (Y >= 30)
    ).astype(np.uint8) * 255

    # Morphological operations to clean noise
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.GaussianBlur(mask, (9, 9), 0)
    return mask


def rect_to_mask(shape_hw: Tuple[int, int], rect: Tuple[int, int, int, int]) -> np.ndarray:
    """Creates a binary mask for a specific rectangular region."""
    H, W = shape_hw
    x1, y1, x2, y2 = rect
    x1 = max(0, min(W, x1)); x2 = max(0, min(W, x2))
    y1 = max(0, min(H, y1)); y2 = max(0, min(H, y2))
    m = np.zeros((H, W), dtype=np.uint8)
    m[y1:y2, x1:x2] = 255
    return m


def face_ellipse_mask(shape_hw: Tuple[int, int], face_rect: Tuple[int, int, int, int], shrink: float) -> np.ndarray:
    """Creates an elliptical mask fitted to the face bounding box."""
    H, W = shape_hw
    x, y, w, h = face_rect
    cx, cy = x + w // 2, y + h // 2
    ax = int((w / 2) * (1 - shrink))
    ay = int((h / 2) * (1 - shrink))

    m = np.zeros((H, W), dtype=np.uint8)
    cv2.ellipse(m, (cx, cy), (max(1, ax), max(1, ay)), 0, 0, 360, 255, -1, lineType=cv2.LINE_AA)
    m = cv2.GaussianBlur(m, (0, 0), 2.2)
    return m


def erode_mask(mask255: np.ndarray, ksize: int = 9, iters: int = 1) -> np.ndarray:
    """Shrinks the mask slightly to avoid edge artifacts."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.erode(mask255, k, iterations=iters)


def mean_L_of_mask(bgr: np.ndarray, mask255: np.ndarray) -> float:
    """Calculates the average lightness (L channel) of the skin area."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)
    m = (mask255 > 40)
    if m.sum() < 50:
        return float(L.mean())
    return float(L[m].mean())


def sample_points_in_mask(rng: np.random.Generator, mask255: np.ndarray, rect: Tuple[int, int, int, int], n: int) -> List[Tuple[int, int]]:
