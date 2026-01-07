import os
import argparse
from dataclasses import dataclass
from typing import Tuple, List
from pathlib import Path

import cv2
import numpy as np


# ----------------------------- Params -----------------------------
@dataclass
class AcneParams:
    seed: int = 1234
    count: int = 32
    r_min: int = 5
    r_max: int = 14
    red_strength: float = 0.46
    pustule_prob: float = 0.55      # כמה פוסטולות (עם מרכז בהיר)
    blur_sigma: float = 0.9


@dataclass
class ErythemaParams:
    seed: int = 4321
    strength: float = 0.52
    patchiness: float = 0.75
    smoothness: int = 41            # גדול = יותר גרדיאנט טבעי
    fine_noise: float = 0.12


# ----------------------------- Helpers -----------------------------
def clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def ensure_odd(k: int) -> int:
    return k if k % 2 == 1 else k + 1


def read_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def detect_face_rect(bgr: np.ndarray) -> Tuple[int, int, int, int]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    face_cascade = cv2.CascadeClassifier(cascade_path)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5,
        flags=cv2.CASCADE_SCALE_IMAGE, minSize=(80, 80)
    )
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        return int(x), int(y), int(w), int(h)

    # fallback
    H, W = bgr.shape[:2]
    w = int(W * 0.55)
    h = int(H * 0.65)
    x = (W - w) // 2
    y = (H - h) // 2
    return x, y, w, h


def skin_mask_ycrcb(bgr: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)

    mask = (
        (Cr >= 133) & (Cr <= 173) &
        (Cb >= 77) & (Cb <= 127) &
        (Y >= 30)
    ).astype(np.uint8) * 255

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.GaussianBlur(mask, (9, 9), 0)
    return mask


def rect_to_mask(shape_hw: Tuple[int, int], rect: Tuple[int, int, int, int]) -> np.ndarray:
    H, W = shape_hw
    x1, y1, x2, y2 = rect
    x1 = max(0, min(W, x1)); x2 = max(0, min(W, x2))
    y1 = max(0, min(H, y1)); y2 = max(0, min(H, y2))
    m = np.zeros((H, W), dtype=np.uint8)
    m[y1:y2, x1:x2] = 255
    return m


def face_ellipse_mask(shape_hw: Tuple[int, int], face_rect: Tuple[int, int, int, int], shrink: float) -> np.ndarray:
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
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.erode(mask255, k, iterations=iters)


def mean_L_of_mask(bgr: np.ndarray, mask255: np.ndarray) -> float:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)
    m = (mask255 > 40)
    if m.sum() < 50:
        return float(L.mean())
    return float(L[m].mean())


def sample_points_in_mask(rng: np.random.Generator, mask255: np.ndarray, rect: Tuple[int, int, int, int], n: int) -> List[Tuple[int, int]]:
    H, W = mask255.shape[:2]
    x1, y1, x2, y2 = rect
    x1 = max(0, min(W - 1, x1)); x2 = max(1, min(W, x2))
    y1 = max(0, min(H - 1, y1)); y2 = max(1, min(H, y2))

    pts = []
    tries = 0
    max_tries = n * 80 + 600
    while len(pts) < n and tries < max_tries:
        tries += 1
        x = int(rng.integers(x1, x2))
        y = int(rng.integers(y1, y2))
        if mask255[y, x] > 60:
            pts.append((x, y))
    return pts


# ----------------------------- Color Ops (LAB) -----------------------------
def blend_red_in_lab(bgr: np.ndarray, alpha01: np.ndarray, amount: float) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = cv2.split(lab)
    A2 = np.clip(A + (48.0 * amount * alpha01), 0, 255)
    out = cv2.merge([L, A2, B]).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_LAB2BGR)


def darken_L_in_lab(bgr: np.ndarray, alpha01: np.ndarray, amount: float) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = cv2.split(lab)
    L2 = np.clip(L - (16.0 * amount * alpha01), 0, 255)
    out = cv2.merge([L2, A, B]).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_LAB2BGR)


def add_pustule_center_lab(bgr: np.ndarray, alpha01: np.ndarray, amount: float) -> np.ndarray:
    a = clamp01(alpha01).astype(np.float32)

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = cv2.split(lab)

    L2 = np.clip(L + (42.0 * amount * a), 0, 255)
    A2 = np.clip(A - (18.0 * amount * a), 0, 255)
    B2 = np.clip(B + (22.0 * amount * a), 0, 255)

    tmp = cv2.merge([L2, A2, B2]).astype(np.uint8)
    out = cv2.cvtColor(tmp, cv2.COLOR_LAB2BGR).astype(np.float32)

    cream = np.array([235, 240, 248], dtype=np.float32)
    a2 = clamp01(a * 0.55)
    out = out * (1.0 - a2[..., None]) + cream[None, None, :] * a2[..., None]

    a3 = clamp01((a ** 1.8) * 0.35)
    out = np.clip(out + (255.0 - out) * a3[..., None], 0, 255)

    return out.astype(np.uint8)


# ----------------------------- Masks: Acne (hard) vs Erythema (soft) -----------------------------
def make_acne_masks(shape_hw: Tuple[int, int], face_rect: Tuple[int, int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
    H, W = shape_hw
    x, y, w, h = face_rect

    face_m = face_ellipse_mask((H, W), face_rect, shrink=0.18)
    face_m = erode_mask(face_m, ksize=9, iters=1)

    block = np.zeros((H, W), dtype=np.uint8)

    eye_band = (x + int(0.06*w), y + int(0.20*h), x + int(0.94*w), y + int(0.52*h))
    mouth    = (x + int(0.18*w), y + int(0.64*h), x + int(0.82*w), y + int(0.96*h))
    nostrils = (x + int(0.38*w), y + int(0.54*h), x + int(0.62*w), y + int(0.74*h))
    left_ear  = (x + int(0.00*w), y + int(0.25*h), x + int(0.16*w), y + int(0.82*h))
    right_ear = (x + int(0.84*w), y + int(0.25*h), x + int(1.00*w), y + int(0.82*h))

    for r in (eye_band, mouth, nostrils, left_ear, right_ear):
        block = cv2.bitwise_or(block, rect_to_mask((H, W), r))

    block = cv2.GaussianBlur(block, (0, 0), 2.4)
    block = cv2.dilate(block, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)), iterations=1)

    return face_m, block


def make_erythema_masks(shape_hw: Tuple[int, int], face_rect: Tuple[int, int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
    H, W = shape_hw
    x, y, w, h = face_rect

    face_m = face_ellipse_mask((H, W), face_rect, shrink=0.12)

    block = np.zeros((H, W), dtype=np.uint8)

    eye_band = (x + int(0.08*w), y + int(0.26*h), x + int(0.92*w), y + int(0.48*h))
    lips = (x + int(0.26*w), y + int(0.76*h), x + int(0.74*w), y + int(0.92*h))

    for r in (eye_band, lips):
        block = cv2.bitwise_or(block, rect_to_mask((H, W), r))

    block = cv2.GaussianBlur(block, (0, 0), 10.0)
    return face_m, block


# ----------------------------- Acne: Irregular lesions + texture -----------------------------
def add_irregular_lesion(rng, halo, core, pustule, x, y, r, do_pustule: bool):
    angle1 = float(rng.uniform(0, 180))
    angle2 = angle1 + float(rng.uniform(-35, 35))

    ax_h = max(2, int(r * rng.uniform(1.4, 2.3)))
    ay_h = max(2, int(r * rng.uniform(1.1, 1.9)))
    cv2.ellipse(halo, (x, y), (ax_h, ay_h), angle1, 0, 360,
                float(rng.uniform(0.18, 0.62)), -1, lineType=cv2.LINE_AA)

    dx = int(rng.integers(-max(1, r//3), max(2, r//3 + 1)))
    dy = int(rng.integers(-max(1, r//3), max(2, r//3 + 1)))

    ax_c = max(1, int(r * rng.uniform(0.55, 1.00)))
    ay_c = max(1, int(r * rng.uniform(0.50, 0.95)))
    cv2.ellipse(core, (x + dx, y + dy), (ax_c, ay_c), angle2, 0, 360,
                float(rng.uniform(0.55, 1.0)), -1, lineType=cv2.LINE_AA)

    if rng.random() < 0.55:
        dx2 = int(rng.integers(-max(1, r//2), max(2, r//2 + 1)))
        dy2 = int(rng.integers(-max(1, r//2), max(2, r//2 + 1)))
        ax2 = max(1, int(r * rng.uniform(0.35, 0.75)))
        ay2 = max(1, int(r * rng.uniform(0.35, 0.75)))
        cv2.ellipse(core, (x + dx2, y + dy2), (ax2, ay2), angle2 + 25, 0, 360,
                    float(rng.uniform(0.25, 0.60)), -1, lineType=cv2.LINE_AA)

    if do_pustule:
        pr = max(1, int(r * rng.uniform(0.18, 0.32)))
        cv2.circle(pustule, (x + dx//2, y + dy//2), pr,
                   float(rng.uniform(0.20, 0.60)), -1, lineType=cv2.LINE_AA)


def simulate_acne(bgr: np.ndarray, face_rect: Tuple[int, int, int, int], skin_mask: np.ndarray, p: AcneParams) -> np.ndarray:
    rng = np.random.default_rng(p.seed)
    H, W = bgr.shape[:2]

    face_m, block = make_acne_masks((H, W), face_rect)

    x, y, w, h = face_rect

    rects = [
        (x + int(0.08*w), y + int(0.38*h), x + int(0.46*w), y + int(0.72*h)),
        (x + int(0.54*w), y + int(0.38*h), x + int(0.92*w), y + int(0.72*h)),
        (x + int(0.20*w), y + int(0.10*h), x + int(0.80*w), y + int(0.34*h)),
        (x + int(0.28*w), y + int(0.58*h), x + int(0.72*w), y + int(0.76*h)),
    ]
    zone_mask = np.zeros((H, W), dtype=np.uint8)
    for r in rects:
        zone_mask = cv2.bitwise_or(zone_mask, rect_to_mask((H, W), r))

    allowed = cv2.bitwise_and(skin_mask, face_m)
    allowed = cv2.bitwise_and(allowed, zone_mask)
    allowed = cv2.bitwise_and(allowed, cv2.bitwise_not(block))
    allowed = erode_mask(allowed, ksize=9, iters=1)

    Lmean = mean_L_of_mask(bgr, allowed)
    red_scale = float(np.interp(Lmean, [45, 210], [0.92, 1.16]))
    depth_scale = float(np.interp(Lmean, [45, 210], [1.10, 0.85]))
    pustule_scale = float(np.interp(Lmean, [45, 210], [0.28, 0.60]))

    H, W = bgr.shape[:2]
    halo = np.zeros((H, W), dtype=np.float32)
    core = np.zeros((H, W), dtype=np.float32)
    pustule = np.zeros((H, W), dtype=np.float32)
    pustule = cv2.dilate(pustule, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    pustule = cv2.GaussianBlur(pustule, (0, 0), 0.8)

    pts = sample_points_in_mask(rng, allowed, (0, 0, W, H), p.count)

    for (x0, y0) in pts:
        r = int(rng.integers(p.r_min, p.r_max + 1))
        if rng.random() < 0.25:
            r = int(r * rng.uniform(1.15, 1.55))

        do_pustule = (rng.random() < p.pustule_prob)
        add_irregular_lesion(rng, halo, core, pustule, x0, y0, r, do_pustule)

    halo = cv2.GaussianBlur(halo, (0, 0), 2.2)
    core = cv2.GaussianBlur(core, (0, 0), 1.2)
    pustule = cv2.GaussianBlur(pustule, (0, 0), 0.9)

    noise = rng.normal(0.0, 1.0, size=(H, W)).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), 2.0)
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-6)
    texture = (noise - 0.5) * 0.18
    core = clamp01(core + texture * (core > 0).astype(np.float32))
    halo = clamp01(halo + texture * 0.6 * (halo > 0).astype(np.float32))

    allowed01 = allowed.astype(np.float32) / 255.0
    halo = clamp01(halo * allowed01)
    core = clamp01(core * allowed01)
    pustule = clamp01(pustule * allowed01)

    out = bgr.copy()
    out = blend_red_in_lab(out, halo, amount=p.red_strength * 0.65 * red_scale)
    out = blend_red_in_lab(out, core, amount=p.red_strength * 0.95 * red_scale)
    out = darken_L_in_lab(out, core, amount=0.70 * depth_scale)
    out = add_pustule_center_lab(out, pustule, amount=1.0 * pustule_scale)

    if p.blur_sigma > 0:
        blur = cv2.GaussianBlur(out, (0, 0), p.blur_sigma)
        m = clamp01(halo * 0.45 + core * 0.45 + pustule * 0.20)[..., None]
        out = (out.astype(np.float32) * (1 - m) + blur.astype(np.float32) * m).astype(np.uint8)

    return out


# ----------------------------- Erythema -----------------------------
def simulate_erythema(bgr: np.ndarray, face_rect: Tuple[int, int, int, int], skin_mask: np.ndarray, p: ErythemaParams) -> np.ndarray:
    rng = np.random.default_rng(p.seed)
    H, W = bgr.shape[:2]

    face_m, block_soft = make_erythema_masks((H, W), face_rect)

    skin01 = (skin_mask.astype(np.float32) / 255.0)
    face01 = (face_m.astype(np.float32) / 255.0)
    block01 = clamp01(block_soft.astype(np.float32) / 255.0)

    allowed_soft = clamp01(skin01 * face01 * (1.0 - block01))

    x, y, w, h = face_rect

    field = np.zeros((H, W), dtype=np.float32)
    centers = [
        (x + int(0.30*w), y + int(0.58*h)),
        (x + int(0.70*w), y + int(0.58*h)),
        (x + int(0.50*w), y + int(0.58*h)),
        (x + int(0.50*w), y + int(0.22*h)),
        (x + int(0.50*w), y + int(0.78*h)),
    ]

    for (cx, cy) in centers:
        cx = int(np.clip(rng.normal(cx, 14), 0, W - 1))
        cy = int(np.clip(rng.normal(cy, 12), 0, H - 1))

        sigma = float(rng.uniform(34, 70))
        amp = float(rng.uniform(0.75, 1.20))

        tmp = np.zeros((H, W), dtype=np.float32)
        cv2.circle(tmp, (cx, cy), int(sigma), amp, -1, lineType=cv2.LINE_AA)
        tmp = cv2.GaussianBlur(tmp, (0, 0), sigmaX=sigma * 0.70, sigmaY=sigma * 0.70)
        field += tmp

    field = field / (field.max() + 1e-6)

    n1 = rng.normal(0.0, 1.0, size=(H, W)).astype(np.float32)
    n1 = cv2.GaussianBlur(n1, (ensure_odd(p.smoothness), ensure_odd(p.smoothness)), 0)
    n1 = (n1 - n1.min()) / (n1.max() - n1.min() + 1e-6)

    n2 = rng.normal(0.0, 1.0, size=(H, W)).astype(np.float32)
    n2 = cv2.GaussianBlur(n2, (ensure_odd(p.smoothness // 2), ensure_odd(p.smoothness // 2)), 0)
    n2 = (n2 - n2.min()) / (n2.max() - n2.min() + 1e-6)

    noise_field = clamp01(0.60 * n1 + 0.40 * n2)

    alpha = field * ((1.0 - p.patchiness) * 0.78 + p.patchiness * noise_field)

    fine = rng.normal(0.0, 1.0, size=(H, W)).astype(np.float32)
    fine = cv2.GaussianBlur(fine, (0, 0), 2.0)
    fine = (fine - fine.min()) / (fine.max() - fine.min() + 1e-6)
    alpha = clamp01(alpha + p.fine_noise * (fine - 0.5))

    alpha = clamp01(alpha * allowed_soft)

    out = blend_red_in_lab(bgr, alpha, amount=p.strength)

    feather = cv2.GaussianBlur(face01, (0, 0), 3.0)[..., None]
    out = (out.astype(np.float32) * feather + bgr.astype(np.float32) * (1 - feather)).astype(np.uint8)

    return out


# ----------------------------- Batch runner -----------------------------
def run_simulation(input_path: str,
                   out_dir: str,
                   acne_count: int = 32,
                   acne_strength: float = 0.46,
                   erythema_strength: float = 0.52):
    """
    פונקציה שניתן לייבא מקבצים אחרים (למשל מהאימון),
    או להריץ כ-CLI כמו קודם.
    """
    in_path = Path(input_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    acne_dir = out_dir / "acne"
    ery_dir = out_dir / "erythema"
    acne_dir.mkdir(parents=True, exist_ok=True)
    ery_dir.mkdir(parents=True, exist_ok=True)

    if in_path.is_dir():
        images = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
            images.extend(in_path.glob(ext))
        images = sorted(images)
        if not images:
            raise FileNotFoundError(f"No images found in folder: {in_path}")
    else:
        if not in_path.exists():
            raise FileNotFoundError(f"Input does not exist: {in_path}")
        images = [in_path]

    for img_path in images:
        bgr = read_bgr(str(img_path))
        face_rect = detect_face_rect(bgr)
        skin = skin_mask_ycrcb(bgr)

        acne_img = simulate_acne(
            bgr, face_rect, skin,
            AcneParams(count=acne_count, red_strength=acne_strength)
        )
        ery_img = simulate_erythema(
            bgr, face_rect, skin,
            ErythemaParams(strength=erythema_strength)
        )

        stem = img_path.stem
        acne_path = acne_dir / f"{stem}.png"
        ery_path = ery_dir / f"{stem}.png"

        ok1 = cv2.imwrite(str(acne_path), acne_img)
        ok2 = cv2.imwrite(str(ery_path), ery_img)

        print(
            f"[{stem}] Saved: acne/{acne_path.name} ({'OK' if ok1 else 'FAILED'}) | "
            f"erythema/{ery_path.name} ({'OK' if ok2 else 'FAILED'})"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="path to input image OR folder of images")
    ap.add_argument("--out_dir", default="out", help="output directory")
    ap.add_argument("--acne_count", type=int, default=32)
    ap.add_argument("--acne_strength", type=float, default=0.46)
    ap.add_argument("--erythema_strength", type=float, default=0.52)
    args = ap.parse_args()

    run_simulation(
        input_path=args.input,
        out_dir=args.out_dir,
        acne_count=args.acne_count,
        acne_strength=args.acne_strength,
        erythema_strength=args.erythema_strength,
    )


if __name__ == "__main__":
    main()
