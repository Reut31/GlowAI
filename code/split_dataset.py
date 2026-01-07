import random
import shutil
from pathlib import Path

SEED = 123
VAL_SPLIT = 0.2  # 20% ל-val

IMG_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def list_images(folder: Path):
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix in IMG_EXTS])


def copy_split(src: Path, train_dst: Path, val_dst: Path):
    files = list_images(src)
    if not files:
        raise FileNotFoundError(f"No images found in: {src}")

    random.shuffle(files)
    n_val = max(1, int(len(files) * VAL_SPLIT))
    val_files = set(files[:n_val])

    train_dst.mkdir(parents=True, exist_ok=True)
    val_dst.mkdir(parents=True, exist_ok=True)

    for p in files:
        dst = val_dst if p in val_files else train_dst
        shutil.copy2(p, dst / p.name)

    print(f"{src.name}: total={len(files)} | train={len(files)-n_val} | val={n_val}")


def main():
    random.seed(SEED)

    project = Path("..")
    out_acne = project / "out" / "acne"
    out_ery  = project / "out" / "erythema"
    out_none = project / "none"  # כאן התמונות הנקיות (בלי מצב)

    dataset = project / "dataset"
    train = dataset / "train"
    val = dataset / "val"

    copy_split(out_acne, train / "acne", val / "acne")
    copy_split(out_ery,  train / "erythema", val / "erythema")
    copy_split(out_none, train / "none", val / "none")

    print("✅ Done. Created dataset/train and dataset/val")


if __name__ == "__main__":
    main()
