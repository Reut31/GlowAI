import random
import shutil
from pathlib import Path

# ----------------------------- Configuration -----------------------------
SEED = 123             # Fixed seed to ensure the split is reproducible
VAL_SPLIT = 0.2        # 20% of data will be reserved for Validation

# Supported image formats
IMG_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def list_images(folder: Path) -> list[Path]:
    """
    Scans a directory and returns a sorted list of valid image files.
    
    Args:
        folder (Path): The directory to scan.
        
    Returns:
        list[Path]: List of file paths found.
    """
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix in IMG_EXTS])


def copy_split(src: Path, train_dst: Path, val_dst: Path):
    """
    Randomly shuffles images from the source folder and copies them 
    into Train and Validation destination folders based on the split ratio.

    Args:
        src (Path): Source directory containing all images of a specific class.
        train_dst (Path): Destination directory for training set.
        val_dst (Path): Destination directory for validation set.
    """
    files = list_images(src)
    if not files:
        print(f"Warning: No images found in {src}, skipping...")
        return

    # Shuffle files randomly to ensure random distribution
    random.shuffle(files)
    
    # Calculate split index
    n_val = max(1, int(len(files) * VAL_SPLIT))
    val_files = set(files[:n_val])

    # Ensure destination directories exist
    train_dst.mkdir(parents=True, exist_ok=True)
    val_dst.mkdir(parents=True, exist_ok=True)

    # Copy files
    for p in files:
        dst = val_dst if p in val_files else train_dst
        # shutil.copy2 preserves file metadata (timestamps, etc.)
        shutil.copy2(p, dst / p.name)

    print(f"Processed {src.name}: Total={len(files)} | Train={len(files)-n_val} | Val={n_val}")


def main():
    """
    Main execution function.
    Assumes the script is run from the 'code/' directory, so '..' refers to the project root.
    """
    random.seed(SEED)

    # Define paths relative to the project root
    project = Path("..") # Go up one level from 'code/'
    
    # Source folders (where generated/raw images are stored)
    out_acne = project / "out" / "acne"
    out_ery  = project / "out" / "erythema"
    out_none = project / "none"  # Raw clean images (Class 'None')

    # Destination folders (The final dataset structure)
    dataset = project / "dataset"
    train = dataset / "train"
    val = dataset / "val"

    print("Starting dataset split...")

    # Process Acne Class
    if out_acne.exists():
        copy_split(out_acne, train / "acne", val / "acne")
    
    # Process Erythema Class
    if out_ery.exists():
        copy_split(out_ery,  train / "erythema", val / "erythema")
    
    # Process None (Healthy) Class
    if out_none.exists():
        copy_split(out_none, train / "none", val / "none")

    print("\n✅ Done. Dataset organized successfully in 'dataset/train' and 'dataset/val'.")


if __name__ == "__main__":
    main()
