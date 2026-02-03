import argparse
from pathlib import Path
import shutil

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from PIL import Image
import torch.nn.functional as F

# Supported image extensions
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".JPG", ".JPEG", ".PNG")


class ImageFolderNoLabels(Dataset):
    """
    Custom Dataset class for loading images from a folder without requiring class subfolders.
    Used for inference on new, unlabeled data.
    """
    def __init__(self, root: str, img_size: int = 224):
        self.root = Path(root)
        # Recursively find all images in the directory
        self.paths = [
            p for p in self.root.rglob("*")
            if p.suffix.lower() in IMAGE_EXTS
        ]
        if not self.paths:
            raise FileNotFoundError(f"No images found under: {self.root}")

        # Standard preprocessing: Resize -> Tensor -> Normalize (ImageNet stats)
        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB") # Ensure 3 channels
        x = self.tf(img)
        return x, str(path)


def load_model_from_ckpt(ckpt_path: str, device: str = "cpu"):
    """
    Loads a trained model architecture and weights from a checkpoint file.
    
    Args:
        ckpt_path (str): Path to the .pt file containing state_dict and metadata.
        device (str): Computation device ('cpu' or 'cuda').
        
    Returns:
        model: The loaded PyTorch model in eval mode.
        idx_to_class: Dictionary mapping prediction indices back to class names.
        img_size: The input image size the model expects.
    """
    ckpt = torch.load(ckpt_path, map_location=device)

    model_name = ckpt["model_name"]
    num_classes = len(ckpt["class_to_idx"])
    img_size = ckpt.get("img_size", 224)

    # Re-create the model architecture (must match training)
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval() # Set to evaluation mode (disable dropout, etc.)

    # Invert the mapping to get Class Name from Index
    idx_to_class = {v: k for k, v in ckpt["class_to_idx"].items()}

    return model, idx_to_class, img_size


@torch.no_grad()
def sort_images(
    ckpt_path: str,
    input_dir: str,
    output_dir: str,
    batch_size: int = 16,
    conf_threshold: float = 0.6,
):
    """
    Main inference loop:
    1. Loads the model.
    2. Iterates over unlabeled images.
    3. Predicts class and confidence score.
    4. Sorts images into folders (Acne/Erythema/None) or 'low_confidence'.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Load model and metadata
    model, idx_to_class, img_size = load_model_from_ckpt(ckpt_path, device=device)

    # Setup dataset and dataloader
    ds = ImageFolderNoLabels(input_dir, img_size=img_size)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Create output directories for each class (e.g., out/acne, out/none)
    class_dirs = {}
    for idx, cls_name in idx_to_class.items():
        d = out_root / cls_name
        d.mkdir(parents=True, exist_ok=True)
        class_dirs[cls_name] = d

    # Create a separate directory for ambiguous cases
    low_conf_dir = out_root / "low_confidence"
    low_conf_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    print(f"Starting classification of {len(ds)} images...")

    for xb, paths in dl:
        xb = xb.to(device)
        
        # Forward pass
        logits = model(xb)
        probs = F.softmax(logits, dim=1) # Convert logits to probabilities (0-1)

        # Get the predicted class index and its confidence score
        confs, preds = probs.max(dim=1)

        for path_str, pred_idx, conf in zip(paths, preds.cpu(), confs.cpu()):
            total += 1
            pred_idx = int(pred_idx.item())
            conf = float(conf.item())

            cls_name = idx_to_class[pred_idx]

            # Decision logic: Low confidence vs. High confidence
            if conf < conf_threshold:
                target_dir = low_conf_dir
            else:
                target_dir = class_dirs[cls_name]

            # Copy image to the target folder
            src = Path(path_str)
            dst = target_dir / src.name
            shutil.copy2(src, dst)

            print(f"{src.name:30s} -> {target_dir.name}  (conf={conf:.3f})")

    print(f"Done. Processed {total} images. Results saved in: {out_root}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to the trained model checkpoint (best.pt)")
    ap.add_argument("--input", required=True, help="Folder containing new images to classify")
    ap.add_argument("--out", required=True, help="Output folder where sorted images will be saved")
    ap.add_argument("--batch", type=int, default=16, help="Batch size for inference")
    ap.add_argument("--conf", type=float, default=0.6, help="Confidence threshold (0.0 - 1.0). Images below this go to 'low_confidence'")
    args = ap.parse_args()

    sort_images(
        ckpt_path=args.ckpt,
        input_dir=args.input,
        output_dir=args.out,
        batch_size=args.batch,
        conf_threshold=args.conf,
    )


if __name__ == "__main__":
    main()
