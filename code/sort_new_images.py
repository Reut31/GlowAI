import argparse
from pathlib import Path
import shutil

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from PIL import Image
import torch.nn.functional as F


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".JPG", ".JPEG", ".PNG")


class ImageFolderNoLabels(Dataset):
    def __init__(self, root: str, img_size: int = 224):
        self.root = Path(root)
        self.paths = [
            p for p in self.root.rglob("*")
            if p.suffix.lower() in IMAGE_EXTS
        ]
        if not self.paths:
            raise FileNotFoundError(f"No images found under: {self.root}")

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
        img = Image.open(path).convert("RGB")
        x = self.tf(img)
        return x, str(path)


def load_model_from_ckpt(ckpt_path: str, device: str = "cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)

    model_name = ckpt["model_name"]
    num_classes = len(ckpt["class_to_idx"])
    img_size = ckpt.get("img_size", 224)

    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    model, idx_to_class, img_size = load_model_from_ckpt(ckpt_path, device=device)

    ds = ImageFolderNoLabels(input_dir, img_size=img_size)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # תיקיות: לכל קלאס + low_confidence
    class_dirs = {}
    for idx, cls_name in idx_to_class.items():
        d = out_root / cls_name
        d.mkdir(parents=True, exist_ok=True)
        class_dirs[cls_name] = d

    low_conf_dir = out_root / "low_confidence"
    low_conf_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for xb, paths in dl:
        xb = xb.to(device)
        logits = model(xb)
        probs = F.softmax(logits, dim=1)

        confs, preds = probs.max(dim=1)

        for path_str, pred_idx, conf in zip(paths, preds.cpu(), confs.cpu()):
            total += 1
            pred_idx = int(pred_idx.item())
            conf = float(conf.item())

            cls_name = idx_to_class[pred_idx]

            if conf < conf_threshold:
                target_dir = low_conf_dir
            else:
                target_dir = class_dirs[cls_name]

            src = Path(path_str)
            dst = target_dir / src.name

            shutil.copy2(src, dst)

            print(f"{src.name:30s} -> {target_dir.name}  (conf={conf:.3f})")

    print(f"Done. Processed {total} images. Results in: {out_root}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to checkpoint best.pt")
    ap.add_argument("--input", required=True, help="folder with images to classify")
    ap.add_argument("--out", required=True, help="output folder for results_runX")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--conf", type=float, default=0.6, help="low-confidence threshold")
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
