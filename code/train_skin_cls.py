import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from tqdm import tqdm

from experiment_logger import ExperimentLogger
import matplotlib.pyplot as plt
import numpy as np

# Import the classification function for post-training inference
from sort_new_images import sort_images  


def build_transforms(img_size: int = 224):
    """
    Constructs the data augmentation pipeline for training and the standard transform for validation.
    
    Args:
        img_size (int): Target image size (default 224 for EfficientNet).
    Returns:
        tuple: (train_transforms, val_transforms)
    """
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5), # Data Augmentation: Random flip
        transforms.RandomRotation(8),           # Data Augmentation: Slight rotation
        transforms.ColorJitter(                 # Data Augmentation: Color adjustments
            brightness=0.15,
            contrast=0.15,
            saturation=0.10,
            hue=0.02
        ),
        transforms.ToTensor(),
        transforms.Normalize(                   # Normalize using ImageNet statistics
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])
    return train_tf, val_tf


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    """
    Evaluates the model on the validation set.
    Returns the average loss and accuracy.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)

        pred = torch.argmax(logits, dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)

    return total_loss / max(1, total), correct / max(1, total)


def train_one_epoch(model, loader, device, optimizer, criterion):
    """
    Runs one epoch of training: Forward pass, Loss computation, Backward pass, Optimizer step.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in tqdm(loader, desc="train", leave=False):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        pred = torch.argmax(logits, dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)

    return total_loss / max(1, total), correct / max(1, total)


@torch.no_grad()
def save_confusion_matrix(model, loader, device, class_names, logs_dir: Path):
    """
    Generates and saves a Confusion Matrix plot to visualize per-class performance.
    """
    model.eval()
    num_classes = len(class_names)

    logs_dir.mkdir(parents=True, exist_ok=True)

    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        preds = torch.argmax(logits, dim=1)

        for t, p in zip(y.view(-1), preds.view(-1)):
            cm[t.long(), p.long()] += 1

    cm_np = cm.cpu().numpy()

    # Plotting using Matplotlib
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_np, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    # Add text annotations to the cells
    thresh = cm_np.max() / 2.0 if cm_np.max() > 0 else 0.5
    for i in range(num_classes):
        for j in range(num_classes):
            value = cm_np[i, j]
            ax.text(
                j,
                i,
                int(value),
                ha="center",
                va="center",
                color="white" if value > thresh else "black",
                fontsize=9,
            )

    plt.tight_layout()
    out_path = logs_dir / "confusion_matrix.png"
    plt.savefig(out_path)
    plt.close(fig)

    print("Saved confusion matrix to:", out_path)


def set_requires_grad(model, flag: bool):
    """
    Helper function to freeze (flag=False) or unfreeze (flag=True) model parameters.
    """
    for p in model.parameters():
        p.requires_grad = flag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        required=True,
        help="path to dataset folder containing train/ and val/",
    )
    ap.add_argument(
        "--model",
        default="efficientnet_b2",
        help="timm model name (efficientnet_b2 / convnext_tiny / resnet50)",
    )
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr_head", type=float, default=3e-4)
    ap.add_argument("--lr_ft", type=float, default=1e-5)
    ap.add_argument("--freeze_epochs", type=int, default=4)
    ap.add_argument(
        "--out",
        default="checkpoints",
        help="output dir for checkpoints + logs",
    )

    # New args - defining where to save results (e.g., results_run3)
    ap.add_argument(
        "--classify_images",
        type=str,
        default=None,
        help="folder with new images to classify AFTER training (e.g. out/to_classify/images)",
    )
    ap.add_argument(
        "--classify_out",
        type=str,
        default=None,
        help="output folder for classified images (e.g. out/to_classify/results_run3)",
    )

    args = ap.parse_args()

    # --- Step 1: Data Setup ---
    data_dir = Path(args.data)
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError("Expected dataset/train and dataset/val folders.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_tf, val_tf = build_transforms(args.img_size)

    train_ds = datasets.ImageFolder(str(train_dir), transform=train_tf)
    val_ds = datasets.ImageFolder(str(val_dir), transform=val_tf)

    print("Classes:", train_ds.classes)
    print("Class mapping:", train_ds.class_to_idx)

    num_classes = len(train_ds.classes)
    if num_classes != 3:
        print("WARNING: expected 3 classes, found:", num_classes)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    # --- Step 2: Model Initialization ---
    model = timm.create_model(
        args.model, pretrained=True, num_classes=num_classes
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = ExperimentLogger(out_dir / "logs")

    best_val_acc = 0.0
    optimizer = None

    # --- Step 3: Training Loop ---
    for epoch in range(1, args.epochs + 1):
        # Phase 1: Freeze backbone, train only the head
        if epoch == 1:
            set_requires_grad(model, False)
            for p in model.get_classifier().parameters():
                p.requires_grad = True
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr_head,
                weight_decay=1e-4,
            )
            print(f"Epoch {epoch}: training HEAD only (frozen backbone).")

        # Phase 2: Unfreeze all layers for Fine-Tuning
        if epoch == args.freeze_epochs + 1:
            set_requires_grad(model, True)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.lr_ft,
                weight_decay=1e-4,
            )
            print(f"Epoch {epoch}: fine-tuning ALL layers.")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, device, optimizer, criterion
        )
        val_loss, val_acc = evaluate(
            model, val_loader, device, criterion
        )

        # Log to CSV (Metrics only, plots generated later)
        logger.log_epoch(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            train_acc=train_acc,
            val_acc=val_acc,
        )

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.3f}"
        )

        # Save Best Checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = out_dir / "best.pt"
            torch.save(
                {
                    "model_name": args.model,
                    "state_dict": model.state_dict(),
                    "class_to_idx": train_ds.class_to_idx,
                    "img_size": args.img_size,
                },
                ckpt_path,
            )
            print("Saved best checkpoint to:", ckpt_path)

    print("Done training. Best val acc:", best_val_acc)

    # --- Step 4: Post-Training Tasks ---

    # 1. First - Results: Classify new images to results_runX (if arguments provided)
    if args.classify_images is not None and args.classify_out is not None:
        print(f"Sorting new images: {args.classify_images} -> {args.classify_out}")
        try:
            sort_images(
                ckpt_path=str(out_dir / "best.pt"),
                input_dir=args.classify_images,
                output_dir=args.classify_out,
                batch_size=args.batch,
                conf_threshold=0.6,
            )
            print("Finished sorting images.")
        except FileNotFoundError as e:
            # If no images found - skip but don't crash
            print(f"[WARN] Could not sort images (no images found). Skipping. Details: {e}")
    else:
        print("Skipping image classification step (no --classify_images/--classify_out).")

    # 2. Only after (or if skipped) - generate plots + confusion matrix
    logger.save_plots()
    logs_dir = out_dir / "logs"
    save_confusion_matrix(
        model=model,
        loader=val_loader,
        device=device,
        class_names=train_ds.classes,
        logs_dir=logs_dir,
    )


if __name__ == "__main__":
    main()
