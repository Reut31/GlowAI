# GlowAI – Synthetic Skin Condition Classification

This repo contains the code and experiments for a GenAI course project on:

1. **Synthetic generation** of facial skin conditions (acne, erythema) from clean face photos.
2. **Image classification** into three classes:
   - `acne`
   - `erythema`
   - `none`

The project is implemented in **PyTorch** with **timm** backbones and custom OpenCV-based simulators.

---

## Project structure

Current layout (simplified):

```text
GlowAIProj/
├─ code/
│   ├─ train_skin_cls.py           # main training + optional classification of new images
│   ├─ experiment_logger.py        # logging to CSV + plots (loss/acc)
│   ├─ simulate_skin_no_mediapipe.py  # synthetic acne/erythema generator (OpenCV)
│   ├─ sort_new_images.py          # classify new images into results_runX
│   └─ split_dataset.py            # helper for splitting data into train/val
│
├─ dataset/
│   ├─ train/
│   │   ├─ acne/
│   │   ├─ erythema/
│   │   └─ none/
│   ├─ val/
│   │   ├─ acne/
│   │   ├─ erythema/
│   │   └─ none/
│   ├─ feedback/
│   │   ├─ acne/
│   │   ├─ erythema/
│   │   └─ none/
│   └─ none/                       # raw "clean" faces (no label) – optional
│
├─ checkpoints/
│   ├─ run3/
│   │   ├─ best.pt                 # best model for this run (ignored by git)
│   │   └─ logs/
│   │       ├─ metrics.csv
│   │       ├─ loss_curve.png
│   │       ├─ accuracy_curve.png
│   │       └─ confusion_matrix.png
│   ├─ run4/
│   ├─ run5/
│   ├─ run6/
│   ├─ run7/
│   ├─ run8/
│   └─ run9/
│
├─ out/
│   ├─ acne/                       # example synthetic outputs (optional)
│   └─ erythema/
│
├─ to_classify/
│   ├─ images/                     # new images to classify with the trained model
│   ├─ results/                    # generic results folder (optional)
│   ├─ results_run2/
│   ├─ results_run3/
│   ├─ results_run4/
│   ├─ results_run5/
│   ├─ results_run6/
│   ├─ results_run7/
│   ├─ results_run8/
│   └─ results_run9/
│
├─ .gitignore
└─ README.md
Model weight files (*.pt) are ignored by git for size reasons.
Logs + plots (metrics.csv, loss_curve.png, accuracy_curve.png, confusion_matrix.png) are tracked.

Setup
1. Virtual environment (optional but recommended)
bash
Copy code
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate
2. Install dependencies
Minimal set (can be refined into requirements.txt):

bash
Copy code
pip install torch torchvision timm tqdm matplotlib numpy opencv-python pillow pandas
Data format
The classifier expects:

text
Copy code
dataset/
├─ train/
│   ├─ acne/
│   ├─ erythema/
│   └─ none/
└─ val/
    ├─ acne/
    ├─ erythema/
    └─ none/
Each class folder contains RGB images with that label.

dataset/feedback/ is used to collect misclassified examples or manually reviewed cases:

text
Copy code
dataset/feedback/
├─ acne/
├─ erythema/
└─ none/
This can be merged into train/ for improved future runs.

1️⃣ Synthetic generation (acne & erythema)
Script: code/simulate_skin_no_mediapipe.py

Given a folder of clean face images, it generates two folders: acne/ and erythema/, where lesions / redness are simulated using classic CV (no diffusion model here, by design for the course project).

Example:

bash
Copy code
python code/simulate_skin_no_mediapipe.py \
  --input dataset/none \
  --out_dir out/simulated_skin \
  --acne_count 32 \
  --acne_strength 0.46 \
  --erythema_strength 0.52
Result:

text
Copy code
out/simulated_skin/
├─ acne/
└─ erythema/
These images can then be moved/merged into dataset/train/acne and dataset/train/erythema to create synthetic-augmented training sets.

2️⃣ Training the classifier
Main training script: code/train_skin_cls.py

Key features:

Backbone from timm (efficientnet_b2 by default).

Two-phase training:

Freeze backbone, train classifier head (--lr_head).

Unfreeze, fine-tune all layers (--lr_ft after --freeze_epochs).

Logs per-epoch metrics to metrics.csv.

Generates plots of loss/accuracy and a confusion matrix on the validation set.

Optionally classifies new images (from to_classify/images) into results_runX folders after training finishes.

Typical run (example: run3):

bash
Copy code
python code/train_skin_cls.py \
  --data dataset \
  --out checkpoints/run3 \
  --model efficientnet_b2 \
  --img_size 224 \
  --batch 16 \
  --epochs 15 \
  --lr_head 3e-4 \
  --lr_ft 1e-5 \
  --freeze_epochs 4 \
  --classify_images to_classify/images \
  --classify_out to_classify/results_run3
This will:

Train a model on dataset/train / dataset/val.

Save the best checkpoint to checkpoints/run3/best.pt.

Log metrics to checkpoints/run3/logs/metrics.csv.

Classify all images in to_classify/images into:

text
Copy code
to_classify/results_run3/
  ├─ acne/
  ├─ erythema/
  ├─ none/
  └─ low_confidence/
Generate:

text
Copy code
checkpoints/run3/logs/
  ├─ loss_curve.png
  ├─ accuracy_curve.png
  └─ confusion_matrix.png
Additional runs (run4, run5, … run9) correspond to different experiments (e.g. different seeds, data, or hyperparameters) and are stored under checkpoints/runX with matching to_classify/results_runX outputs.

3️⃣ Classifying new images (standalone)
If you already have a checkpoint and just want to classify new images without retraining, use:

Script: code/sort_new_images.py

bash
Copy code
python code/sort_new_images.py \
  --ckpt checkpoints/run3/best.pt \
  --input to_classify/images \
  --out to_classify/results_run3 \
  --batch 16 \
  --conf 0.6
Images with max probability < conf go to low_confidence/.

Others go to the folder of the predicted class.

4️⃣ Training logs & metrics
For each run checkpoints/runX (where X ∈ {3,…,9}):

logs/metrics.csv
Contains per-epoch: epoch, train_loss, val_loss, train_acc, val_acc.

logs/loss_curve.png
Train & validation loss vs. epoch.

logs/accuracy_curve.png
Train & validation accuracy vs. epoch.

logs/confusion_matrix.png
Confusion matrix on dataset/val using the final model for that run.

These artifacts are used in the project report and slides to compare runs (e.g. run3, run6, run9).

Repro steps (short version)
Prepare dataset/train and dataset/val with the 3 target classes.

(Optional) Generate synthetic acne/erythema using
simulate_skin_no_mediapipe.py and add to the dataset.

Run training with different configurations (e.g. run3, run4, …) via train_skin_cls.py.

Inspect:

logs (metrics.csv)

plots (loss_curve.png, accuracy_curve.png)

confusion matrices

qualitative classification results in to_classify/results_runX.

For improved versions, incorporate dataset/feedback into the training set and rerun.

