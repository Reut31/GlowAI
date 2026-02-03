Condition Analysis & Data Generation

**GlowAI** is a computer vision project designed to address the scarcity of medical skin datasets. Standard datasets are hard to obtain due to privacy concerns.
This project introduces a **Synthetic Skin Simulator** that uses classic Computer Vision (OpenCV) to generate medically relevant training data, creating realistic **Acne** and **Erythema** lesions on clean faces to train robust classification models.

## 👥 Team
* **Ron Noiman**
* **Reut Sasson**
* **Lin Schneider**

---

## 🚀 Project Pipeline
Our workflow consists of three main stages:

<img width="1536" height="1024" alt="ChatGPT Image Jan 13, 2026, 02_08_06 PM" src="https://github.com/user-attachments/assets/97d64621-50df-4898-81a4-8b2df40ad142" />

### 1. Synthetic Data Generation (The Simulator)
* **Acne Simulation:** We use OpenCV to draw randomized circles with specific Gaussian blur and opacity to mimic the texture and inflammation of pimples.
* **Erythema Simulation:** We apply irregular noise masks and red-channel blending to simulate skin redness patches naturally.
* **Output:** A labeled dataset of synthetic skin conditions generated from clean facial images.



### 2. Model Training (The Classifier)
* **Architecture:** We use an **EfficientNet-B2** backbone pretrained on ImageNet.
* **Training Strategy:** A two-phase approach (Linear Probing followed by Fine-Tuning) to adapt the model to skin features.
* **Loss Function:** Cross Entropy Loss with Adam optimizer.

### 3. Validation (The Proof)
We evaluate the model on two distinct sets:
* **Synthetic Validation Set:** To measure learning on generated patterns.
* **Manual Real-World Test Set:** To prove the model generalizes to real images (Held-out data).

---

## 🛠️ Installation

### 1. Clone the repository

bash 
git clone [https://github.com/ReutSasson/GlowAIProj.git](https://github.com/YourUsername/GlowAIProj.git)
cd GlowAIProj/



### 2. Install Dependencies
pip install -r requirements.txt

### 📂 Folder Structure
```text
GlowAIProj/
│
├── code/
│   ├── simulate_skin_no_mediapipe.py   # The Simulator (OpenCV)
│   ├── train_skin_cls.py               # Main Training Script
│   └── experiment_logger.py            # Metrics logging
│
├── data/
│   ├── train/                          # Training set (Synthetic + Real)
│   ├── val/                            # Validation set
│   └── feedback/                       # Hard examples for retraining
│
├── results/                            # Visualizations & Metrics (Best Run)
│   ├── accuracy_curve.png
│   ├── confusion_matrix.png
│   ├── loss_curve.png
│   └── visual_abstract.png
│
├── slides/                             # Project Presentations (PDF/PPTX)
├── checkpoints/                        # Saved model weights
└── README.md

### 🏃‍♂️ How to Run
Step 1: Generate Data
Run the simulator to create synthetic acne/erythema on clean faces:
 
python code/simulate_skin_no_mediapipe.py --input data/none --out_dir out/simulated_skin

### Step 2: Train the Model
Train the EfficientNet-B2 classifier on the dataset: 

python code/train_skin_cls.py --data dataset --epochs 15 --out checkpoints/run_final


### 📊 Results Snapshot (Best Run)
Below are the results from our best experiment (Run 9), comparing performance on synthetic vs. real data.

<img width="600" height="500" alt="confusion_matrix" src="https://github.com/user-attachments/assets/e1163e01-88f6-4821-8deb-52ce7758c7a1" />

<img width="600" height="400" alt="accuracy_curve" src="https://github.com/user-attachments/assets/370b350a-43f8-4a23-96d2-891f18bf1ac2" />

<img width="600" height="400" alt="loss_curve" src="https://github.com/user-attachments/assets/8bb8b603-ca8a-4eaa-b40f-ef9745620797" />



### 💾 Data Access
The dataset is included in the dataset/ folder.


Submitted as a final project for the GenAI course at HIT.











