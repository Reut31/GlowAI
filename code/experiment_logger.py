import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt



class ExperimentLogger:
    def __init__(self, out_dir: str):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.out_dir / "metrics.csv"

        self.history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["epoch", "train_loss", "val_loss", "train_acc", "val_acc"]
                )

    def log_epoch(self, epoch, train_loss,
                  val_loss=None, train_acc=None, val_acc=None):
        self.history["epoch"].append(epoch)
        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_loss)
        self.history["train_acc"].append(train_acc)
        self.history["val_acc"].append(val_acc)

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, train_acc, val_acc])

    def _plot_single(self, x, ys, labels, title, ylabel, filename):
        plt.figure(figsize=(6, 4))
        for y, label in zip(ys, labels):
            if y is not None and any(v is not None for v in y):
                xs_clean = [xx for xx, yy in zip(x, y) if yy is not None]
                ys_clean = [yy for yy in y if yy is not None]
                plt.plot(xs_clean, ys_clean, marker="o", label=label)

        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.out_dir / filename)
        plt.close()

    def save_plots(self):
        epochs = self.history["epoch"]

        # Loss curves
        self._plot_single(
            epochs,
            [self.history["train_loss"], self.history["val_loss"]],
            ["Train loss", "Val loss"],
            title="Loss vs Epoch",
            ylabel="Loss",
            filename="loss_curve.png",
        )

        # Accuracy curves
        self._plot_single(
            epochs,
            [self.history["train_acc"], self.history["val_acc"]],
            ["Train acc", "Val acc"],
            title="Accuracy vs Epoch",
            ylabel="Accuracy",
            filename="accuracy_curve.png",
        )
