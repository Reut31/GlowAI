import csv
from pathlib import Path
import matplotlib

# Set backend to 'Agg' to write to files without needing a display (GUI).
# This prevents errors when running on servers or containers.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class ExperimentLogger:
    """
    A utility class to track training progress, save metrics to a CSV file,
    and generate visualization plots (Loss & Accuracy curves) automatically.
    """

    def __init__(self, out_dir: str):
        """
        Initializes the logger.

        Args:
            out_dir (str): Path to the directory where logs and plots will be saved.
        """
        self.out_dir = Path(out_dir)
        # Create directory if it doesn't exist (parents=True allows creating nested dirs)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.out_dir / "metrics.csv"

        # Dictionary to store metrics in memory for plotting later
        self.history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

        # Initialize CSV with headers if it doesn't exist yet
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["epoch", "train_loss", "val_loss", "train_acc", "val_acc"]
                )

    def log_epoch(self, epoch, train_loss,
                  val_loss=None, train_acc=None, val_acc=None):
        """
        Logs metrics for a single epoch. Updates memory and appends to CSV.

        Args:
            epoch (int): Current epoch number.
            train_loss (float): Loss on training set.
            val_loss (float, optional): Loss on validation set.
            train_acc (float, optional): Accuracy on training set.
            val_acc (float, optional): Accuracy on validation set.
        """
        # 1. Update in-memory history
        self.history["epoch"].append(epoch)
        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_loss)
        self.history["train_acc"].append(train_acc)
        self.history["val_acc"].append(val_acc)

        # 2. Append to CSV file immediately (safeguard against crashes)
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, train_acc, val_acc])

    def _plot_single(self, x, ys, labels, title, ylabel, filename):
        """
        Helper method to generate and save a single plot (e.g., Loss or Accuracy).
        Handles missing data points (None) gracefully.
        """
        plt.figure(figsize=(6, 4))
        
        # Iterate over multiple lines (e.g., train vs val)
        for y, label in zip(ys, labels):
            # Check if data exists and is not entirely None
            if y is not None and any(v is not None for v in y):
                # Filter out None values to ensure continuous plotting
                xs_clean = [xx for xx, yy in zip(x, y) if yy is not None]
                ys_clean = [yy for yy in y if yy is not None]
                plt.plot(xs_clean, ys_clean, marker="o", label=label)

        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)  # Add faint grid for readability
        plt.legend()
        plt.tight_layout()
        
        # Save plot to disk
        plt.savefig(self.out_dir / filename)
        plt.close()  # Close figure to free memory

    def save_plots(self):
        """
        Generates and saves the final training curves (Loss and Accuracy).
        Should be called at the end of training.
        """
        epochs = self.history["epoch"]

        # Generate Loss Curve
        self._plot_single(
            epochs,
            [self.history["train_loss"], self.history["val_loss"]],
            ["Train loss", "Val loss"],
            title="Loss vs Epoch",
            ylabel="Loss",
            filename="loss_curve.png",
        )

        # Generate Accuracy Curve
        self._plot_single(
            epochs,
            [self.history["train_acc"], self.history["val_acc"]],
            ["Train acc", "Val acc"],
            title="Accuracy vs Epoch",
            ylabel="Accuracy",
            filename="accuracy_curve.png",
        )
