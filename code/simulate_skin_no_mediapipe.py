import csv
from pathlib import Path
import matplotlib

# Set the backend to 'Agg' to allow plotting without a GUI (essential for running on servers or headless environments)
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class ExperimentLogger:
    """
    A utility class to track training progress, save metrics to CSV, 
    and generate visualization plots (Loss & Accuracy curves).
    """

    def __init__(self, out_dir: str):
        """
        Initializes the experiment logger.

        Args:
            out_dir (str): The directory path where logs and plots will be saved.
        """
        self.out_dir = Path(out_dir)
        # Create the output directory if it doesn't exist
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

        # Initialize the CSV file with headers if it doesn't exist
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["epoch", "train_loss", "val_loss", "train_acc", "val_acc"]
                )

    def log_epoch(self, epoch, train_loss, val_loss=None, train_acc=None, val_acc=None):
        """
        Logs metrics for a single epoch to memory and appends them to the CSV file.

        Args:
            epoch (int): Current epoch number.
            train_loss (float): Training loss value.
            val_loss (float, optional): Validation loss value.
            train_acc (float, optional): Training accuracy value.
            val_acc (float, optional): Validation accuracy value.
        """
        # Update in-memory history
        self.history["epoch"].append(epoch)
        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_loss)
        self.history["train_acc"].append(train_acc)
        self.history["val_acc"].append(val_acc)

        # Append row to CSV file immediately (so data isn't lost if the script crashes)
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, train_acc, val_acc])

    def _plot_single(self, x, ys, labels, title, ylabel, filename):
        """
        Helper function to generate and save a single line plot using Matplotlib.
        
        Args:
            x (list): X-axis values (epochs).
            ys (list of lists): Y-axis data series (e.g., [train_loss, val_loss]).
            labels (list of str): Legend labels for each series.
            title (str): Title of the plot.
            ylabel (str): Y-axis label.
            filename (str): Output filename.
        """
        plt.figure(figsize=(6, 4))
        
        # Plot each data series (filtering out None values if validation was skipped)
        for y, label in zip(ys, labels):
            if y is not None and any(v is not None for v in y):
                # Filter out None values to ensure continuous plotting
                xs_clean = [xx for xx, yy in zip(x, y) if yy is not None]
                ys_clean = [yy for yy in y if yy is not None]
                plt.plot(xs_clean, ys_clean, marker="o", label=label)

        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3) # Add a subtle grid for better readability
        plt.legend()
        plt.tight_layout()
        
        # Save the plot to the output directory
        plt.savefig(self.out_dir / filename)
        plt.close() # Close figure to free memory

    def save_plots(self):
        """
        Generates and saves the final Loss and Accuracy curves based on recorded history.
        Should be called at the end of training or periodically.
        """
        epochs = self.history["epoch"]

        # Generate and save Loss vs Epoch plot
        self._plot_single(
            epochs,
            [self.history["train_loss"], self.history["val_loss"]],
            ["Train loss", "Val loss"],
            title="Loss vs Epoch",
            ylabel="Loss",
            filename="loss_curve.png",
        )

        # Generate and save Accuracy vs Epoch plot
        self._plot_single(
            epochs,
            [self.history["train_acc"], self.history["val_acc"]],
            ["Train acc", "Val acc"],
            title="Accuracy vs Epoch",
            ylabel="Accuracy",
            filename="accuracy_curve.png",
        )
