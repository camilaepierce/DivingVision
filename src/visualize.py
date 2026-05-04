import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional


class BarGraphVisualizer:
    """Create bar graphs with support for custom categories defined in config.

    Supports per-category metrics, loss/accuracy tracking, and custom styling.
    """

    def __init__(self, categories: Optional[List[str]] = None, figsize: tuple = (10, 6)):
        """
        Args:
            categories: List of category names (e.g., from config).
            figsize: Figure size as (width, height).
        """
        self.categories = categories or []
        self.figsize = figsize

    def plot_per_class_metrics(
        self,
        metrics_dict: Dict[str, Dict[str, float]],
        metric_name: str = "f1",
        title: str = "Per-Class Metrics",
        save_path: Optional[str] = None,
    ):
        """Plot per-class metrics as a bar graph.

        Args:
            metrics_dict: Dict mapping class names to metric dicts with keys like 'f1', 'precision', etc.
            metric_name: Which metric to plot ('f1', 'precision', 'recall', etc.).
            title: Title for the plot.
            save_path: Path to save figure. If None, displays.

        Example:
            metrics = {
                'Armstand': {'f1': 0.85, 'precision': 0.90},
                'Backward': {'f1': 0.78, 'precision': 0.80},
            }
            viz.plot_per_class_metrics(metrics, metric_name='f1')
        """
        class_names = list(metrics_dict.keys())
        values = [metrics_dict[cls].get(metric_name, 0.0) for cls in class_names]

        fig, ax = plt.subplots(figsize=self.figsize)
        bars = ax.bar(class_names, values, color="steelblue", alpha=0.7, edgecolor="black")
        ax.set_ylabel(metric_name.capitalize(), fontsize=12)
        ax.set_xlabel("Class", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_ylim([0, 1.0])
        ax.grid(axis="y", alpha=0.3, linestyle="--")

        # Add value labels on top of bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=10)

        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved figure to {save_path}")
        else:
            plt.show()
        plt.close()

    def plot_category_support(
        self,
        support_dict: Dict[str, int],
        title: str = "Class Support (Sample Count)",
        save_path: Optional[str] = None,
    ):
        """Plot number of samples per category.

        Args:
            support_dict: Dict mapping class names to sample counts.
            title: Plot title.
            save_path: Path to save figure.

        Example:
            support = {'Armstand': 50, 'Backward': 45, 'Forward': 60}
            viz.plot_category_support(support)
        """
        class_names = list(support_dict.keys())
        counts = list(support_dict.values())

        fig, ax = plt.subplots(figsize=self.figsize)
        bars = ax.bar(class_names, counts, color="coral", alpha=0.7, edgecolor="black")
        ax.set_ylabel("Number of Samples", fontsize=12)
        ax.set_xlabel("Class", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(axis="y", alpha=0.3, linestyle="--")

        # Add count labels on top of bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, height, str(count),
                    ha="center", va="bottom", fontsize=10)

        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved figure to {save_path}")
        else:
            plt.show()
        plt.close()

    def plot_training_history(
        self,
        epochs: List[int],
        train_losses: List[float],
        test_accuracies: Optional[List[float]] = None,
        title: str = "Training History",
        save_path: Optional[str] = None,
    ):
        """Plot training loss and optionally test accuracy over epochs.

        Args:
            epochs: List of epoch numbers.
            train_losses: List of training losses per epoch.
            test_accuracies: Optional list of test accuracies per epoch.
            title: Plot title.
            save_path: Path to save figure.

        Example:
            viz.plot_training_history(
                epochs=[1, 2, 3, 4, 5],
                train_losses=[0.5, 0.4, 0.3, 0.25, 0.2],
                test_accuracies=[0.75, 0.80, 0.82, 0.85, 0.87]
            )
        """
        fig, ax1 = plt.subplots(figsize=self.figsize)

        # Plot training loss
        color_loss = "tab:blue"
        ax1.set_xlabel("Epoch", fontsize=12)
        ax1.set_ylabel("Training Loss", color=color_loss, fontsize=12)
        ax1.plot(epochs, train_losses, color=color_loss, marker="o", label="Train Loss")
        ax1.tick_params(axis="y", labelcolor=color_loss)
        ax1.grid(alpha=0.3, linestyle="--")

        # Plot test accuracy on secondary axis
        if test_accuracies:
            ax2 = ax1.twinx()
            color_acc = "tab:green"
            ax2.set_ylabel("Test Accuracy", color=color_acc, fontsize=12)
            ax2.plot(epochs, test_accuracies, color=color_acc, marker="s", label="Test Accuracy")
            ax2.tick_params(axis="y", labelcolor=color_acc)
            ax2.set_ylim([0, 1.0])

        ax1.set_title(title, fontsize=14, fontweight="bold")
        fig.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved figure to {save_path}")
        else:
            plt.show()
        plt.close()

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        categories: Optional[List[str]] = None,
        title: str = "Confusion Matrix",
        save_path: Optional[str] = None,
        normalize: bool = False,
    ):
        """Plot confusion matrix as a heatmap.

        Args:
            cm: Confusion matrix (2D numpy array).
            categories: Category names. Uses self.categories if not provided.
            title: Plot title.
            save_path: Path to save figure.
            normalize: If True, normalize by row (actual instances).

        Example:
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_true, y_pred)
            viz.plot_confusion_matrix(cm, categories=['A', 'B', 'C'])
        """
        categories = categories or self.categories or [f"Class_{i}" for i in range(cm.shape[0])]

        if normalize:
            cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
            display_cm = cm_norm
            fmt = ".2f"
        else:
            display_cm = cm
            fmt = "d"

        fig, ax = plt.subplots(figsize=(self.figsize[0], self.figsize[0]))
        im = ax.imshow(display_cm, cmap="Blues", aspect="auto")

        # Set ticks and labels
        ax.set_xticks(np.arange(len(categories)))
        ax.set_yticks(np.arange(len(categories)))
        ax.set_xticklabels(categories)
        ax.set_yticklabels(categories)

        # Rotate labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Count" if not normalize else "Proportion", rotation=270, labelpad=20)

        # Add text annotations
        for i in range(len(categories)):
            for j in range(len(categories)):
                val = display_cm[i, j]
                text_val = f"{val:.2f}" if normalize else f"{int(val)}"
                ax.text(j, i, text_val, ha="center", va="center",
                        color="white" if val > display_cm.max() / 2 else "black", fontsize=10)

        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("Actual", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        fig.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved figure to {save_path}")
        else:
            plt.show()
        plt.close()

    def plot_multi_class_comparison(
        self,
        data_dict: Dict[str, List[float]],
        class_names: Optional[List[str]] = None,
        title: str = "Multi-Class Comparison",
        ylabel: str = "Score",
        save_path: Optional[str] = None,
    ):
        """Plot grouped bar chart for multiple metrics across classes.

        Args:
            data_dict: Dict mapping metric names to lists of values.
            class_names: Class names for x-axis. Uses self.categories if not provided.
            title: Plot title.
            ylabel: Y-axis label.
            save_path: Path to save figure.

        Example:
            data = {
                'Precision': [0.85, 0.80, 0.90],
                'Recall': [0.82, 0.75, 0.88],
                'F1': [0.83, 0.77, 0.89],
            }
            viz.plot_multi_class_comparison(data, class_names=['A', 'B', 'C'])
        """
        class_names = class_names or self.categories or [f"Class_{i}" for i in range(len(next(iter(data_dict.values()))))]
        x = np.arange(len(class_names))
        width = 0.2

        fig, ax = plt.subplots(figsize=self.figsize)
        for i, (metric_name, values) in enumerate(data_dict.items()):
            offset = width * (i - len(data_dict) / 2 + 0.5)
            ax.bar(x + offset, values, width, label=metric_name, alpha=0.8, edgecolor="black")

        ax.set_xlabel("Class", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_ylim([0, 1.0])

        fig.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved figure to {save_path}")
        else:
            plt.show()
        plt.close()
