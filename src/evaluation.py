import torch
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


class ConfusionMatrixEvaluator:
    """Evaluate model performance using confusion matrix and classification metrics.

    Supports custom category labels from config.
    """

    def __init__(self, categories=None, label_map=None):
        """
        Args:
            categories: List of category names (e.g., ["Armstand", "Backward", ...]).
            label_map: Dict mapping label indices to names (e.g., {"0": "Armstand", ...}).
        """
        self.categories = categories or []
        self.label_map = label_map or {}
        self.predictions = []
        self.ground_truth = []

    def update(self, preds, labels):
        """Accumulate predictions and labels.

        Args:
            preds: Model predictions, shape (B,) or (B, num_classes). If logits, takes argmax.
            labels: Ground truth labels, shape (B,).
        """
        if isinstance(preds, torch.Tensor):
            if preds.ndim > 1:
                preds = torch.argmax(preds, dim=1)
            preds = preds.cpu().numpy()
        else:
            preds = np.asarray(preds)

        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        else:
            labels = np.asarray(labels)

        self.predictions.extend(preds.flatten())
        self.ground_truth.extend(labels.flatten())

    def reset(self):
        """Clear accumulated predictions and labels."""
        self.predictions = []
        self.ground_truth = []

    def compute_confusion_matrix(self):
        """Compute and return confusion matrix."""
        if not self.predictions:
            raise ValueError("No predictions accumulated. Call update() first.")
        num_classes = len(self.categories) if self.categories else max(
            max(self.predictions), max(self.ground_truth)
        ) + 1
        cm = confusion_matrix(
            self.ground_truth, self.predictions, labels=list(range(num_classes))
        )
        return cm

    def compute_metrics(self):
        """Compute accuracy, precision, recall, F1."""
        if not self.predictions:
            raise ValueError("No predictions accumulated. Call update() first.")

        accuracy = accuracy_score(self.ground_truth, self.predictions)
        precision = precision_score(
            self.ground_truth, self.predictions, average="weighted", zero_division=0
        )
        recall = recall_score(
            self.ground_truth, self.predictions, average="weighted", zero_division=0
        )
        f1 = f1_score(
            self.ground_truth, self.predictions, average="weighted", zero_division=0
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def report(self):
        """Return dict with confusion matrix and per-class metrics."""
        cm = self.compute_confusion_matrix()
        metrics = self.compute_metrics()

        # Per-class metrics
        per_class = {}
        num_classes = cm.shape[0]
        for i in range(num_classes):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - tp - fp - fn

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            class_name = self.categories[i] if i < len(self.categories) else f"Class_{i}"
            per_class[class_name] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int(cm[i, :].sum()),
            }

        return {
            "confusion_matrix": cm,
            "overall_metrics": metrics,
            "per_class_metrics": per_class,
            "categories": self.categories or [f"Class_{i}" for i in range(num_classes)],
        }

    def print_report(self):
        """Print evaluation report to stdout."""
        report = self.report()
        print("\n" + "=" * 60)
        print("CONFUSION MATRIX EVALUATION REPORT")
        print("=" * 60)

        print("\nOverall Metrics:")
        for key, val in report["overall_metrics"].items():
            print(f"  {key.capitalize()}: {val:.4f}")

        print("\nPer-Class Metrics:")
        for class_name, metrics in report["per_class_metrics"].items():
            print(f"\n  {class_name}:")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall:    {metrics['recall']:.4f}")
            print(f"    F1-Score:  {metrics['f1']:.4f}")
            print(f"    Support:   {metrics['support']}")

        print("\nConfusion Matrix:")
        cm = report["confusion_matrix"]
        print("  Predicted →")
        print("  " + " ".join([f"{cat:>10}" for cat in report["categories"]]))
        for i, row in enumerate(cm):
            print(f"  {report['categories'][i]:>10} | " + " ".join([f"{v:>10}" for v in row]))

        print("=" * 60 + "\n")
