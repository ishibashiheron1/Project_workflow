"""ImmersaNET clip-level species classification evaluation.

Standard library only.

Expected input for each labeled audio clip:
    {
        "scores": {
            "sperm_whale": 0.85,
            "bottlenose_dolphin": 0.15
        },
        "detected": True,
        "ground_truth_label": "sperm_whale"
    }

Metrics:
    - Accuracy
    - Per-species precision, recall, and F1
    - Macro precision, recall, and F1
    - Per-species Average Precision (AP)
    - Mean Average Precision (mAP)
    - Confusion matrix

This module evaluates clip-level species classification,
NOT timestamped event detection.

A non-detection counts as a missed classification.
For AP/mAP, a non-detected clip receives a score of zero
for every species, assuming scores are calibrated confidences.

Demo predictions are synthetic and do not represent
actual ImmersaNET model performance.
"""

from collections import Counter
from math import isfinite
from typing import Any


class EvaluationModule:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def update(self, prediction: dict[str, Any]) -> None:
        """Record one labeled clip's classifier output."""

        if not isinstance(prediction, dict):
            raise ValueError("Prediction must be a dictionary.")

        scores = prediction.get(
            "scores",
            prediction.get("class_labels")
        )

        truth = prediction.get(
            "ground_truth_label",
            prediction.get("ground_truth")
        )

        detected = prediction.get("detected")

        if not isinstance(scores, dict) or not scores:
            raise ValueError(
                "'scores' must be a nonempty dictionary: "
                "{species: confidence_score}."
            )

        if not isinstance(truth, str) or not truth.strip():
            raise ValueError(
                "'ground_truth_label' must be a species name."
            )

        if not isinstance(detected, bool):
            raise ValueError(
                "'detected' must be True or False."
            )

        for species, score in scores.items():
            if not isinstance(species, str) or not species.strip():
                raise ValueError(
                    "Species names must be nonempty strings."
                )

            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not isfinite(score)
                or not 0 <= score <= 1
            ):
                raise ValueError(
                    "Scores must be finite numbers between 0 and 1."
                )

        predicted = (
            max(scores, key=scores.get)
            if detected
            else None
        )

        self.records.append({
            "actual": truth,
            "predicted": predicted,
            "detected": detected,
            "scores": dict(scores)
        })

    @staticmethod
    def average_precision(
        labels: list[int],
        scores: list[float]
    ) -> float | None:
        """Calculate binary average precision from ranked scores.

        AP is the average of precision at the ranks occupied
        by positive examples.

        Returns None if there are no positive examples.

        Tied scores are processed as a group to avoid
        dependence on the original sample ordering.
        """

        if len(labels) != len(scores):
            raise ValueError(
                "Labels and scores must have the same length."
            )

        if any(label not in (0, 1) for label in labels):
            raise ValueError(
                "Labels must contain only 0 and 1."
            )

        if any(
            not isfinite(score)
            for score in scores
        ):
            raise ValueError(
                "Scores must be finite."
            )

        positives = sum(labels)

        if positives == 0:
            return None

        ranked = sorted(
            zip(scores, labels),
            key=lambda item: item[0],
            reverse=True
        )

        true_positives = 0
        evaluated = 0
        previous_recall = 0.0
        ap = 0.0
        index = 0

        while index < len(ranked):
            current_score = ranked[index][0]
            group_positives = 0
            group_size = 0

            while (
                index < len(ranked)
                and ranked[index][0] == current_score
            ):
                group_positives += ranked[index][1]
                group_size += 1
                index += 1

            true_positives += group_positives
            evaluated += group_size

            precision = true_positives / evaluated
            recall = true_positives / positives

            ap += (
                recall - previous_recall
            ) * precision

            previous_recall = recall

        return ap

    def summary(self) -> dict[str, Any]:
        """Calculate classification metrics and mAP."""

        if not self.records:
            return {
                "total": 0,
                "accuracy": 0.0,
                "macro_precision": 0.0,
                "macro_recall": 0.0,
                "macro_f1": 0.0,
                "map": 0.0,
                "per_class": {},
                "confusion_matrix": {}
            }

        classes = sorted(
            {r["actual"] for r in self.records}
            |
            {
                r["predicted"]
                for r in self.records
                if r["predicted"] is not None
            }
            |
            {
                species
                for r in self.records
                for species in r["scores"]
            }
        )

        per_class = {}
        ap_values = []

        for species in classes:
            tp = sum(
                r["actual"] == species
                and r["predicted"] == species
                for r in self.records
            )

            fp = sum(
                r["actual"] != species
                and r["predicted"] == species
                for r in self.records
            )

            fn = sum(
                r["actual"] == species
                and r["predicted"] != species
                for r in self.records
            )

            support = tp + fn

            precision = (
                tp / (tp + fp)
                if tp + fp
                else 0.0
            )

            recall = (
                tp / support
                if support
                else 0.0
            )

            f1 = (
                2 * precision * recall
                / (precision + recall)
                if precision + recall
                else 0.0
            )

            binary_labels = [
                1 if r["actual"] == species else 0
                for r in self.records
            ]

            confidence_scores = [
                (
                    r["scores"].get(species, 0.0)
                    if r["detected"]
                    else 0.0
                )
                for r in self.records
            ]

            ap = self.average_precision(
                binary_labels,
                confidence_scores
            )

            if ap is not None:
                ap_values.append(ap)

            per_class[species] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "average_precision": (
                    round(ap, 4)
                    if ap is not None
                    else None
                ),
                "support": support,
                "tp": tp,
                "fp": fp,
                "fn": fn
            }

        # Macro averages exclude classes with no ground truth.
        present = [
            metrics
            for metrics in per_class.values()
            if metrics["support"] > 0
        ]

        def macro_average(key: str) -> float:
            if not present:
                return 0.0

            return round(
                sum(
                    metrics[key]
                    for metrics in present
                ) / len(present),
                4
            )

        # Mean AP across classes with positive examples.
        mean_ap = (
            sum(ap_values) / len(ap_values)
            if ap_values
            else 0.0
        )

        confusion = {
            species: dict(
                Counter(
                    (
                        r["predicted"]
                        if r["predicted"] is not None
                        else "NO_DETECTION"
                    )
                    for r in self.records
                    if r["actual"] == species
                )
            )
            for species in classes
        }

        correct = sum(
            r["actual"] == r["predicted"]
            for r in self.records
        )

        return {
            "total": len(self.records),
            "accuracy": round(
                correct / len(self.records), 4
            ),
            "macro_precision": macro_average(
                "precision"
            ),
            "macro_recall": macro_average(
                "recall"
            ),
            "macro_f1": macro_average(
                "f1"
            ),
            "map": round(mean_ap, 4),
            "per_class": per_class,
            "confusion_matrix": confusion
        }

    def print_report(self) -> None:
        """Print a readable evaluation report."""

        result = self.summary()

        print(
            f"\nImmersaNET Evaluation "
            f"- {result['total']} clips"
        )

        print("-" * 75)

        print(
            f"Accuracy: {result['accuracy']:.2%}"
        )

        print(
            f"Macro Precision: "
            f"{result['macro_precision']:.2%}"
        )

        print(
            f"Macro Recall: "
            f"{result['macro_recall']:.2%}"
        )

        print(
            f"Macro F1: "
            f"{result['macro_f1']:.2%}"
        )

        print(
            f"mAP: {result['map']:.2%}"
        )

        print("\nPer-species performance:")

        print(
            f"{'Species':<24} "
            f"{'Precision':>10} "
            f"{'Recall':>10} "
            f"{'F1':>8} "
            f"{'AP':>8} "
            f"{'Support':>8}"
        )

        print("-" * 85)

        for species, metrics in result["per_class"].items():
            ap = metrics["average_precision"]

            ap_text = (
                f"{ap:.2f}"
                if ap is not None
                else "N/A"
            )

            print(
                f"{species:<24} "
                f"{metrics['precision']:>10.2f} "
                f"{metrics['recall']:>10.2f} "
                f"{metrics['f1']:>8.2f} "
                f"{ap_text:>8} "
                f"{metrics['support']:>8}"
            )

        print(
            "\nConfusion matrix "
            "(actual -> predicted counts):"
        )

        for actual, predicted_counts in (
            result["confusion_matrix"].items()
        ):
            print(
                f"  {actual}: {predicted_counts}"
            )

    def reset(self) -> None:
        """Clear all stored predictions."""
        self.records.clear()


if __name__ == "__main__":
    # Synthetic demo data ONLY.
    # These are NOT trained ImmersaNET predictions.

    evaluator = EvaluationModule()

    evaluator.update({
        "scores": {
            "sperm_whale": 0.90,
            "bottlenose_dolphin": 0.10
        },
        "detected": True,
        "ground_truth_label": "sperm_whale"
    })

    evaluator.update({
        "scores": {
            "sperm_whale": 0.65,
            "bottlenose_dolphin": 0.35
        },
        "detected": True,
        "ground_truth_label": "bottlenose_dolphin"
    })

    evaluator.update({
        "scores": {
            "sperm_whale": 0.20,
            "bottlenose_dolphin": 0.80
        },
        "detected": True,
        "ground_truth_label": "bottlenose_dolphin"
    })

    evaluator.update({
        "scores": {
            "sperm_whale": 0.80,
            "bottlenose_dolphin": 0.20
        },
        "detected": False,
        "ground_truth_label": "sperm_whale"
    })

    evaluator.print_report()
