"""Unit tests for ImmersaNET's clip-level EvaluationModule.

Place this file next to evaluation_module.py and run:
    python test_evaluation.py

These tests use synthetic predictions, not trained-model results.
"""

import unittest

from evaluation_module import EvaluationModule


class TestEvaluationModule(unittest.TestCase):
    def test_empty_summary(self):
        evaluator = EvaluationModule()
        result = evaluator.summary()
        self.assertEqual(result["total"], 0)
        self.assertEqual(result["accuracy"], 0.0)
        self.assertEqual(result["per_class"], {})

    def test_perfect_predictions(self):
        evaluator = EvaluationModule()
        evaluator.update({
            "scores": {"sperm_whale": 0.9, "bottlenose_dolphin": 0.1},
            "detected": True,
            "ground_truth_label": "sperm_whale",
        })
        evaluator.update({
            "scores": {"sperm_whale": 0.1, "bottlenose_dolphin": 0.9},
            "detected": True,
            "ground_truth_label": "bottlenose_dolphin",
        })
        result = evaluator.summary()
        self.assertEqual(result["total"], 2)
        self.assertAlmostEqual(result["accuracy"], 1.0)
        self.assertAlmostEqual(result["macro_f1"], 1.0)

    def test_wrong_prediction_and_no_detection(self):
        evaluator = EvaluationModule()
        evaluator.update({
            "scores": {"sperm_whale": 0.9, "bottlenose_dolphin": 0.1},
            "detected": True,
            "ground_truth_label": "bottlenose_dolphin",
        })
        evaluator.update({
            "scores": {"sperm_whale": 0.9, "bottlenose_dolphin": 0.1},
            "detected": False,
            "ground_truth_label": "sperm_whale",
        })
        result = evaluator.summary()
        self.assertEqual(result["total"], 2)
        self.assertEqual(result["accuracy"], 0.0)
        self.assertEqual(result["confusion_matrix"]["sperm_whale"]["NO_DETECTION"], 1)
        self.assertEqual(result["confusion_matrix"]["bottlenose_dolphin"]["sperm_whale"], 1)

    def test_rejects_invalid_scores(self):
        evaluator = EvaluationModule()
        with self.assertRaises(ValueError):
            evaluator.update({
                "scores": {"sperm_whale": 1.5},
                "detected": True,
                "ground_truth_label": "sperm_whale",
            })
        self.assertEqual(evaluator.summary()["total"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
