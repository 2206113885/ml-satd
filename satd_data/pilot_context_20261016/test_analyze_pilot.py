"""Small checks for evaluation policy, pairing, and human-review leakage."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

import analyze_pilot as ap


class EvaluationPolicyTests(unittest.TestCase):
    def test_parse_failure_is_error_even_if_fallback_matches_gold(self):
        parsed = ap.prediction({"predicted_label": "Data Conf", "parse_ok": False})
        self.assertEqual(parsed, ap.ERROR_LABEL)
        metrics, _ = ap.scores(["Data Conf", "MLKnow"], [parsed, "MLKnow"])
        self.assertEqual(metrics["accuracy"], .5)
        self.assertEqual(metrics["parse_failures"], 1)
        self.assertEqual(metrics["weighted_f1"], .5)

    def test_macro_denominators_are_explicit_and_unseen_prediction_is_fp(self):
        metrics, per_class = ap.scores(["Data Conf", "Data Conf", "MLKnow"],
                                       ["Data Conf", "MLKnow", "MLKnow"])
        self.assertAlmostEqual(metrics["weighted_precision"], 5 / 6)
        self.assertAlmostEqual(metrics["weighted_recall"], 2 / 3)
        self.assertAlmostEqual(metrics["weighted_f1"], 2 / 3)
        self.assertAlmostEqual(metrics["macro_f1"], 2 / 3)
        self.assertAlmostEqual(metrics["macro_f1_23"], (4 / 3) / 23)
        self.assertEqual(len(per_class), 23)

    def test_exact_mcnemar_and_paired_bootstrap(self):
        self.assertEqual(ap.exact_mcnemar(0, 0), 1.)
        self.assertEqual(ap.exact_mcnemar(5, 0), .0625)
        gold = ["Data Conf", "MLKnow"]
        preds = {"comment_only": ["MLKnow", "Data Conf"],
                 "window": gold, "function": gold}
        cis, differences = ap.bootstrap(gold, preds, 20, 9)
        self.assertEqual(cis["window"]["accuracy"], [1., 1.])
        self.assertEqual(differences["window"]["weighted_f1"], [1., 1.])

    def test_primary_pairing_and_blind_forms(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cohort = [dict(sample_id=str(i), gold_label="MLKnow", comment="TODO test",
                           window_context="window_code", function_context="function_code") for i in range(2)]
            predictions = [dict(sample_id=str(i), arm=a, predicted_label="MLKnow", parse_ok=True)
                           for i in range(2) for a in ap.ARMS if not (i == 1 and a == "function")]
            for name, rows in (("cohort", cohort), ("predictions", predictions)):
                (tmp / f"{name}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
            result = ap.analyze(tmp / "cohort.jsonl", tmp / "predictions.jsonl", tmp / "out", 20)
            self.assertEqual(result["common_paired_n"], 1)
            self.assertFalse(result["complete"])
            self.assertEqual(result["all_available_per_arm"]["window"]["n"], 2)
            with (tmp / "out/blind_review_form_A.csv").open(encoding="utf-8-sig") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(not r["code_context"] for r in rows))
            self.assertTrue(all(not r["assigned_labels"] for r in rows))
            self.assertFalse(set(rows[0]) & {"sample_id", "gold_label", "predicted_label"})
            with self.assertRaisesRegex(ValueError, "incomplete"):
                ap.analyze(tmp / "cohort.jsonl", tmp / "predictions.jsonl", tmp / "strict", 20,
                           require_complete=True)

    def test_reject_conflicting_gold(self):
        with self.assertRaises(ValueError):
            ap.scores(["not_a_label"], ["MLKnow"])


if __name__ == "__main__":
    unittest.main()
