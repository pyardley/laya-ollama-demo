"""Checks for the cost rules. Does not load Laya."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    PUBLISHED_SEARCH,
    load_cost_matrix,
    load_messages,
    same_messages,
)
from rules import (  # noqa: E402
    THRESHOLD_OFF,
    fit_repair_thresholds,
    loo_repair,
    predict_mapped_majority,
    predict_repair,
    predict_repeat_majority,
    predict_specialist_route,
)


def message(mid, label):
    return {"id": mid, "label": label, "text": label, "text_sha256": mid}


def answer(choice, confidence=0.6, probabilities=None):
    if probabilities is None:
        probabilities = {choice: confidence}
    return {
        "choice": choice,
        "answer_confidence": confidence,
        "probabilities": probabilities,
    }


class PublishedBaselineTest(unittest.TestCase):
    def test_published_costs(self):
        cost = load_cost_matrix()
        data = json.loads(PUBLISHED_SEARCH.read_text(encoding="utf-8"))
        best = data["best"]["messages"]
        original = data["baseline"]["messages"]
        best_cost = sum(cost[row["label"]][row["choice"]] for row in best)
        original_cost = sum(cost[row["label"]][row["choice"]] for row in original)
        self.assertEqual(best_cost, 19)
        self.assertEqual(sum(row["correct"] for row in best), 41)
        self.assertEqual(original_cost, 35)
        self.assertEqual(sum(row["correct"] for row in original), 19)


class MessageLoaderTest(unittest.TestCase):
    def test_csv_round_trip_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "messages.csv"
            path.write_text(
                "id,label,text\n"
                'm01,auto_reply,"How do I reset my password? I can\'t remember it."\n'
                "m02,ignore,asdf jjj\n",
                encoding="utf-8",
            )
            rows = load_messages(path)
        self.assertEqual(rows[0]["label"], "auto_reply")
        self.assertEqual(rows[1]["id"], "m02")
        self.assertTrue(same_messages(rows[:1], rows[:1]))

    def test_rejects_unknown_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "messages.csv"
            path.write_text("id,label,text\nx,spam,hello\n", encoding="utf-8")
            with self.assertRaises(SystemExit):
                load_messages(path)


class RuleTest(unittest.TestCase):
    def test_repeat_tie_uses_first_when_equal(self):
        row = {"demo": answer("ignore"), "demo_repeat": answer("ignore")}
        self.assertEqual(predict_repeat_majority(row), "ignore")

    def test_specialist_outsider_is_ignore_without_demo(self):
        row = {
            "demo": answer("auto_reply"),
            "ownership": answer("outsider", 0.4),
            "need": answer("steps", 0.4),
        }
        self.assertEqual(predict_specialist_route(row), "ignore")

    def test_repair_does_not_touch_escalation(self):
        row = {
            "demo": answer("escalate_to_human"),
            "ownership": answer("outsider", 0.99),
            "need": answer("none", 0.99),
        }
        self.assertEqual(predict_repair(row, 0.0, 0.0), "escalate_to_human")

    def test_repair_flips_ignore_only_when_both_specialists_agree(self):
        row = {
            "demo": answer("ignore"),
            "ownership": answer("customer", 0.8),
            "need": answer("staff", 0.7),
        }
        self.assertEqual(predict_repair(row, 0.5, 0.5), "escalate_to_human")
        self.assertEqual(predict_repair(row, 0.9, 0.5), "ignore")

    def test_repair_off_threshold_matches_demo(self):
        row = {
            "demo": answer("auto_reply"),
            "ownership": answer("outsider", 0.99),
            "need": answer("none", 0.99),
        }
        self.assertEqual(predict_repair(row, THRESHOLD_OFF, THRESHOLD_OFF), "auto_reply")

    def test_majority_keeps_demo_on_a_tie(self):
        row = {
            "demo": answer("ignore"),
            "ownership": answer("customer"),
            "need": answer("staff"),
        }
        # Votes: ignore (demo) and escalate (need). Tie includes the demo.
        self.assertEqual(predict_mapped_majority(row), "ignore")

    def test_majority_flips_when_both_specialists_agree(self):
        row = {
            "demo": answer("auto_reply"),
            "ownership": answer("outsider"),
            "need": answer("none"),
        }
        self.assertEqual(predict_mapped_majority(row), "ignore")

    def test_fit_can_disable_the_rule(self):
        cost = load_cost_matrix()
        # Firing would escalate a true ignore (cost 1) and the demo is already correct.
        rows = [{
            "demo": answer("ignore"),
            "ownership": answer("customer", 0.9),
            "need": answer("staff", 0.9),
        }]
        labels = ["ignore"]
        fit = fit_repair_thresholds(rows, labels, cost)
        self.assertEqual(fit["cost"], 0)
        self.assertGreaterEqual(fit["t_own"], 0.9)

    def test_loo_does_not_use_the_held_out_label(self):
        cost = load_cost_matrix()
        # Message 0 is a true ignore that looks like a repair candidate.
        # The other four are true escalations the repair would correctly save,
        # so the fit on those four wants to fire. Leave-one-out on message 0
        # still fires, and the cost is the escalate-a-non-ticket cost of 1.
        rows = []
        labels = []
        rows.append({
            "demo": answer("ignore"),
            "ownership": answer("customer", 0.8),
            "need": answer("staff", 0.8),
        })
        labels.append("ignore")
        for _ in range(4):
            rows.append({
                "demo": answer("ignore"),
                "ownership": answer("customer", 0.8),
                "need": answer("staff", 0.8),
            })
            labels.append("escalate_to_human")
        predictions, _chosen = loo_repair(rows, labels, cost)
        self.assertEqual(predictions[0], "escalate_to_human")
        self.assertEqual(cost["ignore"]["escalate_to_human"], 1)


if __name__ == "__main__":
    unittest.main()
