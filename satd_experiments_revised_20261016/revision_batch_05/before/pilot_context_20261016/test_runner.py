"""Model-free runner checks; importing run_pilot does not load MLX or weights."""
import ast
import re
import unittest
from pathlib import Path

import run_pilot as rp


class RecordingTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return [1] + list(messages[0]["content"].encode()) + [2]


class RunnerHelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = rp.load_reference()

    def test_frozen_prompt_and_parser_match_source_without_loading_model(self):
        source = Path(__file__).parent.parent / "basic_zero_shot/Qwen3-8B_zero_shot_re.py"
        names = {"LABEL_ABBR", "ABBR2FULL", "LABELS", "ABBR_SET", "ABBR_LOWER",
                 "PROMPT_HEAD", "PROMPT_TAIL"}
        nodes = []
        for node in ast.parse(source.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets):
                nodes.append(node)
            elif isinstance(node, ast.FunctionDef) and node.name in ("build_prompt", "simple_extract"):
                nodes.append(node)
        namespace = {"re": re}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
        for comment in ("TODO json?", 'TODO: line one\nline two "quoted"', "TODO test"):
            self.assertEqual(namespace["build_prompt"](comment), self.reference.build_prompt(comment))
        for response in ("Data Conf", "MLKnow", "data conf", "**Data Conf**", "MLKnow\nextra", "unknown"):
            self.assertEqual(namespace["simple_extract"](response), self.reference.simple_extract(response))

    def test_three_arms_share_target_and_have_no_gold_leakage(self):
        row = dict(sample_id="test", comment="TODO json?", gold_label="DS Conf",
                   window_context="WINDOW_ONLY", function_context="FUNCTION_ONLY")
        reference = self.reference
        comment = rp.prompt_for(row, "comment_only", reference)
        self.assertEqual(comment, reference.build_prompt(row["comment"]))
        self.assertNotIn("WINDOW_ONLY", comment)
        self.assertNotIn("FUNCTION_ONLY", comment)
        self.assertIn("WINDOW_ONLY", rp.prompt_for(row, "window", reference))
        self.assertNotIn("FUNCTION_ONLY", rp.prompt_for(row, "window", reference))
        self.assertIn("FUNCTION_ONLY", rp.prompt_for(row, "function", reference))
        alternate = dict(row, gold_label="MLKnow")
        for arm in rp.ARMS:
            self.assertEqual(rp.prompt_for(row, arm, reference), rp.prompt_for(alternate, arm, reference))

    def test_seed_and_arm_order_are_reproducible_and_paired(self):
        rows = [dict(sample_id="test", comment="TODO json?", window_context="window", function_context="function")]
        tokenizer = RecordingTokenizer()
        first = rp.build_jobs(rows, tokenizer, self.reference, 17)
        second = rp.build_jobs(rows, RecordingTokenizer(), self.reference, 17)
        self.assertEqual([j["arm"] for j in first], [j["arm"] for j in second])
        self.assertEqual(set(j["arm"] for j in first), set(rp.ARMS))
        self.assertEqual(len(set(j["seed"] for j in first)), 1)
        self.assertNotEqual(rp.row_seed("test", 17), rp.row_seed("other", 17))
        for _, kwargs in tokenizer.calls:
            self.assertTrue(kwargs["tokenize"])
            self.assertFalse(kwargs["enable_thinking"])
            self.assertTrue(kwargs["add_generation_prompt"])

    def test_common_prefix_leaves_suffix_and_never_reuses_differing_tokens(self):
        self.assertEqual(rp.common_prefix([[1, 2, 3], [1, 2, 4]]), [1, 2])
        self.assertEqual(rp.common_prefix([[1, 2], [1, 2]]), [1])
        self.assertEqual(rp.common_prefix([[1, 2], [3, 4]]), [])
        sequences = [[1, 2, 3], [1, 2], [1, 2, 4, 5]]
        prefix = rp.common_prefix(sequences)
        for ids in sequences:
            self.assertTrue(ids[len(prefix):])
            self.assertEqual(ids[:len(prefix)], prefix)

    def test_unfinished_reasoning_and_invalid_labels_are_errors(self):
        self.assertEqual(rp.parse_response("reasoning MLKnow", self.reference, thinking=True),
                         (None, False, "reasoning MLKnow", "thinking_not_closed"))
        self.assertEqual(rp.parse_response("<think>reasoning</think>MLKnow", self.reference),
                         ("MLKnow", True, "MLKnow", None))
        self.assertFalse(rp.parse_response("<think>reasoning</think>invalid", self.reference)[1])
        self.assertEqual(rp.parse_response("<think>reasoning</think>Data Conf", self.reference)[0], "Data Conf")
        self.assertEqual(rp.parse_response("Data Conf", self.reference, thinking=False)[0], "Data Conf")


if __name__ == "__main__":
    unittest.main()
