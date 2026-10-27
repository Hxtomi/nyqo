import unittest

from nyqo.protocol import (
    NEW_POSITIONS, parse_answer, validate_budget, validate_config, validate_messages,
)


class ProtocolTests(unittest.TestCase):
    def test_native_message_shape_and_final_user_are_enforced(self):
        valid = [{"role": "system", "content": "Be concise."},
                 {"role": "user", "content": "Hello"}]
        self.assertEqual(validate_messages(valid), valid)
        invalid = [
            [], [{"role": "tool", "content": "unsafe tool result"}],
            [{"role": "assistant", "content": "Done"}],
            [{"role": "user", "content": "Hello", "reasoning": "extra"}],
            [{"role": "user", "content": ""}],
        ]
        for messages in invalid:
            with self.subTest(messages=messages), self.assertRaises(ValueError):
                validate_messages(messages)

    def test_context_boundary_is_exact_without_truncation(self):
        validate_budget(1280, 768, 2048)
        with self.assertRaisesRegex(ValueError, "no truncation"):
            validate_budget(1281, 768, 2048)
        for count in (0, -1, True, 1.5):
            with self.subTest(count=count), self.assertRaises(ValueError):
                validate_budget(1, count, 2048)

    def test_wrong_model_or_expansion_is_rejected(self):
        valid = {"model_type": "lfm2", "num_hidden_layers": 24,
                 "nyqo_expansion": {"new_positions": NEW_POSITIONS}}
        validate_config(valid)
        for data in ({}, {**valid, "num_hidden_layers": 16},
                     {**valid, "nyqo_expansion": {"new_positions": []}}):
            with self.subTest(config=data), self.assertRaises(ValueError):
                validate_config(data)

    def test_only_leading_structural_reasoning_is_removed(self):
        result = parse_answer("<think>work</think>42<|im_end|>\nignored")
        self.assertEqual(result, {"final": "42", "reason": "work",
                                  "status": "reason_final"})
        literal = 'The literal tag "<think>" is part of the example.'
        self.assertEqual(parse_answer(literal)["final"], literal)

    def test_incomplete_nested_and_repeated_blocks_are_not_final_answers(self):
        cases = {
            "<think>unfinished": "unclosed_reason",
            "<think>a<think>b</think>answer": "nested_reason",
            "<think>a</think><think>b</think>answer": "repeated_reason",
            "<think>a</think>": "missing_final",
            "": "missing_final",
        }
        for raw, status in cases.items():
            with self.subTest(raw=raw):
                result = parse_answer(raw)
                self.assertEqual(result["status"], status)
                self.assertEqual(result["final"], "")


if __name__ == "__main__":
    unittest.main()
