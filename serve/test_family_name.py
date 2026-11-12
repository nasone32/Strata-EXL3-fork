"""serve/test_family_name.py - the API's model name per model family (NEW-MODELS-PLAN.md T0): the config's
"model_name" wins; without one, the family's ("arch": qwen4exp, qwen35moe, mimo_v2), and Flash-Next's for a
config that names neither.  The same "arch" also carries the family's own stop-token ids and draft-head
advice (MiMo v2, notes/MIMO-V2-PORT.md): qwen4exp / qwen35moe and every config without an "arch" keep the
module defaults exactly as they were.

    python -m unittest serve.test_family_name
"""
from __future__ import annotations

import unittest

from serve.server import (DEFAULT_DRAFT_HINT, DEFAULT_EOS_IDS, DRAFT_HEAD_HINT, EOS_IDS, FAMILY_DRAFT_HINT,
                          FAMILY_EOS_IDS, FAMILY_MODEL_NAMES, apply_model_family, default_model_name,
                          family_draft_hint, family_eos_ids)


class FamilyName(unittest.TestCase):
    def test_config_name_wins(self):
        self.assertEqual(default_model_name({"model_name": "qwen3.8-flash-next-q2_0", "arch": "qwen35moe"}),
                         "qwen3.8-flash-next-q2_0")

    def test_per_family(self):
        self.assertEqual(default_model_name({"arch": "qwen35moe"}), "qwen3.6-35b-a3b")
        self.assertEqual(default_model_name({"arch": "qwen4exp"}), "qwen3.8-flash-next")
        self.assertEqual(default_model_name({"arch": "mimo_v2"}), "mimo-v2.6-flash")

    def test_old_configs(self):
        self.assertEqual(default_model_name({}), "qwen3.8-flash-next")
        self.assertEqual(default_model_name({"arch": "something-else"}), "qwen3.8-flash-next")
        self.assertEqual(set(FAMILY_MODEL_NAMES), {"qwen4exp", "qwen35moe", "mimo_v2"})

    def test_stop_ids_per_family(self):
        """MiMo's Qwen2-style vocabulary numbers <|im_end|> 151645 (Flash-Next: 248046); every other family,
        and a config without an "arch", keeps the shipped default set."""
        self.assertEqual(family_eos_ids({"arch": "mimo_v2"}), {151645})
        self.assertEqual(family_eos_ids({"arch": "qwen4exp"}), {248044, 248046})
        self.assertIs(family_eos_ids({"arch": "qwen35moe"}), DEFAULT_EOS_IDS)
        self.assertIs(family_eos_ids({}), DEFAULT_EOS_IDS)
        self.assertEqual(EOS_IDS, DEFAULT_EOS_IDS)     # as imported, before apply_model_family runs
        self.assertEqual(FAMILY_EOS_IDS["mimo_v2"], {151645})

    def test_draft_hint_per_family(self):
        """MiMo's MTP heads share the model's vocabulary: no --draft-vocab advice there."""
        self.assertEqual(family_draft_hint({"arch": "mimo_v2"}), FAMILY_DRAFT_HINT["mimo_v2"])
        hint = family_draft_hint({"arch": "mimo_v2"})
        self.assertNotIn("--draft-vocab en", hint)
        self.assertNotIn("--draft-vocab cyrillic", hint)
        self.assertIs(family_draft_hint({"arch": "qwen4exp"}), DRAFT_HEAD_HINT)
        self.assertIs(family_draft_hint({}), DEFAULT_DRAFT_HINT)

    def test_apply_model_family_rebinds_the_constants(self):
        """main() points the module's EOS_IDS / DRAFT_HEAD_HINT at the family's values before the engine
        starts; a family the tables do not name gets the shipped defaults back."""
        import serve.server as server
        was_hint = server.DRAFT_HEAD_HINT
        try:
            apply_model_family({"arch": "mimo_v2"})
            self.assertEqual(server.EOS_IDS, {151645})
            self.assertEqual(server.DRAFT_HEAD_HINT, FAMILY_DRAFT_HINT["mimo_v2"])
            apply_model_family({"arch": "qwen35moe"})
            self.assertEqual(server.EOS_IDS, DEFAULT_EOS_IDS)
            self.assertIs(server.DRAFT_HEAD_HINT, was_hint)
        finally:
            apply_model_family({})                 # the module default, for the tests that follow


if __name__ == "__main__":
    unittest.main()
