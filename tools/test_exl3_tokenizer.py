"""EXL3 tokenizer export must use the same vocabulary contract as serve/server.py."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from strata_tokenizer import BYTE_TO_UNICODE, QWEN35_PATTERN, Tokenizer

SCRIPT = Path(__file__).resolve().parent / "exl3/tokenizer_to_pack.py"

# MiMo v2's own declared split (tokenizer.json pre_tokenizer), as the real repo ships it: qwen35's
# `[\p{L}\p{M}]+` letter run this is not, and the difference changes ids (a combining mark becomes its
# own punctuation piece here).
MIMO_PATTERN = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*"
                r"|\s*[\r\n]+|\s+(?!\S)|\s+")


class ExportTokenizer(unittest.TestCase):
    def test_server_vocabulary_contract_and_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hf = root / "hf"
            hf.mkdir()
            vocab = {token: byte for byte, token in BYTE_TO_UNICODE.items()}
            (hf / "config.json").write_text(json.dumps({"text_config": {"vocab_size": 260}}))
            (hf / "tokenizer.json").write_text(json.dumps({"model": {"vocab": vocab}, "added_tokens": [
                {"id": 256, "content": "<|im_end|>", "special": True}]}))
            (hf / "merges.txt").write_text("")
            p = subprocess.run([sys.executable, str(SCRIPT), "--hf", str(hf), "--out", str(root / "pack"),
                                "--check"], capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            exported = root / "pack/tokenizer"
            raw = json.loads((exported / "vocab.json").read_text(encoding="utf-8"))
            self.assertIsInstance(raw, dict)
            self.assertEqual(len(raw), 260)
            tokens = [None] * len(raw)
            for token, index in raw.items(): tokens[index] = token
            types = json.loads((exported / "token_type.json").read_text())
            cfg = json.loads((exported / "tokenizer.json").read_text())
            self.assertEqual(cfg["pre_pattern"], QWEN35_PATTERN)   # no model_type: the qwen path, unchanged
            tk = Tokenizer(tokens, [], types)
            for text in ["Hello!", "你好🌍", "def f(x):\n  return x + 1", "<|im_end|>"]:
                self.assertEqual(tk.decode(tk.encode(text, parse_special=True)), text)
            self.assertEqual(tk.encode("<|im_end|>", parse_special=True), [256])

    def test_mimo_v2_ships_its_own_pattern_normalizer_and_specials(self):
        """model_type mimo_v2 (notes/MIMO-V2-PORT.md): the pack ships the declared pre-tokenizer regex
        and NFC normalizer verbatim, pads to the config's vocab_size with the dflash mask row named,
        and carries eos/pad/bos/mask plus every added token above the base vocab."""
        import unicodedata
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hf = root / "hf"
            hf.mkdir()
            vocab = {token: byte for byte, token in BYTE_TO_UNICODE.items()}
            vocab.update({"in": 256, "Ġt": 257})     # the two merges' results, as a base vocab carries them
            added = [{"id": 258, "content": "<|endoftext|>", "special": True},
                     {"id": 259, "content": "<|im_start|>", "special": True},
                     {"id": 260, "content": "<|im_end|>", "special": True},
                     {"id": 261, "content": "<think>", "special": False}]
            (hf / "config.json").write_text(json.dumps({"model_type": "mimo_v2", "vocab_size": 300,
                                                        "eos_token_id": 260, "pad_token_id": 258,
                                                        "bos_token_id": None}))
            (hf / "tokenizer.json").write_text(json.dumps({
                "model": {"vocab": vocab},
                "added_tokens": added,
                "normalizer": {"type": "NFC"},
                "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
                    {"type": "Split", "pattern": {"Regex": MIMO_PATTERN}, "behavior": "Isolated",
                     "invert": False},
                    {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False,
                     "use_regex": False}]}}))
            (hf / "merges.txt").write_text("i n\nĠ t\n")
            decoder = {str(a["id"]): {"content": a["content"], "special": a["special"]} for a in added}
            (hf / "tokenizer_config.json").write_text(json.dumps(
                {"eos_token": "<|im_end|>", "pad_token": "<|endoftext|>", "bos_token": None,
                 "added_tokens_decoder": decoder, "chat_template": "<|im_start|>user\n{% raw x %}"}))
            p = subprocess.run([sys.executable, str(SCRIPT), "--hf", str(hf), "--out", str(root / "pack"),
                                "--check", "--mask-token-id", "270"], capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            exported = root / "pack/tokenizer"
            cfg = json.loads((exported / "tokenizer.json").read_text())
            self.assertEqual(cfg["pre"], "mimo_v2")
            self.assertEqual(cfg["pre_pattern"], MIMO_PATTERN)
            self.assertNotEqual(cfg["pre_pattern"], QWEN35_PATTERN)
            self.assertEqual(cfg["normalizer"], "NFC")
            self.assertEqual((cfg["eos_token_id"], cfg["pad_token_id"], cfg["bos_token_id"]), (260, 258, None))
            self.assertEqual((cfg["mask_token_id"], cfg["mask_token"]), (270, "<|mask|>"))
            self.assertEqual(cfg["added_special_tokens"], [dict(a, special=a["special"]) for a in added])
            self.assertEqual((exported / "chat_template.jinja").read_text(),
                             "<|im_start|>user\n{% raw x %}")     # no separate file: the embedded one
            raw = json.loads((exported / "vocab.json").read_text(encoding="utf-8"))
            self.assertEqual(len(raw), 300)                       # padded to the config's vocab_size
            tokens = [None] * len(raw)
            for token, index in raw.items(): tokens[index] = token
            types = json.loads((exported / "token_type.json").read_text())
            self.assertEqual(tokens[270], "<|mask|>")
            self.assertEqual(types[270], 5)                       # UNUSED: never matched, never emitted
            self.assertEqual(tokens[299], "<|reserved_299|>")
            self.assertEqual(tokens[261], "<think>")              # special=false -> USER_DEFINED, always matched
            self.assertEqual(types[261], 4)
            tk = Tokenizer(tokens, ["i n", "Ġ t"], types, pre_pattern=cfg["pre_pattern"],
                           normalizer=cfg["normalizer"])
            for text in ["Hello!", "你好🌍", "in it", "<|im_end|><think>"]:
                self.assertEqual(tk.decode(tk.encode(text, parse_special=True)), text, text)
            self.assertEqual(tk.encode("<|im_end|>", parse_special=True), [260])
            # NFC, as declared: the round trip is the composed form, exactly the HF tokenizer's own
            self.assertEqual(tk.decode(tk.encode("e\u0301")), unicodedata.normalize("NFC", "e\u0301"))


if __name__ == "__main__":
    unittest.main()
