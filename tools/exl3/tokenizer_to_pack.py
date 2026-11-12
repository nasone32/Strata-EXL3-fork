r"""Convert a HuggingFace tokenizer directory (as shipped with the EXL3 model) into the pack's
`tokenizer/` layout that tools/strata_tokenizer.py and serve/ read.

    python3 tools/exl3/tokenizer_to_pack.py --hf <hf_model_dir> --out <pack_dir>

The HF model has `tokenizer.json` (base vocab + added tokens), `merges.txt`, `tokenizer_config.json`,
`chat_template.jinja`, and `config.json` (embedding rows).  The pack wants `vocab.json` (token -> id
map), `merges.txt`, `token_type.json` (ggml token types), `tokenizer.json` (config), and the template.
The family is `config.json`'s `model_type`:

* Qwen3.8-Flash-Next (none / anything else): the BPE is identical to the GGUF's (247,587 merges,
  `qwen35` pretokenizer), so no new tokenizer is needed; this just reshapes the files.  See
  docs/EXL3.md and the research note in the PR.
* `mimo_v2` (MiMo-V2.6-Flash, notes/MIMO-V2-PORT.md): a Qwen2-style BPE (151,643 base tokens,
  151,387 merges) whose tokenizer.json DECLARES its own pre-tokenizer regex - the qwen35 one does not
  fit it: qwen35 keeps a combining mark inside the letter run (`[\p{L}\p{M}]+`), MiMo's regex has
  plain `\p{L}+` and no `\p{M}` in the punctuation class, so the same text splits into different
  pieces and BPE gives different ids.  The pack therefore ships the declared regex verbatim
  (`pre_pattern`), the declared NFC normalizer (`normalizer`; decode(encode(s)) is NFC(s), exactly as
  the HF tokenizer's own round trip is), pads to the config's vocab_size 152576, and names the
  dflash drafter's mask token (151675, one id past the last added token - the bf16 draft config's
  `dflash_config.mask_token_id`; never text, so it stays type UNUSED like the padding around it).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from strata_tokenizer import QWEN35_PATTERN          # noqa: E402

# ggml token types (as the pack's token_type.json uses them)
NORMAL, UNKNOWN, CONTROL, USER_DEFINED, UNUSED = 1, 2, 3, 4, 5

# The dflash drafter's mask row (notes/MIMO-V2-PORT.md section 1: `dflash-bf16/config.json`
# `dflash_config.mask_token_id` 151675, one id past the last added token 151674 `<|mimo_audio_end|>`).
# The tokenizer files never name it - it is never text - so the family constant is the source.
MIMO_MASK_ID = 151675
MIMO_MASK_TOKEN = "<|mask|>"


def declared_pattern(tj: dict) -> str:
    """The pre-tokenizer regex a HF `tokenizer.json` declares: a Sequence whose first step is a Split
    with a `Regex` pattern (the shape Qwen2 / Qwen3 / MiMo tokenizer.json files use), or that Split
    alone.  The ByteLevel step after it is the byte mapping every pack already applies.  Any other
    shape is an error, not a fallback to some lookalike pattern: the split decides the ids."""
    pre = tj.get("pre_tokenizer") or {}
    steps = pre.get("pretokenizers") if pre.get("type") == "Sequence" else [pre]
    for step in steps or []:
        pat = (step or {}).get("pattern") if isinstance(step, dict) else None
        if isinstance(step, dict) and step.get("type") == "Split" and isinstance(pat, dict):
            rx = pat.get("Regex")
            if rx:
                return rx
    raise ValueError("tokenizer.json's pre_tokenizer declares no Split/Regex pattern (%s...); this tool "
                     "ships what the file declares, and here there is nothing to ship" % json.dumps(pre)[:120])


def declared_normalizer(tj: dict) -> str | None:
    """The `normalizer` a HF `tokenizer.json` declares, as a unicodedata form name ("NFC").  None for
    the null normalizer of the Qwen EXL3 exports.  Anything else (NMT, lowercase, a sequence, ...) is
    a family this pack format does not carry yet: said loudly here, not discovered as bad ids."""
    n = tj.get("normalizer")
    if not n:
        return None
    form = n.get("type") if isinstance(n, dict) else None
    if form in ("NFC", "NFD", "NFKC", "NFKD"):
        return form
    raise ValueError("tokenizer.json declares a normalizer the pack does not carry: %s" % json.dumps(n)[:120])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True, help="directory with tokenizer.json / merges.txt / config.json")
    ap.add_argument("--out", required=True, help="pack directory; tokenizer/ is written under it")
    ap.add_argument("--check", action="store_true", help="round-trip a corpus through the pack tokenizer")
    ap.add_argument("--mask-token-id", type=int, default=MIMO_MASK_ID,
                    help="the dflash drafter's mask row (default %d, the family's; a synthetic fixture can "
                         "lower it)" % MIMO_MASK_ID)
    args = ap.parse_args()

    hf = pathlib.Path(args.hf)
    tj = json.loads((hf / "tokenizer.json").read_text(encoding="utf-8"))
    base = tj["model"]["vocab"]                        # token -> id
    added = tj.get("added_tokens", [])

    cfg = json.loads((hf / "config.json").read_text(encoding="utf-8"))
    vocab_size = (cfg.get("text_config", {}) or cfg).get("vocab_size")
    mimo = cfg.get("model_type") == "mimo_v2"
    mask_id = args.mask_token_id
    if mimo and not isinstance(vocab_size, int):
        raise SystemExit("mimo_v2 config.json has no vocab_size: the pack pads up to it")

    # tokenizer_config.json: the added-token list of record (every special above the base vocab) and,
    # for MiMo, the eos/pad/bos spellings cross-checked against the config's ids and the template
    # fallback when the repo ships no separate chat_template.jinja.
    tcfg = json.loads((hf / "tokenizer_config.json").read_text(encoding="utf-8")) \
        if (hf / "tokenizer_config.json").exists() else {}
    decoder = {int(i): v for i, v in (tcfg.get("added_tokens_decoder") or {}).items()}
    if mimo and decoder:
        by_id = {a["id"]: (a["content"], a.get("special", False)) for a in added}
        if {i: v["content"] for i, v in decoder.items()} != {i: c for i, (c, _) in by_id.items()}:
            raise SystemExit("tokenizer_config.json's added_tokens_decoder disagrees with tokenizer.json's "
                             "added_tokens: which one is the tokenizer?")

    max_id = max([max(base.values())] + [a["id"] for a in added])
    size = max(vocab_size or 0, max_id + 1)

    tokens = [""] * size
    types = [UNUSED] * size
    for tok, i in base.items():
        tokens[i] = tok
        types[i] = NORMAL
    special_ids: dict[str, int] = {}
    for a in added:
        i, tok, sp = a["id"], a["content"], a.get("special", False)
        tokens[i] = tok
        types[i] = CONTROL if sp else USER_DEFINED
        if sp:
            special_ids[tok] = i

    merges = [ln for ln in (hf / "merges.txt").read_text(encoding="utf-8").splitlines() if ln.strip()]

    # The embedding has `vocab_size` rows but the tokenizer defines fewer; the tail is untrained
    # padding.  The engine's Tokenizer requires unique tokens, so name them uniquely (never emitted,
    # never matched: type UNUSED, not in the special/always alternations).  MiMo's mask row is padding
    # to the target model too, but the dflash drafter addresses it by id, so it gets its own name.
    for i in range(size):
        if not tokens[i]:
            tokens[i] = MIMO_MASK_TOKEN if (mimo and i == mask_id) else "<|reserved_%d|>" % i

    out = pathlib.Path(args.out) / "tokenizer"
    out.mkdir(parents=True, exist_ok=True)
    (out / "vocab.json").write_text(json.dumps({token: i for i, token in enumerate(tokens)}, ensure_ascii=False), encoding="utf-8")
    (out / "merges.txt").write_text("\n".join(merges), encoding="utf-8")
    (out / "token_type.json").write_text(json.dumps(types), encoding="utf-8")

    pattern, normalizer = QWEN35_PATTERN, None
    pack_cfg = {
        "model": "gpt2",
        "pre": "qwen35",
        "vocab_size": size,
        "n_merges": len(merges),
        "special_ids": special_ids,
        "add_bos_token": False,
        "pre_pattern": pattern,
        "pre_pattern_source": "HF tokenizer.json (same qwen35 pattern as the GGUF)",
    }
    if mimo:
        pattern = declared_pattern(tj)
        normalizer = declared_normalizer(tj)
        # the config's ids, said with the strings the tokenizer really has there (eos <|im_end|>
        # 151645, pad <|endoftext|> 151643, no bos: MiMo numbers the Qwen specials a Qwen2 vocab does)
        eos, pad, bos = cfg.get("eos_token_id"), cfg.get("pad_token_id"), cfg.get("bos_token_id")
        for i, name in ((eos, "eos_token"), (pad, "pad_token")):
            if isinstance(i, int) and tcfg.get(name) and tokens[i] != tcfg[name]:
                raise SystemExit("config.json's %s id %d is %r in the vocabulary, tokenizer_config.json says %r"
                                 % (name, i, tokens[i], tcfg[name]))
        if not isinstance(eos, int) or not isinstance(pad, int):
            raise SystemExit("mimo_v2 config.json needs eos_token_id and pad_token_id")
        if not max_id < mask_id < size:
            raise SystemExit("the dflash mask id %d must sit between the last declared token %d and the "
                             "vocab_size %d; the family constant is stale" % (mask_id, max_id, size))
        pack_cfg.update({
            "pre": "mimo_v2",
            "pre_pattern": pattern,
            "pre_pattern_source": "HF tokenizer.json pre_tokenizer Split/Regex, verbatim (Qwen2-style: no \\p{M})",
            "normalizer": normalizer,
            "eos_token_id": eos,
            "pad_token_id": pad,
            "bos_token_id": bos,
            "mask_token_id": mask_id,
            "mask_token": tokens[mask_id],
            # every token above the base vocab, from tokenizer_config.json's added_tokens_decoder
            "added_special_tokens": [{"id": i, "content": tokens[i], "special": types[i] == CONTROL}
                                     for i in sorted(decoder)] if decoder else
                                    [{"id": a["id"], "content": a["content"], "special": a.get("special", False)}
                                     for a in added],
        })
    (out / "tokenizer.json").write_text(json.dumps(pack_cfg, ensure_ascii=False, indent=1), encoding="utf-8")

    # the model's own chat template, as a separate file when the repo ships one (MiMo does; the same
    # text embedded in tokenizer_config.json differs only by a blank line), else that embedded one
    tpl = hf / "chat_template.jinja"
    text = tpl.read_text(encoding="utf-8") if tpl.exists() else tcfg.get("chat_template")
    if text:
        (out / "chat_template.jinja").write_text(text, encoding="utf-8", newline="\n")

    if mimo:
        print("wrote %s: MiMo v2 vocab %d (base %d + %d added + %d padding), merges %d, specials %d, "
              "eos %d, pad %d, mask %d (%s), normalizer %s"
              % (out, size, len(base), len(added), size - max_id - 1, len(merges), len(special_ids),
                 eos, pad, mask_id, MIMO_MASK_TOKEN, normalizer or "none"))
    else:
        print("wrote %s: vocab %d (filled %d), merges %d, added %d, specials %d"
              % (out, size, sum(1 for t in tokens if t), len(merges), len(added), len(special_ids)))

    if args.check:
        from strata_tokenizer import Tokenizer
        tk = Tokenizer(tokens, merges, types, pack_cfg["pre"], special_ids,
                       pre_pattern=pattern, normalizer=normalizer)
        corpus = ["", "Hello, world!", "  leading and trailing  ", "a\n\n\nb",
                  "def f(x):\n\treturn x  # comment\n", "\u4f60\u597d\uff0c\u4e16\u754c", "\u0645\u0631\u062d\u0628\u0627",
                  "\U0001f600\U0001f680\U0001f1fa\U0001f1f8", "e\u0301\u0301 combining", "\x00\x01\x7f control",
                  "\u00a0non-breaking\u00a0space", "1234567890", "MixedCASE_and-dashes", "\r\n\r\n", "\u2028\u2029",
                  "x" * 5000]
        bad = 0
        for s in corpus:
            want = s
            if normalizer:
                import unicodedata
                want = unicodedata.normalize(normalizer, s)
            if tk.decode(tk.encode(s)) != want:
                bad += 1
                print("  ROUND-TRIP FAIL: %r" % s[:60])
        print("round-trip: %d/%d ok%s" % (len(corpus) - bad, len(corpus),
                                          " (against the %s of the input)" % normalizer if normalizer else ""))
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
