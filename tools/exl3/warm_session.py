"""Two sequential GENs in one engine session: the first warms the EXL3 expert cache
(page cache + VRAM slots), the second measures warm decode. Prints DONE lines and
keeps the engine log (which carries the STRATA_EXL3_TIME counters at shutdown).

Usage: warm_session.py --engine PATH --config config.json --prompt-tokens N --max-new M --output DIR
The caller exports LD_LIBRARY_PATH / HIP_VISIBLE_DEVICES / STRATA_EXL3_TIME as usual.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench import Engine, load_tokenizer, prompt   # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--essay", action="store_true",
                    help="long-form generation instead of the needle answer, for TG runs that do not stop early")
    ap.add_argument("--turns", type=int, default=2,
                    help="number of GENs in one session; later turns repeat the same prompt, so the "
                         "expert cache and the trajectory stabilise (all-resident decode probing)")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--timeout", type=int, default=1800)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(a.config.read_text(encoding="utf-8"))
    tk = load_tokenizer(cfg["tokenizer"])
    ids, keys = prompt(tk, a.prompt_tokens, 0, not a.essay)
    (a.output / "prompt.json").write_text(json.dumps({"ids": ids, "keys": keys}), encoding="utf-8")
    command = [str(a.engine.resolve()), "--serve", *cfg["args"]]
    eng = Engine(command, cfg, a.output)
    deadline = 1e9   # time.monotonic()-relative, effectively "no deadline"
    line = eng.receive(deadline)
    while not line.startswith("READY "):
        line = eng.receive(deadline)
    report = {"command": command, "info": line}
    for turn in range(a.turns):
        eng.send(f"GEN {a.max_new} temperature=0 seed=7319 " + ",".join(map(str, ids)))
        out, done = [], None
        while True:
            line = eng.receive(deadline)
            if line.startswith("T "):
                out.append(int(line.split()[1]))
            elif line.startswith("DONE "):
                done = line
                break
        text = tk.decode(out)
        found = [key in text for key in keys]
        report[f"turn{turn}"] = {"done": done, "keys_found": found, "tokens": len(out)}
        print(done, flush=True)
    eng.close()
    (a.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
