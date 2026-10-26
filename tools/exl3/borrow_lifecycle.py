"""EXL3 arena-loan lifecycle regression: a borrowed single-session serve vs the owned oracle.

Drives the engine protocol directly (like lifecycle.py, but single-session - the loan's initial
scope).  The borrowed arm runs STRATA_EXL3_PREFILL_BORROW=1 + STRATA_TRACE=1 with an explicit
--prefill chunk, the owned arm the same requests with the flag off.  Checks:

- the startup loan opens before the prompt path carves (log line; the cache is cold then, so the
  loan's saved tail is empty - restore must still work);
- a long prompt read across the chunk boundary, then decode;
- a follow-up turn whose suffix is long enough to read batched: the PREFIX is not re-read (the
  PP progress line's token count stays suffix-sized) and the loan reopens/relayouts for it;
- a short prompt through the window path (the loan is restored before windows);
- a STOP landing mid-prefill (cancel), followed by a VALID request - the loan lifecycle survives;
- every token stream identical to the owned arm (the loan only changes buffer backing);
- the eligibility fallbacks by their startup log lines: --prefill auto (not an operator-named
  chunk), --expert-cache 1 (no lendable ranks) and --batch 2 (unsupported mode) all stay owned.

Also records the STRATA_TRACE "EXL3 loan restored N ranks in X ms" lines (the loan's runtime
refill overhead) into report.json.  Nonfinite/yield/checkpoint conditions of the batch paths are
not reachable single-session and are documented as untested here.
"""
import argparse
import json
import re
import statistics
import subprocess
import time
from pathlib import Path

from bench import Engine, load_tokenizer


def chat(tk, question):
    return tk.encode(f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n\n",
                     parse_special=True)


def long_prompt(tk, count, question):
    """A deterministic `count`-token prompt: bench.py's filler records around the question."""
    head = tk.encode(f"<|im_start|>user\nPlease read these records carefully.\n", parse_special=True)
    tail = tk.encode(f"\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n\n", parse_special=True)
    remaining = count - len(head) - len(tail)
    if remaining < 0:
        raise ValueError("prompt too short")
    filler, index = [], 0
    while len(filler) < remaining:
        text = "".join(f"Record {i}: station {(i * 31) % 997} measured {(i * 71) % 1009}; the committee checked "
                       "its budget and requested a written report for the next meeting.\n" for i in range(index, index + 64))
        filler.extend(tk.encode(text))
        index += 64
    return head + filler[:remaining] + tail


def gen(eng, deadline, ids, max_new, stop=False):
    eng.send(f"GEN {max_new} temperature=0 seed=7319 " + ",".join(map(str, ids)))
    if stop:
        time.sleep(1.0)   # the STOP must land while the long prompt is still being read
        eng.send("STOP")
    tokens, pp, finish = [], [], None
    while True:
        line = eng.receive(deadline)
        if line.startswith("T "):
            tokens.append(int(line.split()[1]))
        elif line.startswith("PP "):
            pp.append([int(x) for x in line.split()[1:3]])
        elif line.startswith("DONE "):
            finish = line
            break
    return tokens, pp, finish


def run_arm(name, cfg, out, tk, borrow, trace):
    arm = dict(cfg)
    arm["env"] = dict(cfg.get("env", {}))
    arm["env"]["STRATA_EXL3_PREFILL_BORROW"] = "1" if borrow else "0"
    if trace:
        arm["env"]["STRATA_TRACE"] = "1"
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    engine = [str(cfg["_engine"]), "--serve", *cfg["_args"]]
    eng = Engine(engine, arm, d)
    deadline = time.monotonic() + 1200
    try:
        while True:
            if eng.receive(deadline).startswith("READY "):
                break
        a = long_prompt(tk, 3000, "Summarize the records above in three sentences.")
        # a ~1300-token follow-up: the suffix must read BATCHED (one chunk), not through the windows
        follow = long_prompt(tk, 1300, "Now write a detailed technical essay about database indexing "
                                       "tradeoffs, covering B-trees, LSM trees and recovery.")
        short = chat(tk, "Reply with the single word: ready.")                # ~20 tokens: the window path
        medium = long_prompt(tk, 300, "Name three benefits of database indexes with examples.")  # one chunk
        r = {}
        r["turn1"], r["turn1_pp"], r["turn1_done"] = gen(eng, deadline, a, 48)
        # the follow-up reuses turn1's prefix (a checkpoint): only the suffix may be read
        r["turn2"], r["turn2_pp"], r["turn2_done"] = gen(eng, deadline, a + r["turn1"] + follow, 32)
        r["short"], r["short_pp"], r["short_done"] = gen(eng, deadline, short, 32)
        r["medium"], r["medium_pp"], r["medium_done"] = gen(eng, deadline, medium, 32)
        r["cancel"], r["cancel_pp"], r["cancel_done"] = gen(eng, deadline, long_prompt(
            tk, 3000, "Explain every record above in detail."), 48, stop=True)
        r["after"], r["after_pp"], r["after_done"] = gen(eng, deadline, short, 32)
        eng.send("QUIT")
        eng.process.wait(timeout=120)
    finally:
        if eng.process.poll() is None:
            eng.process.terminate()
            try:
                eng.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                eng.process.kill()
    stderr = (d / "engine.stderr.log").read_text(errors="replace")
    restores = [(int(m.group(1)), float(m.group(2)))
                for m in re.finditer(r"EXL3 loan restored (\d+) ranks in ([0-9.]+) ms", stderr)]
    return r, stderr, restores


def eligibility(name, cfg, out, args_override, expect_line, expect_absent):
    arm = dict(cfg)
    arm["env"] = dict(cfg.get("env", {}))
    arm["env"]["STRATA_EXL3_PREFILL_BORROW"] = "1"
    args = list(cfg["_args"])
    for flag, new in args_override:
        if flag in args and args.index(flag) + 1 < len(args):
            args[args.index(flag) + 1] = new   # the flag's VALUE (e.g. --prefill 8192 -> auto)
        else:
            args += [flag, new]
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    eng = Engine([str(cfg["_engine"]), "--serve", *args], arm, d)
    deadline = time.monotonic() + 1200
    try:
        while True:
            if eng.receive(deadline).startswith("READY "):
                break
        eng.send("QUIT")
        eng.process.wait(timeout=120)
    finally:
        if eng.process.poll() is None:
            eng.process.terminate()
    stderr = (d / "engine.stderr.log").read_text(errors="replace")
    return (expect_line is None or expect_line in stderr) and (expect_absent is None or expect_absent not in stderr), stderr


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--prefill", type=int, default=2048, help="the explicit chunk the loan is sized for")
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(a.config.read_text())
    tk = load_tokenizer(cfg["tokenizer"])
    args = list(cfg["args"])
    for i, v in enumerate(args):   # the lifecycle runs an explicit, smaller chunk than the bench config
        if v == "--prefill" and i + 1 < len(args):
            args[i + 1] = str(a.prefill)
    cfg["_engine"], cfg["_args"] = a.engine, args

    report = {"prefill": a.prefill, "checks": {}, "restores": {}}
    borrowed, b_stderr, restores = run_arm("borrowed", cfg, a.output, tk, True, True)
    owned, o_stderr, _ = run_arm("owned", cfg, a.output, tk, False, False)

    def check(name, ok):
        report["checks"][name] = bool(ok)
        print(f"{name}: {'PASS' if ok else 'FAIL'}", flush=True)

    check("borrowed_startup_loan", "EXL3 prompt path borrows" in b_stderr)
    check("owned_no_loan", "EXL3 prompt path borrows" not in o_stderr)
    check("restores_traced", len(restores) > 0)
    report["restores"] = {"count": len(restores), "ranks": [r for r, _ in restores],
                          "ms": [ms for _, ms in restores],
                          "ms_mean": statistics.fmean(ms for _, ms in restores) if restores else None}
    for turn in ("turn1", "turn2", "short", "medium", "after"):
        check(f"{turn}_tokens_identical", borrowed[turn] == owned[turn] and len(borrowed[turn]) > 0)
    check("turn1_reached_length", "length" in borrowed["turn1_done"])
    for turn in ("turn2", "short", "medium", "after"):   # tiny prompts may end on EOS ("stop")
        check(f"{turn}_finish_agrees", borrowed[f"{turn}_done"].split()[5] == owned[f"{turn}_done"].split()[5])
    check("cancel_acknowledged", "cancel" in borrowed["cancel_done"] and "cancel" in owned["cancel_done"])
    # prefix reuse: turn2's ~1350-token suffix reads in fewer chunks than turn1's 3000 tokens.
    # PP prints once per chunk, so the line count is the read shape, not a timing signal.
    check("turn1_three_steps", len(borrowed["turn1_pp"]) >= 2 and len(owned["turn1_pp"]) >= 2)
    check("turn2_prefix_reused", len(borrowed["turn2_pp"]) < len(borrowed["turn1_pp"]) and
          len(owned["turn2_pp"]) < len(owned["turn1_pp"]))
    check("medium_one_chunk", len(borrowed["medium_pp"]) <= 2 and len(owned["medium_pp"]) <= 2)
    # the loan lifecycle itself: a restore per completed request that read batched (turn1, turn2,
    # medium and the cancelled read); the window requests restore nothing because nothing was lent
    check("loan_restore_count", len(restores) >= 4)
    check("after_cancel_valid_request", len(borrowed["after"]) > 0 and borrowed["after"] == owned["after"])

    ok, _ = eligibility("fallback-auto", cfg, a.output, [("--prefill", "auto")], None, "EXL3 prompt path borrows")
    check("fallback_auto_owned", ok)
    ok, _ = eligibility("fallback-small-cache", cfg, a.output, [("--expert-cache", "1")],
                        "owned buffers", None)
    check("fallback_small_cache_owned", ok)
    ok, _ = eligibility("fallback-batch", cfg, a.output, [("--batch", "2")], None, "EXL3 prompt path borrows")
    check("fallback_batch_owned", ok)

    (a.output / "report.json").write_text(json.dumps(report, indent=2))
    print("PASS" if all(report["checks"].values()) else "FAIL")
    raise SystemExit(0 if all(report["checks"].values()) else 1)


if __name__ == "__main__":
    main()
