"""EXL3 static peer tier lifecycle smoke: a peer arm vs the no-peer oracle.

Drives the engine protocol directly (like borrow_lifecycle.py): short, medium and long prompts
(the prefill transition), a STOP landing mid-request (cancel) followed by a VALID request, and
a follow-up that reuses the prefix.  The peer arm runs STRATA_EXL3_PEER=1 with --peer-device 1
and --peer-slots; the oracle runs the same requests with the tier off.  Checks: every token
stream identical, no peer-fallback line in the peer arm's log, and the tier's startup line
present.  Usage: peer_lifecycle.py --engine PATH --config config.json --out DIR [--slots N].
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench import Engine, load_tokenizer, prompt  # noqa: E402


def gen(eng, deadline, ids, max_new, stop=False):
    eng.send(f"GEN {max_new} temperature=0 seed=7319 " + ",".join(map(str, ids)))
    if stop:
        time.sleep(1.0)
        eng.send("STOP")
    tokens, finish = [], None
    while True:
        line = eng.receive(deadline)
        if line.startswith("T "):
            tokens.append(int(line.split()[1]))
        elif line.startswith("DONE "):
            finish = line
            break
    return tokens, finish


def run_arm(name, cfg, out, tk, peer, slots, fault=None):
    arm = dict(cfg)
    arm["env"] = dict(cfg.get("env", {}))
    args = list(cfg["args"])
    if peer:
        arm["env"]["STRATA_EXL3_PEER"] = "1"
        arm["env"]["HIP_VISIBLE_DEVICES"] = "0,1"   # the peer device must be visible
        args += ["--peer-device", "1", "--peer-slots", str(slots)]
        if fault is not None:
            arm["env"]["STRATA_EXL3_PEER_FAULT_LAUNCH"] = str(fault)   # injected LAUNCH failures
    else:
        arm["env"].pop("STRATA_EXL3_PEER", None)
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    eng = Engine([str(cfg["_engine"]), "--serve", *args], arm, d)
    deadline = time.monotonic() + 1800
    try:
        while True:
            if eng.receive(deadline).startswith("READY "):
                break
        r = {}
        short, _ = prompt(tk, 512, 0, False)
        long_ids, _ = prompt(tk, 2048, 0, False)
        chunk_ids, _ = prompt(tk, 10000, 0, False)   # CROSSES the 8192-token chunk boundary
        r["short"], r["short_done"] = gen(eng, deadline, short, 48)
        r["medium"], r["medium_done"] = gen(eng, deadline, long_ids, 32)
        r["long"], r["long_done"] = gen(eng, deadline, long_ids, 32)   # prefix-reusing follow-up
        r["chunk8k"], r["chunk8k_done"] = gen(eng, deadline, chunk_ids, 32)   # multi-chunk read
        r["cancel"], r["cancel_done"] = gen(eng, deadline, long_ids, 64, stop=True)
        r["after"], r["after_done"] = gen(eng, deadline, short, 32)
        eng.send("QUIT")
        eng.process.wait(timeout=180)
    finally:
        if eng.process.poll() is None:
            eng.process.terminate()
            try:
                eng.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                eng.process.kill()
    return r, (d / "engine.stderr.log").read_text(errors="replace")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--slots", type=int, default=256)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(a.config.read_text(encoding="utf-8"))
    cfg["_engine"] = str(a.engine.resolve())
    tk = load_tokenizer(cfg["tokenizer"])
    oracle, olog = run_arm("oracle", cfg, a.out, tk, False, a.slots)
    peer, plog = run_arm("peer", cfg, a.out, tk, True, a.slots)
    checks = {}
    # every non-cancel request must produce EXACTLY its requested count with finish "length";
    # the cancel must finish "cancel" - count equality alone is not enough
    for key, n in (("short", 48), ("medium", 32), ("long", 32), ("chunk8k", 32), ("after", 32)):
        checks[f"{key}_identical"] = oracle[key] == peer[key] and len(oracle[key]) == n
        checks[f"{key}_length"] = "length" in oracle[f"{key}_done"] and "length" in peer[f"{key}_done"]
    checks["cancel_finished"] = "cancel" in oracle["cancel_done"] and "cancel" in peer["cancel_done"]
    checks["peer_tier_started"] = "EXL3 peer tier on device 1" in plog
    checks["no_peer_fallback"] = "full primary fallback" not in plog
    checks["oracle_has_no_tier"] = "EXL3 peer tier on device" not in olog
    checks["no_error_latch"] = all(x not in plog and x not in olog
                                  for x in ("ERR ", "failed to", "stopping the engine", "stall report"))
    # real peer work, not just a started tier: the shutdown summary must show computed rows and
    # the mixed/all/zero window categories exercised
    import re as _re
    m = _re.search(r"EXL3 peer summary: (\d+) rows computed by the peer, windows mixed/all/zero (\d+)/(\d+)/(\d+), (\d+) fallbacks", plog)
    if m:
        rows, wm, wa, wz, fb = (int(x) for x in m.groups())
        checks["peer_rows_computed"] = rows > 0
        checks["peer_categories"] = wm > 0 and wz > 0 and wa >= 0   # mixed and zero-peer seen
        checks["peer_summary_counters"] = {"rows": rows, "mixed": wm, "all": wa, "zero": wz, "fallbacks": fb}
    else:
        checks["peer_rows_computed"] = False
        checks["peer_categories"] = False
    # failure injection: a peer whose every launch fails must fall back to the FULL primary
    # service and still produce oracle-identical streams (nothing partial, no cache poisoning)
    pinj, ilog = run_arm("peer-injected-failure", cfg, a.out, tk, True, a.slots, fault=1)
    checks["injected_failure_streams"] = all(pinj[k] == oracle[k] for k in ("short", "medium", "chunk8k", "after"))
    checks["injected_failure_fallback_logged"] = "full primary fallback" in ilog
    checks["injected_failure_no_error_latch"] = all(x not in ilog for x in ("ERR ", "stall report", "stopping the engine"))
    report = {"checks": checks,
              "oracle_done": {k: v for k, v in oracle.items() if k.endswith("_done")},
              "peer_done": {k: v for k, v in peer.items() if k.endswith("_done")},
              "slots": a.slots}
    (a.out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["checks"], indent=2))
    print("PASS" if all(checks.values()) else "FAIL")
    raise SystemExit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
