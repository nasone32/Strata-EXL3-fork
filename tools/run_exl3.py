#!/usr/bin/env python3
"""Portable EXL3 server entry point. No machine-specific model paths or presets."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

import strata_experts as experts

ROOT = Path(__file__).resolve().parents[1]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--family", choices=("qwen", "mimo"), default="qwen")
    p.add_argument("--model-dir", default=os.environ.get("STRATA_EXL3_MODEL"),
                   help="local EXL3 directory (or STRATA_EXL3_MODEL)")
    p.add_argument("--tokenizer", default=os.environ.get("STRATA_EXL3_TOKENIZER"),
                   help="converted tokenizer directory; otherwise detected in the model")
    p.add_argument("--engine-bin", default=os.environ.get("STRATA_EXL3_ENGINE",
                   str(ROOT / "build-hip-exl3/strata")))
    p.add_argument("--gpu", default=os.environ.get("STRATA_EXL3_GPUS"),
                   help="physical GPU list, default 0 (0,1 when a peer is requested)")
    p.add_argument("--peer-device", type=int, default=os.environ.get("STRATA_EXL3_PEER_DEVICE"),
                   help="peer index within the visible --gpu list; usually 1")
    p.add_argument("--resident", choices=("all", "peer", "0"),
                   default=os.environ.get("STRATA_EXL3_RESIDENT", "all"))
    p.add_argument("--cpu", choices=("auto", "on", "off"), default="auto",
                   help="auto: on for one-GPU Qwen and for MiMo; off for peer Qwen")
    p.add_argument("--cpu-threads", type=int, help="CPU expert workers (default at most 8)")
    p.add_argument("--aux-cpus", default=os.environ.get("STRATA_AUX_CPUS"),
                   help="auxiliary thread affinity: list, auto, or off; default unpinned")
    p.add_argument("--expert-cache", default=os.environ.get("STRATA_EXL3_EXPERT_CACHE", "auto"))
    p.add_argument("--peer-slots", type=int, default=None,
                   help="Qwen peer cap; default fills the engine's free-VRAM budget")
    p.add_argument("--max-context", type=int, help="default: Qwen 65536, MiMo 16384")
    p.add_argument("--prefill", type=int, help="default: Qwen 8192, MiMo 512")
    p.add_argument("--expert-vram", type=float, help="MiMo: primary expert budget, GiB")
    p.add_argument("--reserve", type=int, default=2048, help="primary VRAM reserve, MiB")
    p.add_argument("--peer-reserve", type=int, default=4096, help="peer VRAM reserve, MiB")
    p.add_argument("--expert-profile", help="optional measured expert placement profile")
    p.add_argument("--no-expert-profile", action="store_true")
    p.add_argument("--ple-file", default=os.environ.get("STRATA_EXL3_PLE_FILE"))
    p.add_argument("--mtp-dir", default=os.environ.get("STRATA_EXL3_MTP_DIR"))
    p.add_argument("--spec", type=int, default=4)
    p.add_argument("--spec-min-p", type=float, default=0.5)
    p.add_argument("--drafter", help="MiMo DFlash directory (optional)")
    p.add_argument("--experts", default=os.environ.get("STRATA_EXPERTS") or None,
                   help="existing .strata-experts.json descriptor; repacking is off otherwise")
    p.add_argument("--repack-in-memory", action="store_true")
    p.add_argument("--port", type=int, default=8085)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--api-key", help="required when listening beyond localhost")
    p.add_argument("--open", action="store_true", help="open the browser after loading")
    p.add_argument("--dry-run", action="store_true", help="print the config; start nothing")
    return p


def make_config(a):
    a.max_context = a.max_context or (16384 if a.family == "mimo" else 65536)
    a.prefill = a.prefill or (512 if a.family == "mimo" else 8192)
    if not a.model_dir:
        raise experts.Refused("--model-dir (or STRATA_EXL3_MODEL) is required")
    model = Path(a.model_dir).expanduser().resolve()
    if not model.is_dir():
        raise experts.Refused(f"model directory does not exist: {model}")
    family, _ = experts.detect_family(str(model), experts.read_weight_map(str(model)))
    want = "mimo_v2" if a.family == "mimo" else "qwen4exp"
    if family != want:
        raise experts.Refused(f"--family {a.family} does not match this model ({family})")
    if a.host not in ("127.0.0.1", "localhost", "::1") and not a.api_key:
        raise experts.Refused("--api-key is required with a non-local --host")
    if a.resident == "peer" and (a.family == "qwen" or a.peer_device is None):
        raise experts.Refused("RESIDENT=peer currently requires MiMo with a peer GPU; "
                              "for low-RAM Qwen use --resident 0 and reduce the GPU cache")
    if min(a.max_context, a.prefill) < 1 or min(a.reserve, a.peer_reserve) < 0:
        raise experts.Refused("context/prefill must be positive; VRAM reserves cannot be negative")
    if a.cpu_threads is not None and a.cpu_threads < 1:
        raise experts.Refused("--cpu-threads must be positive")
    if a.peer_slots is not None and not 0 <= a.peer_slots <= 65535:
        raise experts.Refused("--peer-slots must be 0..65535")
    try:
        gpus = [int(g) for g in (a.gpu or ("0,1" if a.peer_device is not None else "0")).split(",")]
    except ValueError:
        raise experts.Refused("--gpu must be a comma-separated list of device numbers")
    if not gpus or len(set(gpus)) != len(gpus) or min(gpus) < 0:
        raise experts.Refused("--gpu must list distinct, nonnegative device numbers")
    if a.peer_device is not None and not 0 < a.peer_device < len(gpus):
        raise experts.Refused("--peer-device must name a non-primary index in the visible --gpu list")
    if a.peer_device is None and len(gpus) != 1:
        raise experts.Refused("multiple GPUs need --peer-device; these launchers do not split layers")
    tokenizer = Path(a.tokenizer).expanduser().resolve() if a.tokenizer else next(
        (t for t in (model / "tokenizer", model) if (t / "vocab.json").is_file()), None)
    if tokenizer is None or not (tokenizer / "vocab.json").is_file():
        raise experts.Refused("no pack-layout tokenizer found; run tools/exl3/tokenizer_to_pack.py "
                              "--hf MODEL --out TOKENIZER, then pass --tokenizer TOKENIZER")
    cpu_key = "STRATA_MIMO_CPU_TIER" if a.family == "mimo" else "STRATA_EXL3_CPU_TIER"
    cpu = a.cpu == "on" or (a.cpu == "auto" and (a.family == "mimo" or a.peer_device is None))
    if a.cpu == "auto" and cpu_key in os.environ:
        cpu = os.environ[cpu_key] != "0"
    env = {"HIP_VISIBLE_DEVICES": ",".join(map(str, gpus)),
           "STRATA_EXL3_RESIDENT": a.resident, cpu_key: "1" if cpu else "0"}
    workers = a.cpu_threads or int(os.environ.get(
        "STRATA_MIMO_CPU_WORKERS" if a.family == "mimo" else "STRATA_EXL3_CPU_THREADS",
        str(experts.cpu_worker_cap())))
    if cpu:
        env["STRATA_MIMO_CPU_WORKERS" if a.family == "mimo" else "STRATA_EXL3_CPU_THREADS"] = str(workers)
    if a.aux_cpus is not None:
        env["STRATA_AUX_CPUS"] = a.aux_cpus
    args = ["--exl3", str(model)]
    if a.family == "qwen":
        ple = Path(a.ple_file).expanduser().resolve() if a.ple_file else next(
            (model / n for n in ("ngram_embedding.safetensors", "gram_embedding.safetensors")
             if (model / n).is_file()), None)
        if ple is None or not ple.is_file():
            raise experts.Refused("Qwen's n-gram table is missing; download the complete quant "
                                  "or pass --ple-file")
        args += ["--ple-exl3", str(ple), "--ple-io", "direct", "--expert-cache", a.expert_cache,
                 "--prefill", str(a.prefill), "--max-context", str(a.max_context), "--kv", "int8",
                 "--vram-reserve-mib", str(a.reserve)]
        profile = Path(a.expert_profile).expanduser() if a.expert_profile else ROOT / "data/expert-profile.bin"
        if not a.no_expert_profile and profile.is_file():
            args += ["--expert-profile", str(profile)]
        elif a.expert_profile and not a.no_expert_profile:
            raise experts.Refused(f"expert profile not found: {profile}")
        if cpu:
            for k, v in experts.QWEN_AUTO_CPU_FLAGS:
                if k != cpu_key:
                    env[k] = os.environ.get(k, v)
            if a.peer_device is not None:
                env["STRATA_EXL3_CPU_HYBRID_PEER"] = os.environ.get("STRATA_EXL3_CPU_HYBRID_PEER", "1")
        if a.mtp_dir:
            args += ["--mtp", str(Path(a.mtp_dir).expanduser()), "--spec", str(a.spec),
                     "--spec-min-p", str(a.spec_min_p)]
    else:
        args += ["--dense-q8", "--max-context", str(a.max_context), "--prefill", str(a.prefill)]
        if a.expert_vram is not None:
            args += ["--expert-vram", str(a.expert_vram)]
        if cpu:
            args += ["--cpu-tier"]
        if a.drafter:
            args += ["--dflash", str(Path(a.drafter).expanduser())]
        # MiMo remains a separate experimental path, not Qwen's default profile.
        for k, v in experts.MIMO_AUTO_FLAGS:
            env[k] = os.environ.get(k, v)
        if cpu:
            for k, v in experts.MIMO_AUTO_CPU_FLAGS:
                env[k] = os.environ.get(k, v)
        env["STRATA_MIMO_LOAD_MTP"] = "0"
        env["STRATA_MIMO_DENSE_CACHE"] = "0"
        env["STRATA_EXL3_VARIABLE_EXPERT_BITS"] = "1"
        env["STRATA_MIMO_WINDOW_GPU_HITS"] = os.environ.get("STRATA_MIMO_WINDOW_GPU_HITS", "1")
        env["STRATA_MIMO_WINDOW_DEVICE_MISSES"] = os.environ.get("STRATA_MIMO_WINDOW_DEVICE_MISSES", "1")
        if a.peer_device is not None:
            env["STRATA_MIMO_PEER_DEVICE_RESULTS"] = os.environ.get("STRATA_MIMO_PEER_DEVICE_RESULTS", "1")
    if a.peer_device is not None:
        args += ["--peer-device", str(a.peer_device)]
        if a.family == "qwen":
            args += ["--peer-reserve-mib", str(a.peer_reserve)]
            # The inherited engine's unset cap is only 256 experts. Let its
            # existing byte-budget check fill the card instead of hard-coding
            # 10200/7200 slots for the developer's cards and quantization.
            args += ["--peer-slots", str(a.peer_slots if a.peer_slots is not None else 65535)]
        else:
            if a.peer_slots is not None:
                raise experts.Refused("MiMo uses --expert-vram and STRATA_MIMO_PEER_RESERVE_MIB, not --peer-slots")
            env["STRATA_MIMO_PEER_RESERVE_MIB"] = str(a.peer_reserve)
    return {"exe": str(Path(a.engine_bin).expanduser().resolve()), "cwd": str(ROOT),
            "backend": "hip", "gpu": gpus, "tokenizer": str(tokenizer), "lib_dirs": [],
            **({"arch": "mimo_v2"} if a.family == "mimo" else {}), "env": env, "args": args}


def main(argv=None):
    p = parser()
    a, server_args = p.parse_known_args(argv)
    try:
        cfg = make_config(a)
        if not a.dry_run and not os.access(cfg["exe"], os.X_OK):
            raise experts.Refused(f"engine missing: {cfg['exe']}; build the strata target first")
        with tempfile.TemporaryDirectory(prefix="strata-exl3-config-") as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps(cfg), encoding="utf-8")
            extra = ["--host", a.host, "--port", str(a.port)]
            if a.api_key:
                extra += ["--api-key", a.api_key]
            if a.open:
                extra += ["--open"]
            server_args = server_args[1:] if server_args[:1] == ["--"] else server_args
            if any(x.split("=", 1)[0] in ("--host", "--port", "--api-key", "--config", "--engine")
                   for x in server_args):
                raise experts.Refused("pass host/port/key before '--'; server config/engine overrides are not accepted here")
            extra += server_args
            return experts.cmd_launch(argparse.Namespace(
                config=str(path), experts=a.experts, repack_in_memory=a.repack_in_memory,
                single_gpu=False, performance="baseline", server=None, python=sys.executable,
                dry_run=a.dry_run, server_args=extra), ROOT)
    except (experts.Refused, ValueError) as e:
        p.error(str(e))


if __name__ == "__main__":
    sys.exit(main())
