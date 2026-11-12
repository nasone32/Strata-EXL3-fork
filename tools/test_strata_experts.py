#!/usr/bin/env python3
"""CPU-only tests for tools/strata_experts.py (the named packed-expert artifact CLI).

No GPU, no engine, no model download: tiny synthetic safetensors models (the same
writer layout src/artifact/exl3_band8_store_test.cpp uses) exercise:
  - repack publish + validate with the REAL CPU-only helper when it is built
    (skipped with a clear message otherwise), including reuse and MiMo's dense
    layer 0 being skipped (engine-list parity);
  - dry-run writes nothing; a failed helper publishes no descriptor;
  - overwrite refusal without --force; the .strata-experts.json naming rule;
  - descriptor validation: malformed, unknown-key, stale source, wrong model,
    family mismatch, damaged generation;
  - only trunk routed experts pack: dense, ngram, MTP-block and marker tensors
    are asserted absent from the manifest;
  - paths containing spaces end to end.

Usage: python tools/test_strata_experts.py
"""
from __future__ import annotations

import json
import os
import re
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import strata_experts as se  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "build-hip-exl3" / "exl3_band8_build"
REAL_HELPER = HELPER.is_file() and os.access(HELPER, os.X_OK)

QWEN_PREFIX = "model.language_model.layers.{l}.mlp.experts.{e}.{proj}_proj"
MIMO_PREFIX = "model.layers.{l}.mlp.experts.{e}.{proj}_proj"


def synth_bytes(name: str, n: int) -> bytes:
    """Deterministic per-tensor bytes (stable across runs, like the C++ fixture)."""
    out = bytearray(n)
    h = se._fnv1a(se._FNV_OFFSET, name.encode())
    for i in range(n):
        h = se._fnv1a(h, b"\x00")
        out[i] = h & 0xFF
    return bytes(out)


def write_shard(path: Path, tensors: dict) -> None:
    """A safetensors shard laid out exactly like the C++ store test's writer."""
    header, off = {}, 0
    for name in sorted(tensors):
        dtype, shape, data = tensors[name]
        header[name] = {"dtype": dtype, "shape": shape,
                        "data_offsets": [off, off + len(data)]}
        off += len(data)
    j = json.dumps(header, separators=(",", ":")).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(j)))
        f.write(j)
        for name in sorted(tensors):
            f.write(tensors[name][2])


def write_index(path: Path, weight_map: dict) -> None:
    text = json.dumps({"metadata": {"format": "strata-experts-fixture"},
                       "weight_map": weight_map}, separators=(",", ":"))
    path.write_text(text, encoding="utf-8")


def add_expert(tensors: dict, weight_map: dict, prefix: str, ki: int, nj: int,
               last_dim: int, shard: str) -> None:
    """One routed-expert triple + its mul1 marker, at a verified rate."""
    names = (prefix + s for s in (".trellis", ".suh", ".svh", ".mul1"))
    shapes = {"trellis": ("I16", [ki, nj, last_dim], ki * nj * last_dim * 2),
              "suh": ("F16", [ki * 16], ki * 32),
              "svh": ("F16", [nj * 16], nj * 32)}
    for name in names:
        suffix = name.rsplit(".", 1)[-1]
        if suffix in shapes:
            dtype, shape, n = shapes[suffix]
            tensors[name] = (dtype, shape, synth_bytes(name, n))
        else:
            tensors[name] = ("U8", [1], synth_bytes(name, 1))
        weight_map[name] = shard


def make_qwen_model(dir_path: Path, layers: int = 2, experts: int = 3) -> None:
    """A Qwen-family EXL3 dir: model.language_model.layers.* experts (dense grid),
    plus trap tensors that must NEVER pack (dense q_proj, a norm, an ngram table,
    and a full expert trio hiding under mtp.layers.0)."""
    dir_path.mkdir(parents=True, exist_ok=True)
    tensors: dict = {}
    weight_map: dict = {}
    shards = ["qw-fix-00001-of-00002.safetensors", "qw-fix-00002-of-00002.safetensors"]
    which = 0
    for l in range(layers):
        for e in range(experts):
            for proj, (ki, nj) in (("gate", (16, 8)), ("up", (16, 8)), ("down", (8, 16))):
                add_expert(tensors, weight_map,
                           QWEN_PREFIX.format(l=l, e=e, proj=proj), ki, nj,
                           64 if (l + e) % 2 == 0 else 32, shards[which])
                which ^= 1
    # traps: dense projection, a norm, an ngram table, an MTP-block expert trio
    tensors["model.language_model.layers.0.self_attn.q_proj.trellis"] = \
        ("I16", [8, 8, 32], synth_bytes("dense-q", 8 * 8 * 32 * 2))
    tensors["model.language_model.layers.0.self_attn.q_proj.suh"] = ("F16", [128], b"x" * 256)
    tensors["model.language_model.layers.0.self_attn.q_proj.svh"] = ("F16", [128], b"x" * 256)
    tensors["model.language_model.layers.0.self_attn.q_proj.mul1"] = ("U8", [1], b"\x01")
    tensors["model.language_model.layers.0.input_layernorm.weight"] = ("F16", [128], b"n" * 256)
    tensors["ngram.model.tables"] = ("I32", [64], b"g" * 256)
    for proj in ("gate", "up", "down"):
        add_expert(tensors, weight_map,
                   "mtp.layers.0.mlp.experts.0." + proj + "_proj", 16, 8, 64, shards[0])
    for name in list(tensors):
        weight_map.setdefault(name, shards[1 if len(name) % 2 else 0])
    for shard in shards:
        write_shard(dir_path / shard, {k: v for k, v in tensors.items()
                                       if weight_map[k] == shard})
    write_index(dir_path / "model.safetensors.index.json", weight_map)


def make_mimo_model(dir_path: Path, layers: int = 3, experts: int = 2) -> None:
    """A MiMo-family EXL3 dir: layer 0 DENSE (no expert tensors), layers 1..2 MoE -
    the layout that catches a request list built as 0..count-1 instead of the real
    MoE indices (the engine skips dense layers; the helper must too)."""
    dir_path.mkdir(parents=True, exist_ok=True)
    cfg = {
        "model_type": "mimo_v2", "hidden_size": 512, "num_hidden_layers": layers,
        "vocab_size": 32000, "num_attention_heads": 8, "head_dim": 64,
        "v_head_dim": 64, "num_key_value_heads": 4, "swa_num_key_value_heads": 8,
        "sliding_window": 128, "partial_rotary_factor": 0.5,
        "n_routed_experts": experts, "num_experts_per_tok": 2,
        "moe_intermediate_size": 512, "intermediate_size": 1024,
        "hybrid_layer_pattern": [0] + [1] * (layers - 1),
        "moe_layer_freq": [0] + [1] * (layers - 1),
    }
    (dir_path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    tensors: dict = {}
    weight_map: dict = {}
    shard = "mimo-fix-00001-of-00001.safetensors"
    for l in range(1, layers):  # MoE layers are 1..layers-1
        for e in range(experts):
            for proj in ("gate", "up", "down"):
                add_expert(tensors, weight_map, MIMO_PREFIX.format(l=l, e=e, proj=proj),
                           16, 8, 48, shard)
    tensors["model.layers.0.self_attn.q_proj.trellis"] = ("I16", [8, 8, 32], b"d" * 4096)
    tensors["model.layers.0.self_attn.q_proj.suh"] = ("F16", [128], b"d" * 256)
    tensors["model.layers.0.self_attn.q_proj.svh"] = ("F16", [128], b"d" * 256)
    tensors["model.layers.0.self_attn.q_proj.mul1"] = ("U8", [1], b"\x01")
    for name in list(tensors):
        weight_map.setdefault(name, shard)
    write_shard(dir_path / shard, tensors)
    write_index(dir_path / "model.safetensors.index.json", weight_map)


def run_cli(*argv: str, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "STRATA_EXPERTS_HELPER"}
    env.update(env_extra or {})
    return subprocess.run([sys.executable, str(ROOT / "tools" / "strata_experts.py"),
                           *argv], capture_output=True, text=True, env=env)


def fail_helper_script(path: Path, code: int) -> str:
    path.write_text(f"#!/bin/sh\necho 'stub helper refusing to run' >&2\nexit {code}\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return str(path)


class Fixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="strata-experts-test.",
                                        dir="/tmp/opencode"))
        cls.qwen = cls.tmp / "qwen model" / "Qwen3.8-NEXT-exl3-test"
        make_qwen_model(cls.qwen)
        cls.qwen2 = cls.tmp / "qwen-model-b"
        make_qwen_model(cls.qwen2)
        cls.mimo = cls.tmp / "mimo model" / "MiMo2.6_EXL3_test"
        make_mimo_model(cls.mimo)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def out(self, name: str) -> Path:
        d = self.tmp / "out dir"
        d.mkdir(exist_ok=True)
        return d / name


@unittest.skipUnless(REAL_HELPER, "exl3_band8_build not built "
                      "(cmake --build build-hip-exl3 --target exl3_band8_build)")
class RepackWithRealHelper(Fixtures):
    """End-to-end: the real CPU-only helper, tiny fixtures (KiB-scale payloads)."""

    def test_qwen_publish_validate_and_exclusions(self):
        out = self.out("qwen test.strata-experts.json")
        r = run_cli("repack", "--model", str(self.qwen), "--output", str(out))
        self.assertEqual(r.returncode, 0, r.stderr)
        d = json.loads(out.read_text())
        self.assertEqual(d["schema"], "strata-experts")
        self.assertEqual(d["family"], "qwen4exp")
        self.assertEqual(d["expert_prefixes"], 2 * 3 * 3)
        self.assertEqual(d["record_count"], d["expert_prefixes"] * 3)
        gen = Path(d["cache_root"]) / d["generation"]
        self.assertTrue((gen / "payload.bin").is_file())
        self.assertEqual((gen / "payload.bin").stat().st_size, d["payload_bytes"])
        # the exact estimate formula: 256 reserved + every record 256-aligned
        m = se.parse_manifest(str(gen / "manifest.bin"))
        self.assertEqual(m["payload_bytes"], d["payload_bytes"])
        # STRICT routed-expert names only: no dense, ngram, MTP-block or marker tensors
        names = {rec["name"] for rec in m["records"]}
        expected = {QWEN_PREFIX.format(l=l, e=e, proj=p) + s
                    for l in range(2) for e in range(3)
                    for p in ("gate", "up", "down")
                    for s in (".trellis", ".suh", ".svh")}
        self.assertEqual(names, expected)
        for trap in ("ngram.model.tables",
                     "mtp.layers.0.mlp.experts.0.gate_proj.trellis",
                     "model.language_model.layers.0.self_attn.q_proj.trellis",
                     "model.language_model.layers.0.input_layernorm.weight"):
            self.assertNotIn(trap, names)
        for name in names:
            self.assertTrue(se.STRICT_EXPERT_TENSOR.match(name), name)
        # validate passes against the same model
        rv = run_cli("validate", str(out), "--model", str(self.qwen))
        self.assertEqual(rv.returncode, 0, rv.stderr)
        # env prints exactly the two engine lines
        re_ = run_cli("env", str(out), "--model", str(self.qwen))
        self.assertEqual(re_.returncode, 0, re_.stderr)
        lines = sorted(re_.stdout.strip().splitlines())
        self.assertEqual(lines, [f"STRATA_EXL3_BAND8_CACHE={d['cache_root']}",
                                 "STRATA_QWEN_BAND8=1"])

    def test_reuse_same_generation_no_rewrite(self):
        out = self.out("reuse test.strata-experts.json")
        r1 = run_cli("repack", "--model", str(self.qwen), "--output", str(out))
        self.assertEqual(r1.returncode, 0, r1.stderr)
        d = json.loads(out.read_text())
        payload = Path(d["cache_root"]) / d["generation"] / "payload.bin"
        first = (payload.stat().st_mtime_ns, payload.stat().st_size)
        # a second repack (forced descriptor replace) must REUSE the generation
        r2 = run_cli("repack", "--model", str(self.qwen), "--output", str(out),
                     "--force")
        self.assertEqual(r2.returncode, 0, r2.stderr)
        self.assertIn("reused", r2.stdout)
        self.assertEqual((payload.stat().st_mtime_ns, payload.stat().st_size), first)
        # a DIFFERENT descriptor name sharing the cache root also reuses it
        out2 = self.out("reuse adopt.strata-experts.json")
        r3 = run_cli("repack", "--model", str(self.qwen), "--output", str(out2),
                     "--cache-root", d["cache_root"])
        self.assertEqual(r3.returncode, 0, r3.stderr)
        d3 = json.loads(out2.read_text())
        self.assertEqual(d3["generation"], d["generation"])
        gens = [p.name for p in Path(d["cache_root"]).iterdir() if p.is_dir()]
        self.assertEqual(gens, [d["generation"]])  # no second 80-GiB-style copy

    def test_mimo_dense_layer0_skipped_engine_list_parity(self):
        out = self.out("mimo test.strata-experts.json")
        r = run_cli("repack", "--model", str(self.mimo), "--output", str(out))
        self.assertEqual(r.returncode, 0, r.stderr)
        d = json.loads(out.read_text())
        self.assertEqual(d["family"], "mimo_v2")
        self.assertEqual(d["moe_layers"], 2)  # layers 1 and 2; layer 0 is dense
        gen = Path(d["cache_root"]) / d["generation"]
        m = se.parse_manifest(str(gen / "manifest.bin"))
        layers = {int(se.STRICT_EXPERT_TENSOR.match(rec["name"]).group(2))
                  for rec in m["records"]}
        self.assertEqual(layers, {1, 2})  # the ENGINE's list, not 0..count-1
        rv = run_cli("validate", str(out), "--model", str(self.mimo))
        self.assertEqual(rv.returncode, 0, rv.stderr)
        re_ = run_cli("env", str(out), "--model", str(self.mimo))
        self.assertIn("STRATA_MIMO_BAND8=1", re_.stdout)

    def test_stale_source_refused_at_validate(self):
        out = self.out("stale test.strata-experts.json")
        r = run_cli("repack", "--model", str(self.qwen), "--output", str(out))
        self.assertEqual(r.returncode, 0, r.stderr)
        shard = next((self.qwen).glob("qw-fix-*.safetensors"))
        st = shard.stat()
        os.utime(shard, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
        rv = run_cli("validate", str(out), "--model", str(self.qwen))
        self.assertEqual(rv.returncode, 2)
        self.assertIn("stale artifact", rv.stderr)
        self.assertIn("rerun the repack", rv.stderr)

    def test_wrong_model_refused_at_validate(self):
        out = self.out("wrongmodel test.strata-experts.json")
        r = run_cli("repack", "--model", str(self.qwen), "--output", str(out))
        self.assertEqual(r.returncode, 0, r.stderr)
        rv = run_cli("validate", str(out), "--model", str(self.qwen2))
        self.assertEqual(rv.returncode, 2)
        self.assertTrue("fingerprints as" in rv.stderr or "source mismatch" in rv.stderr,
                        rv.stderr)
        # family mismatch, too
        rv2 = run_cli("validate", str(out), "--model", str(self.mimo))
        self.assertEqual(rv2.returncode, 2)
        self.assertIn("family mismatch", rv2.stderr)

    def test_damaged_generation_refused(self):
        out = self.out("damaged test.strata-experts.json")
        r = run_cli("repack", "--model", str(self.qwen), "--output", str(out))
        self.assertEqual(r.returncode, 0, r.stderr)
        d = json.loads(out.read_text())
        payload = Path(d["cache_root"]) / d["generation"] / "payload.bin"
        keep = payload.read_bytes()
        try:
            payload.write_bytes(keep[:-256])  # truncated
            rv = run_cli("validate", str(out), "--model", str(self.qwen))
            self.assertEqual(rv.returncode, 2)
            self.assertIn("!= manifest", rv.stderr)
        finally:
            payload.write_bytes(keep)


class CliLogic(Fixtures):
    """Refusals and dry-run behavior that need no real helper."""

    def test_dry_run_writes_nothing(self):
        stub = fail_helper_script(self.tmp / "helper-must-not-run", 9)
        out = self.out("dryrun test.strata-experts.json")
        r = run_cli("repack", "--model", str(self.qwen), "--output", str(out),
                    "--dry-run", "--helper", stub)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("payload estimate", r.stdout)
        self.assertIn("dry run", r.stdout)
        self.assertFalse(out.exists())
        assets = se.assets_dir_of(out)
        self.assertFalse(assets.exists())  # not even the assets dir was created

    def test_output_suffix_enforced(self):
        out = self.out("plain.json")
        r = run_cli("repack", "--model", str(self.qwen), "--output", str(out),
                    "--dry-run")
        self.assertEqual(r.returncode, 2)
        self.assertIn("must end with .strata-experts.json", r.stderr)

    def test_overwrite_refused_without_force(self):
        stub = fail_helper_script(self.tmp / "helper-exit3", 3)
        out = self.out("overwrite test.strata-experts.json")
        out.write_text("{}\n")
        r = run_cli("repack", "--model", str(self.qwen), "--output", str(out),
                    "--helper", stub)
        self.assertEqual(r.returncode, 2)
        self.assertIn("already exists", r.stderr)
        self.assertIn("--force", r.stderr)
        self.assertEqual(out.read_text(), "{}\n")  # untouched

    def test_helper_failure_publishes_no_descriptor(self):
        stub = fail_helper_script(self.tmp / "helper-exit3", 3)
        out = self.out("helperfail test.strata-experts.json")
        r = run_cli("repack", "--model", str(self.qwen), "--output", str(out),
                    "--helper", stub)
        self.assertEqual(r.returncode, 2)
        self.assertIn("exited with 3", r.stderr)
        self.assertIn("no descriptor published", r.stderr)
        self.assertFalse(out.exists())

    def test_helper_zero_exit_but_no_generation_refuses_publish(self):
        stub = self.tmp / "helper-fake-ok"
        stub.write_text("#!/bin/sh\nexit 0\n")
        stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
        out = self.out("fakeok test.strata-experts.json")
        r = run_cli("repack", "--model", str(self.qwen), "--output", str(out),
                    "--helper", str(stub))
        self.assertEqual(r.returncode, 2)
        self.assertFalse(out.exists())

    def test_missing_helper_message(self):
        out = self.out("nohelper test.strata-experts.json")
        not_exec = self.tmp / "plain-file"
        not_exec.write_text("data\n")
        r = run_cli("repack", "--model", str(self.qwen), "--output", str(out),
                    "--helper", str(not_exec))
        self.assertEqual(r.returncode, 2)
        self.assertIn("is not an executable file", r.stderr)
        # the discovery path (no --helper, no default candidate) prints the build command
        empty_root = self.tmp / "empty-root"
        empty_root.mkdir(exist_ok=True)
        with self.assertRaises(se.Refused) as ctx:
            se.find_helper(empty_root, None)
        self.assertIn("cmake --build", str(ctx.exception))

    def test_family_flag_mismatch_refused_before_helper(self):
        stub = fail_helper_script(self.tmp / "helper-must-not-run", 9)
        out = self.out("family test.strata-experts.json")
        r = run_cli("repack", "--model", str(self.qwen), "--output", str(out),
                    "--family", "mimo", "--helper", stub)
        self.assertEqual(r.returncode, 2)
        self.assertIn("refused", r.stderr)
        self.assertIn("qwen4exp", r.stderr)

    def test_not_a_model_dir(self):
        out = self.out("nomodel test.strata-experts.json")
        r = run_cli("repack", "--model", str(self.tmp / "nope"), "--output", str(out),
                    "--dry-run")
        self.assertEqual(r.returncode, 2)
        self.assertIn("is not a directory", r.stderr)

    def test_neither_family(self):
        neither = self.tmp / "nofamily model"
        neither.mkdir(exist_ok=True)
        write_index(neither / "model.safetensors.index.json",
                    {"some.dense.tensor": "shard.safetensors"})
        out = self.out("nofamily test.strata-experts.json")
        r = run_cli("repack", "--model", str(neither), "--output", str(out), "--dry-run")
        self.assertEqual(r.returncode, 2)
        self.assertIn("neither family detected", r.stderr)

    def test_malformed_descriptors(self):
        # a structurally valid descriptor for the mimo fixture, mutated case by case
        weight_map = se.read_weight_map(str(self.mimo))
        family, geometry = se.detect_family(str(self.mimo), weight_map)
        prefixes, summary = se.expert_prefixes(family, geometry)
        payload, records = se.payload_estimate(str(self.mimo), weight_map, prefixes)
        fp = se.compute_fingerprint(str(self.mimo), prefixes, weight_map)
        base = {
            "schema": se.SCHEMA, "schema_version": 1, "family": family,
            "created_utc": "2026-11-08T00:00:00Z", "tool": se.TOOL,
            "model_dir": str(self.mimo), "model_dir_canonical": str(self.mimo),
            "cache_root": str(self.tmp / "adopted"), "generation": fp,
            "fingerprint": fp, "record_count": records,
            "payload_bytes": payload, "expected_payload_bytes": payload,
            "expert_prefixes": len(prefixes), "moe_layers": summary["moe_layers"],
            "experts_per_layer": summary["experts_per_layer"],
            "source": {"index_bytes": 1, "index_mtime_ns": 1, "shards": ["a"]},
            "engine_env": {"STRATA_MIMO_BAND8": "1",
                           "STRATA_EXL3_BAND8_CACHE": str(self.tmp / "adopted")},
        }
        cases = {
            "truncated": "{",
            "not-object": "[1,2]",
            "unknown-key": json.dumps({**base, "surprise": 1}),
            "missing-key": json.dumps({k: v for k, v in base.items() if k != "cache_root"}),
            "bad-hex": json.dumps({**base, "fingerprint": "XYZ", "generation": "XYZ"}),
            "bad-schema": json.dumps({**base, "schema": "other-thing"}),
            "bad-version": json.dumps({**base, "schema_version": 99}),
            "bad-family": json.dumps({**base, "family": "gpt"}),
            "bad-env": json.dumps({**base, "engine_env": {"STRATA_QWEN_BAND8": "1",
                                                          "STRATA_EXL3_BAND8_CACHE": "/x"}}),
            "count-mismatch": json.dumps({**base, "record_count": 7}),
            "size-mismatch": json.dumps({**base, "expected_payload_bytes": 5}),
            # wrong TYPES must refuse cleanly (exit 2, a message - never a traceback)
            "type-cache-root-int": json.dumps({**base, "cache_root": 5,
                                               "engine_env": {**base["engine_env"],
                                                              "STRATA_EXL3_BAND8_CACHE": 5}}),
            "type-env-value-int": json.dumps({**base, "engine_env": {
                "STRATA_MIMO_BAND8": 1,
                "STRATA_EXL3_BAND8_CACHE": base["cache_root"]}}),
            "type-relative-cache": json.dumps({**base, "cache_root": "rel/path"}),
            "type-created-int": json.dumps({**base, "created_utc": 5}),
            "type-shards-int": json.dumps({**base, "source": {**base["source"],
                                                              "shards": [1]}}),
            "type-env-cache-mismatch": json.dumps({**base, "engine_env": {
                "STRATA_MIMO_BAND8": "1", "STRATA_EXL3_BAND8_CACHE": "/other/root"}}),
        }
        for name, text in cases.items():
            p = self.out(f"bad-{name}.strata-experts.json")
            p.write_text(text)
            r = run_cli("validate", str(p))
            self.assertEqual(r.returncode, 2, f"{name}: {r.stdout}{r.stderr}")
            self.assertTrue(r.stderr.strip(), name)
            self.assertNotIn("Traceback", r.stderr,
                             f"{name}: must refuse cleanly, not crash")

    def test_descriptor_for_missing_generation(self):
        weight_map = se.read_weight_map(str(self.mimo))
        family, geometry = se.detect_family(str(self.mimo), weight_map)
        prefixes, summary = se.expert_prefixes(family, geometry)
        payload, records = se.payload_estimate(str(self.mimo), weight_map, prefixes)
        fp = se.compute_fingerprint(str(self.mimo), prefixes, weight_map)
        d = {
            "schema": se.SCHEMA, "schema_version": 1, "family": family,
            "created_utc": "2026-11-08T00:00:00Z", "tool": se.TOOL,
            "model_dir": str(self.mimo), "model_dir_canonical": str(self.mimo),
            "cache_root": str(self.tmp / "nowhere"), "generation": fp,
            "fingerprint": fp, "record_count": records,
            "payload_bytes": payload, "expected_payload_bytes": payload,
            "expert_prefixes": len(prefixes), "moe_layers": summary["moe_layers"],
            "experts_per_layer": summary["experts_per_layer"],
            "source": {"index_bytes": 1, "index_mtime_ns": 1, "shards": ["a"]},
            "engine_env": {"STRATA_MIMO_BAND8": "1",
                           "STRATA_EXL3_BAND8_CACHE": str(self.tmp / "nowhere")},
        }
        p = self.out("nogen test.strata-experts.json")
        p.write_text(json.dumps(d))
        r = run_cli("validate", str(p))
        self.assertEqual(r.returncode, 2)
        self.assertIn("missing", r.stderr.lower())

    def test_windows_refused_in_process(self):
        # main() turns Refused into exit code 2 with the platform message on stderr
        saved = sys.platform
        try:
            se.sys.platform = "win32"
            rc = se.main(["repack", "--model", str(self.qwen),
                          "--output", str(self.out("win test.strata-experts.json")),
                          "--dry-run"])
        finally:
            se.sys.platform = saved
        self.assertEqual(rc, 2)

    def test_publish_no_clobber_and_race(self):
        out = self.out("race test.strata-experts.json")
        out.write_text("ORIGINAL\n")
        with self.assertRaises(se.Refused):
            se.publish_descriptor(out, {"schema": "x"}, force=False)
        self.assertEqual(out.read_text(), "ORIGINAL\n")  # untouched, no TOCTOU window
        se.publish_descriptor(out, {"schema": "y"}, force=True)  # explicit replace
        self.assertIn('"schema": "y"', out.read_text())
        # a real race: N concurrent no-clobber publishes to a fresh name, one winner
        import threading
        target = self.out("race fresh.strata-experts.json")
        results = []

        def worker(i):
            try:
                se.publish_descriptor(target, {"schema": f"t{i}"}, force=False)
                results.append(("ok", i))
            except se.Refused:
                results.append(("refused", i))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        winners = [r for r in results if r[0] == "ok"]
        self.assertEqual(len(winners), 1, results)
        self.assertIn(f'"schema": "t{winners[0][1]}"', target.read_text())

    def test_repack_prints_tracked_launch_command(self):
        if not REAL_HELPER:
            self.skipTest("exl3_band8_build not built")
        for model, family_hint, single in ((self.mimo, "mimo", False),
                                           (self.qwen, "qwen", True)):
            out = self.out(f"hint {family_hint}.strata-experts.json")
            r = run_cli("repack", "--model", str(model), "--output", str(out))
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("strata_experts.py launch --config", r.stdout)
            if single:
                self.assertIn("--single-gpu", r.stdout)
            else:
                self.assertNotIn("--single-gpu", r.stdout)
            self.assertNotIn("run-peer.sh", r.stdout)  # no misleading wrong-model hint

    def test_fingerprint_matches_helper_generation(self):
        """The Python fingerprint and the helper's generation dir must agree (guards
        the compute_fingerprint mirror against drift)."""
        if not REAL_HELPER:
            self.skipTest("exl3_band8_build not built")
        weight_map = se.read_weight_map(str(self.qwen))
        family, geometry = se.detect_family(str(self.qwen), weight_map)
        prefixes, _ = se.expert_prefixes(family, geometry)
        fp = se.compute_fingerprint(str(self.qwen), prefixes, weight_map)
        out = self.out("fp test.strata-experts.json")
        r = run_cli("repack", "--model", str(self.qwen), "--output", str(out))
        self.assertEqual(r.returncode, 0, r.stderr)
        d = json.loads(out.read_text())
        self.assertEqual(d["fingerprint"], fp)


if __name__ == "__main__":
    unittest.main(verbosity=2)
