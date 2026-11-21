#!/usr/bin/env python3
"""Named packed-expert artifacts ("repack") for EXL3 models - user-facing CLI.

Strata's routed experts can be repacked once, offline and CPU-only, into a persistent
band-8 cache (see notes/EXL3-BAND8-FEASIBILITY-2026-11-06.md and
notes/MIMO-BAND8-INTEGRATION-2026-11-06.md): a pure, byte-size-preserving tile
permutation of ONLY the routed experts' trellis planes.  Dense projections, norms,
ngram tables, the in-file MTP block and any external drafter stay in the original
model dir and are always read from there.  The pack is strictly opt-in: without an
artifact selected at launch, the engine's native byte path runs exactly as before.

This CLI wraps the CPU-only builder binary (``exl3_band8_build``, no GPU, no engine
context) so a user can produce a NAMED artifact:

    python tools/strata_experts.py repack \
        --model /mnt/models/MiMo2.6_EXL3_2.27bpw \
        --output /mnt/bigssd/mimo-2.27.strata-experts.json

``--output`` is a real file the user names (created last, atomically, only after the
builder finished and every check below passed).  The heavy payload lives in a managed
directory NEXT TO it (``/mnt/bigssd/mimo-2.27.strata-experts/<fingerprint-hex16>/``):
descriptor ~1 KB, payload about the routed experts' size (measured full models:
MiMo 2.27 bpw 79.54 GiB, Qwen3.8-NEXT 4 bpw 56.69 GiB; built once, reused by
fingerprint afterwards).

Subcommands:
  repack    build (or reuse) a packed-expert generation and publish the descriptor
  validate  check a descriptor against its generation and (with --model) the source
  env       print the engine environment lines a descriptor selects (launcher use)
  launch    start the server from a config file, optionally selecting an artifact
            (the tracked, portable interface - the run-*.sh presets are gitignored
            personal convenience wrappers around the same machinery), or
            --repack-in-memory for the same packed bytes built in RAM at startup
            (no disk artifact at all)

The descriptor is DATA ONLY - it is read as JSON, nothing in it is ever executed, and
the launch path never evals shell text: it derives a second config file and execs the
server on it.

Generation-integrity note (exact behavior): a generation whose manifest fingerprint,
record names or payload SIZE disagree with what is recomputed is REBUILT IN PLACE
under the same fingerprint directory by the store; a matching generation is reused
untouched.  Validation is metadata-level (fingerprints, sizes, manifest bounds, name
contract): same-size corrupt payload bytes are NOT detected unless the cache was
built with a payload checksum and the engine's verify option is on.

The descriptor is DATA ONLY - it is read as JSON, nothing in it is ever executed.
Validation recomputes the same ordered change-detector fingerprint the engine's
Exl3Band8Store uses (canonical model path, index bytes+size+mtime, every referenced
shard's path/size/mtime, exact prefix list order), so a model that moved or changed
since the repack is refused with a clear message instead of silently mismatching.

Platform limits (engine-side, mirrored here for the offline pack): POSIX/Linux only
(the store needs mmap; Windows refuses); the packed GPU path is validated on HIP
gfx11 (RDNA3) builds; the packed CPU tier needs AVX-512BW.  The repack helper itself
is CPU-only and touches no GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SCHEMA = "strata-experts"
SCHEMA_VERSION = 1
TOOL = "strata-experts/1"
OUTPUT_SUFFIX = ".strata-experts.json"
DESCRIPTOR_KEYS = frozenset({
    "schema", "schema_version", "family", "created_utc", "tool",
    "model_dir", "model_dir_canonical", "cache_root", "generation", "fingerprint",
    "record_count", "payload_bytes", "expected_payload_bytes", "expert_prefixes",
    "moe_layers", "experts_per_layer", "source", "engine_env",
})
FAMILIES = ("mimo_v2", "qwen4exp")
FAMILY_ENV_SWITCH = {"mimo_v2": "STRATA_MIMO_BAND8", "qwen4exp": "STRATA_QWEN_BAND8"}
HEX16 = re.compile(r"^[0-9a-f]{16}$")
STRICT_EXPERT_TENSOR = re.compile(
    r"^(model\.layers|model\.language_model\.layers)\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate|up|down)_proj\.(trellis|suh|svh)$")
QWEN_INDEX_KEY = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.")
MIMO_INDEX_KEY = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.")

GIB = 1073741824.0

# ---- band8 store format constants (mirror of src/artifact/exl3_band8_store.cpp) ------------
BAND8_FORMAT_VERSION = 1          # include/strata/core/exl3_layout.hpp
MANIFEST_MAGIC = b"STRB8\x00\x00"
MANIFEST_VERSION = 1
MANIFEST_HEADER_BYTES = 80
MANIFEST_RECORD_BYTES = 40
RECORD_ALIGN = 256
VERIFIED_RATES = {(2, 0), (2, 1), (3, 0), (4, 0)}


class Refused(Exception):
    """A clear, user-facing refusal (exit code 2).  Nothing was published."""


# ---- FNV-1a 64 (the store's ordered change detector) ----------------------------------------
_FNV_OFFSET = 14695981039346656037
_FNV_PRIME = 1099511628211


def _fnv1a(h: int, data: bytes) -> int:
    for b in data:
        h ^= b
        h = (h * _FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return h


def _fnv_u64(h: int, v: int) -> int:
    return _fnv1a(h, struct.pack("<Q", v))


def canonical_path(path: str) -> str:
    """The store's canonical_path: weakly-canonical, trailing '/' stripped.

    os.path.realpath equals std::filesystem::weakly_canonical for paths whose
    prefix exists (both resolve symlinks and drop '.'/'..'), which is always the
    case here because the caller stat'ed the file first.
    """
    out = os.path.realpath(path)
    while len(out) > 1 and out.endswith("/"):
        out = out[:-1]
    return out


def compute_fingerprint(model_dir: str, prefixes: list[str],
                        weight_map: dict[str, str]) -> str:
    """Mirror of Exl3Band8Store's compute_fingerprint (see exl3_band8_store.cpp).

    NOT a cryptographic identity: an adversarial edit that preserves every hashed
    datum can still alias.  It detects moved/edited sources, which is what the
    descriptor's pre-flight check needs.  Any drift from the C++ side is caught by
    the post-build generation cross-check in repack (the helper's own directory is
    authoritative; a mismatch refuses the publish).
    """
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    idx = Path(index_path).read_bytes()
    st = os.stat(index_path)
    # 26 bytes: the C++ seeds with the literal PLUS its NUL terminator (length 26
    # over a 25-char string) - mirrored byte for byte, NUL included.
    h = _fnv1a(_FNV_OFFSET, b"strata/exl3-band8-store/1\x00")
    h = _fnv_u64(h, BAND8_FORMAT_VERSION)
    d = canonical_path(model_dir).encode()
    h = _fnv_u64(h, len(d))
    h = _fnv1a(h, d)
    h = _fnv_u64(h, len(idx))
    h = _fnv1a(h, idx)
    h = _fnv_u64(h, st.st_size)
    h = _fnv_u64(h, st.st_mtime_ns)
    shards = sorted({weight_map[p + s] for p in prefixes for s in (".trellis", ".suh", ".svh")})
    h = _fnv_u64(h, len(shards))
    for s in shards:  # sorted order, same as the C++ std::set iteration
        sst = os.stat(os.path.join(model_dir, s))
        full = canonical_path(os.path.join(model_dir, s)).encode()
        sb = s.encode()
        h = _fnv_u64(h, len(sb))
        h = _fnv1a(h, sb)
        h = _fnv_u64(h, len(full))
        h = _fnv1a(h, full)
        h = _fnv_u64(h, sst.st_size)
        h = _fnv_u64(h, sst.st_mtime_ns)
    h = _fnv_u64(h, len(prefixes))
    for p in prefixes:
        pb = p.encode()
        h = _fnv_u64(h, len(pb))
        h = _fnv1a(h, pb)
    return f"{h:016x}"


# ---- family detection and the exact request list the engine builds --------------------------
def mimo_config_geometry(model_dir: str) -> dict:
    """The subset of MiMo's config.json the builder's request list needs.

    Mirrors mimo_geometry_from_config's layer bookkeeping (moe_layer_freq default
    all-MoE, list length must equal num_hidden_layers) - NOT the full geometry.
    """
    path = os.path.join(model_dir, "config.json")
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        raise NotMimo(path)  # no readable config.json: not a MiMo dir (the C++
        # mimo_detect_config also answers false for a missing/unreadable config)
    if len(text.encode()) > 8 * 1024 * 1024:
        raise Refused(f"{path}: implausibly large for a MiMo config")
    if '"mimo_v2"' not in text:
        raise NotMimo(path)
    try:
        cfg = json.loads(text)
    except json.JSONDecodeError as e:
        raise Refused(f"{path}: declares mimo_v2 but is not valid JSON ({e})")
    if cfg.get("model_type") != "mimo_v2":
        raise NotMimo(path)
    n_layers = cfg.get("num_hidden_layers")
    n_expert = cfg.get("n_routed_experts")
    if not isinstance(n_layers, int) or not isinstance(n_expert, int) \
            or n_layers <= 0 or n_expert <= 0:
        raise Refused(f"{path}: mimo_v2 config without num_hidden_layers/n_routed_experts")
    moe = cfg.get("moe_layer_freq")
    if moe is None:
        moe = [1] * n_layers
    if not isinstance(moe, list) or len(moe) != n_layers or \
            not all(isinstance(v, int) for v in moe):
        raise Refused(f"{path}: moe_layer_freq must be a list of num_hidden_layers ints")
    return {"n_layers": n_layers, "n_expert": n_expert, "moe_layer": moe}


class NotMimo(Exception):
    def __init__(self, path: str):
        super().__init__(path)
        self.path = path


def qwen_grid_from_index(weight_map: dict[str, str]) -> tuple[int, int] | None:
    """(layers, experts) as the dense grid 0..max the engine's STRATA_QWEN_BAND8 list uses."""
    max_l = max_e = -1
    for name in weight_map:
        m = QWEN_INDEX_KEY.match(name)
        if m:
            max_l = max(max_l, int(m.group(1)))
            max_e = max(max_e, int(m.group(2)))
    if max_l < 0 or max_e < 0:
        return None
    return max_l + 1, max_e + 1


def detect_family(model_dir: str, weight_map: dict[str, str]) -> tuple[str, dict]:
    """Auto-detect the family the way the builder does; refuse ambiguity."""
    mimo_error = None
    is_mimo = False
    mimo_geometry = None
    try:
        mimo_geometry = mimo_config_geometry(model_dir)
        is_mimo = True
    except NotMimo as e:
        mimo_error = f"config.json at {e.path} does not declare model_type mimo_v2"
    except Refused as e:
        mimo_error = str(e)  # declares mimo_v2 but the layer bookkeeping is off
        is_mimo = True        # still a (broken) mimo dir: do not also treat it as qwen
    qwen_grid = qwen_grid_from_index(weight_map)
    if is_mimo and qwen_grid:
        raise Refused(
            f"{model_dir}: both families detected (config.json declares mimo_v2 and the index "
            f"has model.language_model.layers.*.mlp.experts.* tensors) - pass --family mimo or "
            f"--family qwen explicitly")
    if is_mimo:
        if mimo_geometry is None:
            raise Refused(f"not a usable MiMo v2 model dir: {mimo_error}")
        return "mimo_v2", mimo_geometry
    if qwen_grid:
        return "qwen4exp", {"grid": qwen_grid}
    raise Refused(
        f"{model_dir}: neither family detected - no config.json declaring model_type mimo_v2, "
        f"and the safetensors index has no model.language_model.layers.*.mlp.experts.* tensors "
        f"(the two currently packable families)")


def expert_prefixes(family: str, geometry: dict) -> tuple[list[str], dict]:
    """The EXACT request list the engine builds (order is part of the fingerprint).

    mimo_v2: actual MoE layer indices (dense layers skipped), layer-major, expert-minor,
    gate/up/down - byte-identical with mimo_engine.cu's STRATA_MIMO_BAND8 list and the
    exl3_band8_build MiMo loop.  qwen4exp: dense grid 0..max, the
    model.language_model.layers.<l>.mlp.experts.<e>.<proj>_proj names the engine's
    generate.cpp STRATA_QWEN_BAND8 block builds.
    """
    prefixes: list[str] = []
    if family == "mimo_v2":
        moe_layers = [l for l, m in enumerate(geometry["moe_layer"]) if m != 0]
        n_expert = geometry["n_expert"]
        for l in moe_layers:
            for e in range(n_expert):
                for proj in ("gate", "up", "down"):
                    prefixes.append(f"model.layers.{l}.mlp.experts.{e}.{proj}_proj")
        summary = {"moe_layers": len(moe_layers), "experts_per_layer": n_expert}
    else:
        layers, n_expert = geometry["grid"]
        for l in range(layers):
            for e in range(n_expert):
                for proj in ("gate", "up", "down"):
                    prefixes.append(
                        f"model.language_model.layers.{l}.mlp.experts.{e}.{proj}_proj")
        summary = {"moe_layers": layers, "experts_per_layer": n_expert}
    if not prefixes:
        raise Refused("the model declares no MoE experts - nothing to pack")
    return prefixes, summary


# ---- safetensors metadata (headers only; shards are never copied) ---------------------------
def read_weight_map(model_dir: str) -> dict[str, str]:
    path = os.path.join(model_dir, "model.safetensors.index.json")
    if not os.path.isfile(path):
        raise Refused(f"{path} is missing (an EXL3 model dir needs a safetensors index)")
    try:
        wm = json.loads(Path(path).read_text(encoding="utf-8"))["weight_map"]
    except (json.JSONDecodeError, KeyError) as e:
        raise Refused(f"{path}: cannot read the weight_map ({e})")
    if not isinstance(wm, dict) or not wm:
        raise Refused(f"{path}: the weight_map is empty or not an object")
    return wm


def shard_tensor_meta(model_dir: str, shard: str) -> dict[str, tuple[str, list[int]]]:
    """{tensor: (dtype, shape)} from one shard's safetensors header (bytes read: the
    header only, never the data blocks)."""
    path = os.path.join(model_dir, shard)
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise Refused(f"{path}: truncated safetensors header")
        (hlen,) = struct.unpack("<Q", raw)
        if hlen <= 2 or hlen > (256 << 20):
            raise Refused(f"{path}: implausible safetensors header length {hlen}")
        header = json.loads(f.read(hlen))
    out = {}
    for name, entry in header.items():
        if name == "__metadata__" or not isinstance(entry, dict):
            continue
        out[name] = (entry.get("dtype"), [int(d) for d in entry.get("shape", [])])
    return out


def payload_estimate(model_dir: str, weight_map: dict[str, str],
                     prefixes: list[str]) -> tuple[int, int]:
    """Exact payload byte count, metadata-only (the helper's formula).

    Mirrors the store's build: 256 reserved bytes, then every record 256-byte
    aligned.  Also runs the same per-projection source validation the store does
    (complete triples, mul1 marker, no bias, I16 trellis / F16 scales at an
    H128-multiple geometry, verified rates K2/K2.5/K3/K4), so a non-packable model
    is refused BEFORE the helper is invoked.
    """
    shard_meta: dict[str, dict] = {}

    def meta(name: str) -> tuple[str, list[int]] | None:
        shard = weight_map.get(name)
        if shard is None:
            return None
        if shard not in shard_meta:
            try:
                shard_meta[shard] = shard_tensor_meta(model_dir, shard)
            except Refused as e:
                raise Refused(str(e))
        return shard_meta[shard].get(name)

    total = RECORD_ALIGN  # the store reserves the first 256 payload bytes
    for p in prefixes:
        for suffix in (".trellis", ".suh", ".svh"):
            if (p + suffix) not in weight_map:
                raise Refused(f"{p + suffix} is not in the model index "
                              f"(all requested triples must be complete)")
        if (p + ".bias") in weight_map:
            raise Refused(f"{p}: biased EXL3 linears are not supported")
        if (p + ".mul1") not in weight_map:
            raise Refused(f"{p} lacks the mul1 marker (the packed store is mul1-only)")
        tr = meta(p + ".trellis")
        suh = meta(p + ".suh")
        svh = meta(p + ".svh")
        if tr is None or suh is None or svh is None:
            raise Refused(f"{p} is indexed but missing from its shard (corrupt source)")
        dtype, shape = tr
        if dtype != "I16" or len(shape) != 3 or shape[0] <= 0 or shape[1] <= 0 \
                or shape[0] % 8 or shape[1] % 8 or shape[2] < 16 or shape[2] > 128 \
                or shape[2] % 8:
            raise Refused(f"{p}: need I16 trellis [in/16,out/16,16*K] with H128 dims "
                          f"(got dtype {dtype}, shape {shape})")
        ki, nj, last = shape
        bits, frac = last // 16, (last % 16) // 8
        if (bits, frac) not in VERIFIED_RATES:
            raise Refused(f"{p}: rate K={bits + 0.5 * frac} is outside the verified set "
                          f"{{2, 2.5, 3, 4}}")
        if suh[0] != "F16" or svh[0] != "F16" or suh[1] != [ki * 16] or svh[1] != [nj * 16]:
            raise Refused(f"{p}: need F16 suh[in] and svh[out]")
        trellis_bytes = ki * nj * (16 * bits + 8 * frac) * 2
        suh_bytes = ki * 32
        svh_bytes = nj * 32
        for n in (trellis_bytes, suh_bytes, svh_bytes):
            total += (n + RECORD_ALIGN - 1) // RECORD_ALIGN * RECORD_ALIGN
    return total, len(prefixes) * 3


# ---- manifest.bin parsing (the store's own binary format, read-only) ------------------------
def parse_manifest(manifest_path: str) -> dict:
    """Structural parse of a generation's manifest.bin: header + records + names."""
    data = Path(manifest_path).read_bytes()
    if len(data) < MANIFEST_HEADER_BYTES:
        raise Refused(f"{manifest_path}: shorter than its header")
    if data[:7] != MANIFEST_MAGIC or data[7] != BAND8_FORMAT_VERSION:
        raise Refused(f"{manifest_path}: magic/format not recognized "
                      f"(band8 format version {data[7]})")
    (manifest_version, flags, fingerprint, record_count, payload_bytes, payload_fnv,
     pool_bytes) = struct.unpack_from("<IIQQQQQ", data, 8)
    if manifest_version != MANIFEST_VERSION:
        raise Refused(f"{manifest_path}: unsupported manifest version {manifest_version}")
    need = MANIFEST_HEADER_BYTES + record_count * MANIFEST_RECORD_BYTES
    if need + pool_bytes != len(data):
        raise Refused(f"{manifest_path}: size does not match its counts")
    records = []
    for i in range(record_count):
        off = MANIFEST_HEADER_BYTES + i * MANIFEST_RECORD_BYTES
        r_offset, r_bytes, name_off, name_len = struct.unpack_from("<QQIH", data, off)
        kind, bits, frac = data[off + 22], data[off + 23], data[off + 24]
        ki, nj = struct.unpack_from("<II", data, off + 28)
        base = need + name_off
        name = data[base:base + name_len]
        if len(name) != name_len or data[base + name_len] != 0:
            raise Refused(f"{manifest_path}: record name outside the pool")
        records.append({
            "name": name.decode("utf-8"), "offset": r_offset, "bytes": r_bytes,
            "kind": kind, "bits": bits, "frac": frac, "ki": ki, "nj": nj,
        })
    return {"fingerprint": f"{fingerprint:016x}", "record_count": record_count,
            "payload_bytes": payload_bytes, "flags": flags, "payload_fnv": payload_fnv,
            "records": records}


def check_generation(cache_root: str, generation: str, expect_fingerprint: str,
                     expect_payload: int, expect_records: int,
                     expect_names: frozenset[str] | None = None) -> dict:
    """Validate cache_root/<generation>/ the way the engine's reuse path does, plus
    the strict routed-expert-only name contract (dense/ngram/MTP never pack)."""
    gen = Path(cache_root) / generation
    payload = gen / "payload.bin"
    manifest = gen / "manifest.bin"
    if not manifest.is_file():
        raise Refused(f"{manifest} is missing - the generation is incomplete "
                      f"(rerun the repack; nothing is published from a partial build)")
    if not payload.is_file():
        raise Refused(f"{payload} is missing while manifest.bin exists")
    m = parse_manifest(str(manifest))
    size = payload.stat().st_size
    if m["fingerprint"] != expect_fingerprint:
        raise Refused(f"{manifest}: fingerprint {m['fingerprint']} does not name this "
                      f"generation {expect_fingerprint} (moved or hand-edited)")
    if m["payload_bytes"] != size:
        raise Refused(f"{payload}: file size {size} != manifest {m['payload_bytes']} "
                      f"(truncated or damaged payload)")
    if size != expect_payload:
        raise Refused(f"{payload}: size {size} != the metadata estimate {expect_payload}")
    if m["record_count"] != expect_records:
        raise Refused(f"{manifest}: {m['record_count']} records != expected {expect_records}")
    names = [r["name"] for r in m["records"]]
    bad = [n for n in names if not STRICT_EXPERT_TENSOR.match(n)]
    if bad:
        sample = ", ".join(sorted(bad)[:3])
        raise Refused(f"{manifest}: records outside the routed-expert contract ({sample}...) - "
                      f"only trunk mlp.experts gate/up/down trellis/suh/svh may pack")
    if expect_names is not None and frozenset(names) != expect_names:
        missing = len(expect_names - frozenset(names))
        extra = len(frozenset(names) - expect_names)
        raise Refused(f"{manifest}: record names do not match this model's routed experts "
                      f"({missing} missing, {extra} extra) - wrong source model?")
    return m


# ---- descriptor ------------------------------------------------------------------------------
def load_descriptor(path: str) -> dict:
    p = Path(path)
    if not p.is_file():
        raise Refused(f"{p}: no such artifact descriptor (expected the .strata-experts.json "
                      f"file a repack wrote)")
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise Refused(f"{p}: malformed descriptor ({e})")
    if not isinstance(d, dict):
        raise Refused(f"{p}: malformed descriptor (not a JSON object)")
    unknown = sorted(set(d) - DESCRIPTOR_KEYS)
    if unknown:
        raise Refused(f"{p}: unknown descriptor key(s) {unknown} - written by another tool "
                      f"version? Refusing instead of guessing")
    missing = sorted(DESCRIPTOR_KEYS - set(d))
    if missing:
        raise Refused(f"{p}: descriptor is missing key(s) {missing} - truncated or "
                      f"hand-edited; rerun the repack")
    if d["schema"] != SCHEMA or d["schema_version"] != SCHEMA_VERSION:
        raise Refused(f"{p}: descriptor schema {d['schema']}/{d['schema_version']} is not "
                      f"{SCHEMA}/{SCHEMA_VERSION}")
    if d["family"] not in FAMILIES:
        raise Refused(f"{p}: unknown family {d['family']!r}")
    # Complete type validation: every field is checked BEFORE any consumer touches it,
    # so a hand-edited descriptor can never surface as a TypeError traceback.
    string_keys = ("created_utc", "tool", "model_dir", "model_dir_canonical", "cache_root")
    for key in string_keys:
        if not isinstance(d[key], str) or not d[key] or "\x00" in d[key] or "\n" in d[key]:
            raise Refused(f"{p}: {key} must be a non-empty single-line string")
    for key in ("cache_root", "model_dir_canonical"):
        if not os.path.isabs(d[key]):
            raise Refused(f"{p}: {key} must be an absolute path (got {d[key]!r})")
    for key in ("generation", "fingerprint"):
        if not isinstance(d[key], str) or not HEX16.match(d[key]):
            raise Refused(f"{p}: {key} must be 16 lowercase hex digits")
    if d["generation"] != d["fingerprint"]:
        raise Refused(f"{p}: generation {d['generation']} != fingerprint {d['fingerprint']}")
    for key in ("record_count", "payload_bytes", "expected_payload_bytes",
                "expert_prefixes", "moe_layers", "experts_per_layer"):
        if not isinstance(d[key], int) or isinstance(d[key], bool) or d[key] <= 0:
            raise Refused(f"{p}: {key} must be a positive integer")
    if d["record_count"] != d["expert_prefixes"] * 3:
        raise Refused(f"{p}: record_count != expert_prefixes * 3")
    if d["payload_bytes"] != d["expected_payload_bytes"]:
        raise Refused(f"{p}: payload_bytes != expected_payload_bytes (the generation was "
                      f"not built from this descriptor's estimate)")
    env = d["engine_env"]
    if not isinstance(env, dict) or set(env) != {
            FAMILY_ENV_SWITCH[d["family"]], "STRATA_EXL3_BAND8_CACHE"}:
        raise Refused(f"{p}: engine_env must select exactly this family's band8 switch plus "
                      f"STRATA_EXL3_BAND8_CACHE")
    if env[FAMILY_ENV_SWITCH[d["family"]]] != "1":
        raise Refused(f"{p}: engine_env[{FAMILY_ENV_SWITCH[d['family']]}] must be \"1\"")
    for key, value in env.items():
        if not isinstance(key, str) or not isinstance(value, str) or not value \
                or "\n" in value or "\x00" in value:
            raise Refused(f"{p}: engine_env[{key!r}] must be a non-empty single-line string")
    if not os.path.isabs(env["STRATA_EXL3_BAND8_CACHE"]):
        raise Refused(f"{p}: engine_env[STRATA_EXL3_BAND8_CACHE] must be an absolute path")
    if env["STRATA_EXL3_BAND8_CACHE"] != d["cache_root"]:
        raise Refused(f"{p}: engine_env[STRATA_EXL3_BAND8_CACHE] "
                      f"{env['STRATA_EXL3_BAND8_CACHE']!r} does not match cache_root "
                      f"{d['cache_root']!r} - hand-edited descriptor; rerun the repack")
    src = d["source"]
    if not isinstance(src, dict):
        raise Refused(f"{p}: source must be an object")
    for key in ("index_bytes", "index_mtime_ns"):
        if not isinstance(src.get(key), int) or isinstance(src.get(key), bool) \
                or src[key] < 0:
            raise Refused(f"{p}: source.{key} must be a non-negative integer")
    if not isinstance(src.get("shards"), list) or not src["shards"] or \
            not all(isinstance(s, str) and s and "\n" not in s and "\x00" not in s
                    for s in src["shards"]):
        raise Refused(f"{p}: source.shards must be a non-empty list of shard file names")
    return d


def publish_descriptor(output: Path, descriptor: dict, force: bool) -> None:
    """Publish the descriptor atomically without a TOCTOU window.

    Without --force the publish is NO-CLOBBER by construction: the temp file is
    HARDLINKED to the destination (same filesystem, atomic create-or-fail), so a
    descriptor that appeared after any earlier existence check makes the link fail
    with EEXIST and nothing is replaced - the earlier ``os.path.exists`` check is
    only a fast, friendly pre-check, never the guard.  With --force the link target
    is replaced by rename (the explicit user intent).  The parent directory is
    fsynced so the published name survives a crash, as far as the filesystem
    promises it for the payload it points at.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not force:
        raise Refused(f"{output} already exists - pass --force to replace it explicitly "
                      f"(the packed payload in {output.with_name(output.name[:-5])} is "
                      f"fingerprint-addressed and is not touched by a replace)")
    text = json.dumps(descriptor, indent=2, sort_keys=False) + "\n"
    fd, tmp = tempfile.mkstemp(dir=str(output.parent), prefix=output.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        if force:
            os.replace(tmp, output)
        else:
            try:
                os.link(tmp, output)   # atomic no-clobber: EEXIST if any race won
            except FileExistsError:
                raise Refused(f"{output} already exists (created concurrently) - not "
                              f"replaced; pass --force to replace it explicitly")
            os.unlink(tmp)
        dir_fd = os.open(str(output.parent), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def helper_candidates(root: Path) -> list[Path]:
    out = []
    env = os.environ.get("STRATA_EXPERTS_HELPER")
    if env:
        out.append(Path(env))
    out.append(root / "build-hip-exl3" / "exl3_band8_build")
    out.append(root / "build" / "exl3_band8_build")
    return out


def find_helper(root: Path, explicit: str | None) -> str:
    if explicit:
        if not (os.path.isfile(explicit) and os.access(explicit, os.X_OK)):
            raise Refused(f"--helper {explicit} is not an executable file")
        return explicit
    for cand in helper_candidates(root):
        if cand.is_file() and os.access(cand, os.X_OK):
            return str(cand)
    raise Refused(
        "the CPU-only repack helper exl3_band8_build was not found (looked at "
        "$STRATA_EXPERTS_HELPER, build-hip-exl3/exl3_band8_build, build/exl3_band8_build). "
        "Build it with:  cmake --build build-hip-exl3 --target exl3_band8_build "
        "(plain CMake target, no GPU needed)")


def assets_dir_of(output: Path) -> Path:
    """The managed asset directory adjacent to the descriptor: name.strata-experts."""
    name = output.name
    if not name.endswith(OUTPUT_SUFFIX):
        raise Refused(f"--output must end with {OUTPUT_SUFFIX} (for example "
                      f"/mnt/bigssd/my-model.strata-experts.json); got {output}")
    return output.with_name(name[: -len(".json")])


def existing_generations(cache_root: Path) -> list[tuple[str, int]]:
    out = []
    if cache_root.is_dir():
        for child in sorted(cache_root.iterdir()):
            if HEX16.match(child.name) and child.is_dir():
                payload = child / "payload.bin"
                try:
                    out.append((child.name, payload.stat().st_size))
                except OSError:
                    out.append((child.name, -1))
    return out


def gig(n: int) -> str:
    return f"{n / GIB:.2f} GiB ({n:,} B)"


# ---- resident-arena arithmetic (the runtime in-memory repack's RAM budget) -------------------
def expert_plane_sizes(model_dir: str, weight_map: dict[str, str],
                       prefixes: list[str]) -> list[tuple[int, int, int]]:
    """(trellis, suh, svh) byte sizes per projection, metadata-only.

    Mirrors Exl3ResidentBacking::create's per-plane expectation
    (ki*nj*16K*2, ki*32, nj*32).  Call payload_estimate() first for the full
    source validation (mul1, verified rates, dtype/shape contract); this helper
    only re-reads the same safetensors headers to size the arena.
    """
    shard_meta: dict[str, dict] = {}

    def meta(name: str) -> tuple[str, list[int]] | None:
        shard = weight_map.get(name)
        if shard is None:
            return None
        if shard not in shard_meta:
            shard_meta[shard] = shard_tensor_meta(model_dir, shard)
        return shard_meta[shard].get(name)

    out = []
    for p in prefixes:
        dtype, shape = meta(p + ".trellis")
        ki, nj = shape[0], shape[1]
        out.append(((ki * nj * shape[2]) * 2, ki * 32, nj * 32))
    return out


def arena_estimate(model_dir: str, weight_map: dict[str, str],
                   prefixes: list[str]) -> int:
    """The resident RAM arena's exact byte total for these experts.

    Mirrors Exl3ResidentBacking::create: each expert's nine planes are laid out
    consecutively and the expert span rounds up to one 4096-byte page; the arena
    is the sum of the spans.  This is what --repack-in-memory pins at startup
    (before the engine's own 4 GiB lock headroom is applied).
    """
    sizes = expert_plane_sizes(model_dir, weight_map, prefixes)
    total = 0
    for i in range(0, len(sizes), 3):   # three projections per expert, create() order
        at = sum(n for triple in sizes[i:i + 3] for n in triple)
        total += (at + 4095) // 4096 * 4096
    return total


# ---- subcommands -----------------------------------------------------------------------------
def cmd_repack(args, root: Path) -> int:
    if sys.platform.startswith("win"):
        raise Refused("the packed-expert store needs POSIX mmap - Windows is not supported "
                      "(the engine refuses it too; documented limit)")
    model_dir = str(Path(args.model).resolve()) if args.model else ""
    if not model_dir or not os.path.isdir(model_dir):
        raise Refused(f"--model {args.model!r} is not a directory")
    output = Path(args.output).expanduser()
    if not output.is_absolute():
        output = Path.cwd() / output
    assets = assets_dir_of(output)
    cache_root = Path(args.cache_root).expanduser() if args.cache_root else assets
    if not cache_root.is_absolute():
        cache_root = Path.cwd() / cache_root
    if cache_root.exists() and not cache_root.is_dir():
        raise Refused(f"{cache_root} exists and is not a directory")

    weight_map = read_weight_map(model_dir)
    family, geometry = detect_family(model_dir, weight_map)
    if args.family != "auto":
        want = {"mimo": "mimo_v2", "qwen": "qwen4exp"}[args.family]
        if family != want:
            raise Refused(f"--family {args.family} refused: {model_dir} detects as {family}")
    prefixes, summary = expert_prefixes(family, geometry)
    expected_payload, record_count = payload_estimate(model_dir, weight_map, prefixes)
    fingerprint = compute_fingerprint(model_dir, prefixes, weight_map)
    generation_dir = cache_root / fingerprint
    index_stat = os.stat(os.path.join(model_dir, "model.safetensors.index.json"))
    shards = sorted({weight_map[p + s] for p in prefixes for s in (".trellis", ".suh", ".svh")})

    reuse = False
    if (generation_dir / "manifest.bin").is_file():
        try:
            check_generation(str(cache_root), fingerprint, fingerprint,
                             expected_payload, record_count,
                             frozenset(p + s for p in prefixes
                                       for s in (".trellis", ".suh", ".svh")))
            reuse = True
        except Refused as e:
            print(f"strata-experts: existing generation {fingerprint} is NOT reusable "
                  f"({e}); the helper will rebuild it in place", file=sys.stderr)

    manifest_estimate = MANIFEST_HEADER_BYTES + record_count * (MANIFEST_RECORD_BYTES + 32)
    needed = 0 if reuse else expected_payload + manifest_estimate + (64 << 20)
    probe = cache_root if cache_root.exists() else cache_root.parent
    while not probe.exists():
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    others = existing_generations(cache_root)

    print(f"family:            {family}")
    print(f"model:             {model_dir}")
    print(f"routed experts:    {summary['moe_layers']} MoE layer(s) x "
          f"{summary['experts_per_layer']} experts x 3 projections "
          f"= {len(prefixes):,} triples ({record_count:,} plane records)")
    print(f"payload estimate:  {gig(expected_payload)} exact (metadata-only)")
    print(f"descriptor:        {output}  (~1 KB)")
    print(f"packed assets:     {cache_root}/{fingerprint}/  (managed; the payload above "
          f"plus a ~{manifest_estimate / 1e6:.1f} MB manifest)")
    if others:
        for name, size in others:
            tag = " (this repack's generation)" if name == fingerprint else ""
            print(f"existing generation there: {name}  {gig(size) if size >= 0 else 'size unknown'}{tag}")
    if reuse:
        print("reuse:             YES - a matching generation is already valid; the helper "
              f"will open it (no new bytes written)")
    else:
        print(f"free space:        {gig(free)} on {probe}; this repack needs {gig(needed)}"
              f"{'  -> NOT ENOUGH' if free < needed else ''}")
    if free < needed:
        raise Refused(f"not enough free space for the packed payload: need {needed:,} bytes, "
                      f"have {free:,} (missing {needed - free:,}) - point --output (or "
                      f"--cache-root) at a drive that fits it")
    if args.dry_run:
        print("dry run:           nothing written, helper not invoked")
        return 0
    if output.exists() and not args.force:
        raise Refused(f"{output} already exists - pass --force to replace it explicitly")

    helper = find_helper(root, args.helper)
    cmd = [helper, "--model", model_dir, "--cache", str(cache_root)]
    if family == "qwen4exp":
        cmd.append("--qwen")
    if args.progress:
        cmd += ["--progress", str(args.progress)]
    print(f"helper:            {' '.join(cmd)}", file=sys.stderr)
    t0 = time.monotonic()
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        raise Refused(f"the repack helper exited with {rc} - no descriptor published "
                      f"(a partial payload, if any, stays in {generation_dir} and is "
                      f"repaired or reused by the next run)")

    m = check_generation(str(cache_root), fingerprint, fingerprint,
                         expected_payload, record_count,
                         frozenset(p + s for p in prefixes
                                   for s in (".trellis", ".suh", ".svh")))
    if m["record_count"] != record_count:
        raise Refused(f"helper produced {m['record_count']} records, expected {record_count}")
    descriptor = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "family": family,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tool": TOOL,
        "model_dir": str(Path(args.model).expanduser()),
        "model_dir_canonical": canonical_path(model_dir),
        "cache_root": str(cache_root),
        "generation": fingerprint,
        "fingerprint": fingerprint,
        "record_count": record_count,
        "payload_bytes": expected_payload,
        "expected_payload_bytes": expected_payload,
        "expert_prefixes": len(prefixes),
        "moe_layers": summary["moe_layers"],
        "experts_per_layer": summary["experts_per_layer"],
        "source": {
            "index_bytes": index_stat.st_size,
            "index_mtime_ns": index_stat.st_mtime_ns,
            "shards": shards,
        },
        "engine_env": {
            FAMILY_ENV_SWITCH[family]: "1",
            "STRATA_EXL3_BAND8_CACHE": str(cache_root),
        },
    }
    publish_descriptor(output, descriptor, args.force)
    elapsed = time.monotonic() - t0
    print(f"published {output} ({'reused' if reuse else 'packed'} generation {fingerprint}, "
          f"{gig(expected_payload)} payload, {elapsed:.1f} s)")
    if family == "qwen4exp":
        print(f"launch with:       python tools/strata_experts.py launch --config <a config "
              f"serving this model> --experts {output} --single-gpu")
        print("                   (packed Qwen is validated single GPU + CPU tier; the peer "
              "tier stays native/experimental - the launch layer refuses it)")
    else:
        print(f"launch with:       python tools/strata_experts.py launch --config <a config "
              f"serving this model> --experts {output}")
    print("                   (a config can be made with: strata_experts.py config --model "
          "<dir> --tokenizer <pack> --output cfg.json; the launch's auto performance "
          "profile is on by default, --performance baseline opts out)")
    return 0


def validate_descriptor(descriptor_path: str, model_dir: str | None,
                        names_expected: bool) -> dict:
    d = load_descriptor(descriptor_path)
    prefixes = None
    model_state = None
    if model_dir:
        if not os.path.isdir(model_dir):
            raise Refused(f"--model {model_dir!r} is not a directory")
        weight_map = read_weight_map(model_dir)
        family, geometry = detect_family(model_dir, weight_map)
        if family != d["family"]:
            raise Refused(f"family mismatch: the descriptor packs {d['family']}, but "
                          f"{model_dir} detects as {family} - wrong artifact for this model")
        prefixes, _ = expert_prefixes(family, geometry)
        expected_payload, record_count = payload_estimate(model_dir, weight_map, prefixes)
        if record_count != d["record_count"] or expected_payload != d["expected_payload_bytes"]:
            raise Refused(
                f"source mismatch: {model_dir} now measures {record_count} records / "
                f"{expected_payload:,} bytes, the descriptor says {d['record_count']} / "
                f"{d['expected_payload_bytes']:,} - the model changed since the repack "
                f"(rerun the repack; do NOT launch: the engine would build a new "
                f"{gig(expected_payload)} generation)")
        fingerprint = compute_fingerprint(model_dir, prefixes, weight_map)
        model_state = (family, fingerprint, weight_map)
        if fingerprint != d["fingerprint"]:
            index_changed = False
            try:
                st = os.stat(os.path.join(model_dir, "model.safetensors.index.json"))
                index_changed = (st.st_size != d["source"]["index_bytes"]
                                 or st.st_mtime_ns != d["source"]["index_mtime_ns"])
            except OSError:
                index_changed = True
            detail = "the index changed" if index_changed else \
                "a shard's size/mtime or the model's path changed"
            raise Refused(
                f"stale artifact: {model_dir} fingerprints as {fingerprint} but the "
                f"descriptor was packed from {d['fingerprint']} ({detail}). The packed "
                f"generation does not match this source - rerun the repack. (The generation "
                f"is bound to the model's canonical path and file metadata; a moved model "
                f"dir needs a fresh repack by design.)")
    expect_names = None
    if names_expected and prefixes is not None:
        expect_names = frozenset(p + s for p in prefixes
                                 for s in (".trellis", ".suh", ".svh"))
    check_generation(d["cache_root"], d["generation"], d["fingerprint"],
                     d["payload_bytes"], d["record_count"], expect_names)
    return d


def cmd_validate(args, root: Path) -> int:
    d = validate_descriptor(args.descriptor, args.model, names_expected=True)
    payload = Path(d["cache_root"]) / d["generation"] / "payload.bin"
    print(f"OK {args.descriptor}")
    print(f"  family {d['family']}, {d['record_count']:,} plane records, "
          f"payload {gig(d['payload_bytes'])}")
    print(f"  generation {d['generation']} at {payload}")
    if args.model:
        print(f"  source model {args.model} matches (fingerprint {d['fingerprint']})")
    print("  routed experts only - dense, ngram, MTP and drafter tensors stay in the "
          "original model dir")
    return 0


def cmd_env(args, root: Path) -> int:
    d = validate_descriptor(args.descriptor, args.model, names_expected=False)
    for key in sorted(d["engine_env"]):
        value = d["engine_env"][key]
        if "\n" in value or "=" in key:
            raise Refused("refusing to emit a malformed environment line")
        print(f"{key}={value}")


# ---- launch: the tracked, portable selection interface ---------------------------------------
PERFORMANCE_PROFILES = ("auto", "baseline")

# Capability probes.  auto only arms a profile it can actually attribute to this machine;
# any uncertainty keeps the config unchanged with a logged reason.  Both probes accept an
# explicit override env (also the test seam, so tests never depend on the host hardware):
#   STRATA_EXPERTS_GFX11=1|0     force the gfx11 (RDNA3) verdict
#   STRATA_EXPERTS_AVX512BW=1|0  force the AVX-512BW verdict
RDNA3_PRODUCT_MARKS = ("7900 xtx", "7900 xt", "7900 gre", "7800 xt", "7700 xt",
                       "rx 7600", "7600 xt", "7600m xt", "7600s", "7700s",
                       "w7900", "w7800", "w7700", "w7600")


def _rocminfo_gfx_names(rocminfo: str) -> list[str]:
    """The gfx names of rocminfo's GPU-type agents only (CPU/DSP agents excluded)."""
    agents, current = [], {}
    for line in subprocess.run([rocminfo], capture_output=True, text=True,
                               timeout=30).stdout.splitlines():
        line = line.strip()
        if line.startswith("Agent "):
            if current.get("gpu") and current.get("name"):
                agents.append(current["name"])
            current = {}
        elif line.startswith("Device Type:"):
            current["gpu"] = line.endswith("GPU")
        elif line.startswith("Name:"):
            current["name"] = line.split(":", 1)[1].strip()
    if current.get("gpu") and current.get("name"):
        agents.append(current["name"])
    return agents


def _sysfs_gpu_evidence() -> list[tuple[str, str]]:
    """(vendor, marketing name) per DRM card that reports one, for the sysfs fallback."""
    out = []
    for card in sorted(Path("/sys/class/drm").glob("card[0-9]*/device/product_name")):
        try:
            out.append(("amd", card.read_text(encoding="utf-8",
                                              errors="replace").strip().lower()))
        except OSError:
            continue
    return out


def _nvidia_display_present() -> bool:
    for vendor in sorted(Path("/sys/bus/pci/devices").glob("*/vendor")):
        try:
            if vendor.read_text().strip() == "0x10de":
                return True
        except OSError:
            continue
    return False


def probe_gfx11() -> bool | None:
    """True/False when the machine's GPU set is CONFIRMED/DENIED all-gfx11, None unknown.

    Conservative by design: auto must not be authorized by an unrelated card (an iGPU or
    a second vendor's part) while the selected GPU is something else, and this layer
    cannot see which card the config will actually use.  So the rocminfo path requires
    EVERY GPU-type agent to report gfx11*; the sysfs fallback requires every DRM card
    that reports a product name to be an RDNA3 part AND no NVIDIA card present.  Any
    contrary evidence returns False (auto skips; STRATA_EXPERTS_GFX11=1 overrides on a
    setup the operator knows, e.g. a non-gfx11 card hidden from the engine).
    STRATA_EXPERTS_ROCMINFO=<path> points the probe at a specific rocminfo (tests).
    """
    forced = os.environ.get("STRATA_EXPERTS_GFX11", "")
    if forced == "1":
        return True
    if forced == "0":
        return False
    rocminfo = os.environ.get("STRATA_EXPERTS_ROCMINFO") or \
        shutil.which("rocminfo") or "/opt/rocm/bin/rocminfo"
    if os.path.isfile(rocminfo) and os.access(rocminfo, os.X_OK):
        try:
            names = _rocminfo_gfx_names(rocminfo)
            if names:
                return all(n.startswith("gfx11") for n in names)
        except (OSError, subprocess.SubprocessError):
            pass
    products = _sysfs_gpu_evidence()
    if products:
        if _nvidia_display_present():
            return False
        return all(any(mark in name for mark in RDNA3_PRODUCT_MARKS)
                   for _, name in products)
    if _nvidia_display_present():
        return False
    return None


def probe_avx512bw() -> bool | None:
    forced = os.environ.get("STRATA_EXPERTS_AVX512BW", "")
    if forced == "1":
        return True
    if forced == "0":
        return False
    try:
        flags = ""
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("flags"):
                    flags = line
                    break
    except OSError:
        return None
    if not flags:
        return None
    return " avx512bw " in flags or flags.rstrip().endswith(" avx512bw") \
        or " avx512bw\t" in flags


def cpu_worker_cap() -> int:
    """A conservative worker count for the bounded CPU tier: at most 8, at most half the
    logical CPUs (SMT siblings of real cores), never below 2.  Machine topology decides;
    the config's explicit value always wins."""
    return max(2, min(8, (os.cpu_count() or 4) // 2))


def load_server_config(path: str) -> dict:
    p = Path(path)
    if not p.is_file():
        raise Refused(f"--config {path}: no such server config file (any config serve/"
                      f"server.py takes: model, tokenizer and drafter paths are yours)")
    try:
        cfg = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise Refused(f"--config {path}: malformed JSON ({e})")
    if not isinstance(cfg, dict) or not isinstance(cfg.get("args"), list) or not cfg["args"]:
        raise Refused(f"--config {path}: not a server config (need an 'args' list with the "
                      f"engine arguments, at least --exl3 <model dir>)")
    for i, a in enumerate(cfg["args"]):
        if not isinstance(a, str):
            raise Refused(f"--config {path}: args[{i}] is not a string")
    return cfg


def config_exl3_dir(cfg: dict) -> str | None:
    args = cfg["args"]
    if "--exl3" in args:
        i = args.index("--exl3")
        if i + 1 >= len(args):
            raise Refused("config args end at '--exl3' with no model dir value")
        if args[i + 1].startswith("--"):
            raise Refused(f"config args have '--exl3 {args[i + 1]}' - the model dir value "
                          f"is missing")
        return args[i + 1]
    return None


PEER_ARG_FLAGS = ("--peer-device", "--peer-reserve-mib", "--peer-slots",
                  "--peer-adapt-swaps", "--peer-adapt-mib", "--peer-adapt-min-gain",
                  "--peer-prefill-rows")


def peer_armed_in_args(args: list[str]) -> bool:
    return any(a == "--peer-device" or a.startswith("--peer-device=") for a in args)


def strip_peer_from_args(args: list[str]) -> tuple[list[str], bool]:
    """Remove the whole documented --peer-* argument family (each takes a value);
    --peer-device also accepts the --peer-device=1 spelling.  True when something was
    removed.  A peer flag with a missing value refuses instead of silently consuming
    the next argument."""
    out, skip_next, removed = [], False, False
    for a in args:
        if skip_next:
            skip_next = False
            continue
        if a in PEER_ARG_FLAGS:
            skip_next = True
            removed = True
            continue
        if a.startswith("--peer-device="):
            removed = True
            continue
        if any(a.startswith(f + "=") for f in PEER_ARG_FLAGS):
            removed = True
            continue
        out.append(a)
    if skip_next:
        raise Refused(f"config args end at a {PEER_ARG_FLAGS[0]}-family flag with no "
                      f"value; refusing instead of consuming the next argument")
    return out, removed


def normalize_gpu_list(cfg: dict, config_path: str) -> list[int]:
    """The config's 'gpu' field as a list of ints; refuses shapes we cannot interpret."""
    gpus = cfg.get("gpu")
    if gpus is None:
        return []
    if isinstance(gpus, bool):
        gpus = None
    if isinstance(gpus, int):
        return [gpus]
    if isinstance(gpus, list) and gpus and all(isinstance(g, int) and not isinstance(g, bool)
                                               for g in gpus):
        return list(gpus)
    if isinstance(gpus, str):
        try:
            return [int(t) for t in gpus.split(",") if t != ""]
        except ValueError:
            pass
    raise Refused(f"--config {config_path}: cannot interpret the 'gpu' field {gpus!r} "
                  f"(expected an int, a list of ints, or '0,1'); fix the config first")


def explicit_env_value(cfg_env: dict, key: str) -> tuple[str, str] | None:
    """Where an explicit value for `key` comes from: the shell wins, then the config's
    own env block.  None means 'nobody set it' (the launch layer may then set it)."""
    shell = os.environ.get(key)
    if shell is not None:
        return ("shell", shell)
    if key in cfg_env:
        value = cfg_env[key]
        if not isinstance(value, str):
            raise Refused(f"config env[{key}] must be a string (got {type(value).__name__})")
        return ("config", value)
    return None


# The auto profiles: the COHERENT measured sets from the local launchers, translated
# key-for-key, minus everything this layer must never decide on its own (the packed
# backing and its keys, resident RAM arenas, the LFU/adapt cache preset, drafter/MTP
# settings).  Every key honors an explicit shell/config value; an explicit 0 skips the
# feature AND its dependents, with a logged explanation - never an override.
MIMO_AUTO_FLAGS = (
    # the frac-K fast stack (run-mimo-mopd.sh, measured on the gfx1100 pair)
    ("STRATA_EXL3_WMMA", "1"), ("STRATA_EXL3_EXPERT_BATCH", "1"),
    ("STRATA_EXL3_RDNA", "1"), ("STRATA_EXL3_FUSE", "3"),
    ("STRATA_EXL3_WMMA_STG_PAD", "1"),
    # decode/window machinery + native grouping (same launcher block)
    ("STRATA_MIMO_SWA_RING", "1"), ("STRATA_EXL3_PEER_ARENA", "1"),
    ("STRATA_MIMO_DECODE_DEVICE", "1"), ("STRATA_EXL3_RDNA_NATIVE", "1"),
    ("STRATA_EXL3_STAGE_WAKE_LIMIT", "1"), ("STRATA_MIMO_DECODE_HANDOFF", "1"),
    ("STRATA_EXL3_NATIVE_GROUPED", "1"), ("STRATA_EXL3_PEER_NATIVE_WIDTH4", "1"),
    ("STRATA_MIMO_WINDOW_HANDOFF", "1"),
    # audited attention fixes + the measured shared-input prefill width
    ("STRATA_MIMO_ATTN_SPLIT", "1"), ("STRATA_MIMO_QKV_SHARE", "1"),
    ("STRATA_MIMO_Q8_PREFILL_COLS", "8"),
)
MIMO_AUTO_CPU_FLAGS = (  # needs AVX-512BW; also arms the --cpu-tier engine arg
    ("STRATA_MIMO_CPU_TIER", "1"), ("STRATA_MIMO_HANDOFF_CPU", "1"),
    ("STRATA_MIMO_CPU_MIN_MISSES", "2"), ("STRATA_EXL3_CPU_MAX_PER_CALL", "2"),
    ("STRATA_EXL3_CPU_CACHE_MIB", "0"),
    # STRATA_MIMO_CPU_WORKERS is filled from the machine topology (see cpu_worker_cap)
)
MIMO_AUTO_WINDOW_FLAGS = (  # needs the CPU block armed + the mapped handoff
    ("STRATA_MIMO_WINDOW_CPU", "1"),
    ("STRATA_MIMO_WINDOW_CPU_MAX_GROUPS", "2"), ("STRATA_MIMO_WINDOW_CPU_MIN_GROUPS", "1"),
)
QWEN_AUTO_FLAGS = (  # primary stream, measured on gfx1100 (run-peer.sh preset set)
    ("STRATA_EXL3_RDNA_NATIVE", "1"), ("STRATA_EXL3_NATIVE_GROUPED", "1"),
    ("STRATA_EXL3_WMMA_STG_PAD", "1"), ("STRATA_EXL3_EXPERT_BATCH", "1"),
    ("STRATA_EXL3_PRIMARY_PACK_PLAN", "1"), ("STRATA_EXL3_PRIMARY_MULTIROW", "2"),
    ("STRATA_EXL3_STAGE_INLINE", "2"),
)
QWEN_AUTO_PEER_FLAGS = (  # only when the config arms a peer tier
    ("STRATA_EXL3_PEER_PREFILL", "1"), ("STRATA_EXL3_PEER_BATCH", "1"),
    ("STRATA_EXL3_PEER_PACK_PLAN", "1"), ("STRATA_EXL3_PEER_MULTIROW", "4"),
)
QWEN_AUTO_CPU_FLAGS = (  # needs AVX-512BW; the measured bounded native-CPU policy.
    # Qwen's CPU participation is the HYBRID singleton-miss quota (generate.cpp's
    # cpu_hybrid_*), NOT the MiMo pool's STRATA_EXL3_CPU_MAX_PER_CALL: quota 2 groups,
    # participating from the first miss (default 4), repack cache off, swizzle off.
    ("STRATA_EXL3_CPU_TIER", "1"),
    ("STRATA_EXL3_CPU_HYBRID", "2"),
    ("STRATA_EXL3_CPU_HYBRID_MIN_MISSES", "1"),
    ("STRATA_EXL3_CPU_CACHE_MIB", "0"),
    ("STRATA_EXL3_CPU_SWIZZLE", "0"),
    # STRATA_EXL3_CPU_HYBRID_PEER=1 joins only when a peer tier is armed (see below)
    # STRATA_EXL3_CPU_THREADS is filled from the machine topology (see cpu_worker_cap)
)


def apply_auto_profile(cfg, cfg_env, notes, family, model_dir, probe_env):
    """The capability-gated 'auto' profile.  Returns nothing; every decision is logged.

    Gates: cfg backend must be hip and gfx11 must be confirmed (probe or explicit
    override) - otherwise the config is left UNCHANGED with a logged reason (nothing is
    broadly enabled on CUDA or on an unverifiable GPU).  CPU assistance additionally
    needs AVX-512BW.  The packed backing, resident arenas, the LFU/adapt cache preset
    and every drafter/MTP setting are never touched here.
    """
    if str(cfg.get("backend", "")).lower() != "hip":
        notes.append(f"auto profile skipped: config backend "
                     f"{cfg.get('backend')!r} is not 'hip' (the measured flag sets are "
                     f"HIP/gfx11-only; nothing was changed)")
        return
    if probe_env["gfx11"] is not True:
        why = ("no gfx11 (RDNA3) evidence found (rocminfo/product name probes)"
               if probe_env["gfx11"] is None else "the gfx11 probe said no")
        notes.append(f"auto profile skipped: {why}; set STRATA_EXPERTS_GFX11=1 to "
                     f"override on a known-RDNA3 box. The config is unchanged")
        return
    if family is None and model_dir is not None and os.path.isdir(model_dir):
        try:
            weight_map = read_weight_map(model_dir)
            family, _ = detect_family(model_dir, weight_map)
        except (Refused, OSError):
            family = None

    def resolved(key):
        got = explicit_env_value(cfg_env, key)
        return None if got is None else got[1]

    def arm(keys):
        for key, value in keys:
            got = explicit_env_value(cfg_env, key)
            if got is None:
                cfg_env[key] = value
            else:
                cfg_env[key] = got[1]  # materialize the explicit value into the record
                if got[1] != value:
                    notes.append(f"{key} kept at {got[1]} ({got[0]} wins over auto)")

    if family == "mimo_v2":
        arm(MIMO_AUTO_FLAGS)
        # Explicit 0 on the CPU tier or its handoff service is a user opt-out: auto
        # declines the WHOLE CPU block - env keys, the --cpu-tier engine arg and the
        # window arm - instead of arming half a stack (or overwriting the 0).
        tier0 = resolved("STRATA_MIMO_CPU_TIER") == "0"
        handoff_cpu0 = resolved("STRATA_MIMO_HANDOFF_CPU") == "0"
        cpu_ok = probe_env["avx512bw"] is True and not tier0 and not handoff_cpu0
        if tier0:
            notes.append("CPU assistance skipped: STRATA_MIMO_CPU_TIER=0 is explicit - "
                         "no CPU tier env, no --cpu-tier arg")
        elif handoff_cpu0:
            notes.append("CPU assistance skipped: STRATA_MIMO_HANDOFF_CPU=0 is explicit "
                         "(the tier's service handoff) - no CPU tier env, no --cpu-tier "
                         "arg; unset it to let auto arm the measured CPU block")
        elif probe_env["avx512bw"] is not True:
            notes.append("CPU assistance skipped: no AVX-512BW on this CPU (probe or "
                         "STRATA_EXPERTS_AVX512BW=1|0); --cpu-tier NOT added")
        if cpu_ok:
            arm(MIMO_AUTO_CPU_FLAGS)
            workers = str(cpu_worker_cap())
            if resolved("STRATA_MIMO_CPU_WORKERS") is None:
                cfg_env["STRATA_MIMO_CPU_WORKERS"] = workers
                notes.append(f"STRATA_MIMO_CPU_WORKERS={workers} (half the logical CPUs, "
                             f"capped at 8 - machine topology; an explicit value wins)")
            if "--cpu-tier" not in cfg["args"]:
                cfg["args"] = list(cfg["args"]) + ["--cpu-tier"]
                notes.append("added --cpu-tier to the engine args (the CPU tier env "
                             "alone does not arm the MiMo tier)")
        # WINDOW_CPU only when every dependency can actually be satisfied: the CPU block
        # above armed the tier, the mapped handoff + the decode handoff are not
        # explicitly off, and no incompatible diagnostic switch is on.  An explicit 0
        # anywhere in the chain means the user declined - skip with an explanation,
        # never override.
        window_blockers = []
        if resolved("STRATA_MIMO_WINDOW_CPU") == "0":
            window_blockers.append("STRATA_MIMO_WINDOW_CPU=0 is explicit")
        if not cpu_ok and not window_blockers:
            if tier0 or handoff_cpu0:
                window_blockers.append("the CPU tier is explicitly off (the window arm "
                                       "requires it)")
            else:
                window_blockers.append("it requires the CPU tier (--cpu-tier), which "
                                       "this CPU cannot take (no AVX-512BW)")
        if resolved("STRATA_MIMO_WINDOW_HANDOFF") == "0":
            window_blockers.append("STRATA_MIMO_WINDOW_HANDOFF=0 is explicit - the window "
                                   "arm needs the mapped handoff")
        if resolved("STRATA_MIMO_DECODE_HANDOFF") == "0":
            window_blockers.append("STRATA_MIMO_DECODE_HANDOFF=0 is explicit - the "
                                   "measured window stack is coherent with it")
        for blocked in ("STRATA_MIMO_TIME", "STRATA_MIMO_HANDOFF_EAGER"):
            if resolved(blocked) not in (None, "0"):
                window_blockers.append(f"{blocked} is explicitly on, which selects the "
                                       f"diagnostic window branch (incompatible)")
        if window_blockers:
            notes.append("window-CPU skipped: " + "; ".join(window_blockers))
        else:
            arm(MIMO_AUTO_WINDOW_FLAGS)
    elif family == "qwen4exp":
        arm(QWEN_AUTO_FLAGS)
        if peer_armed_in_args(cfg["args"]):
            arm(QWEN_AUTO_PEER_FLAGS)
        # STRATA_EXL3_RESIDENT on the NATIVE path (the packed/artifact composition below
        # enforces =all on its own): an explicit shell/config value is the user choosing
        # file-backed (0), the full pinned arena (all) or the peer-complement arena (peer,
        # 2026-11-20).  Materialize it into the derived env - a launch layer that dropped
        # the choice would silently downgrade the memory mode.  The engine's own gate
        # (generate.cpp) stays the single validator of the value and its requirements.
        resident_sel = explicit_env_value(cfg_env, "STRATA_EXL3_RESIDENT")
        if resident_sel is not None:
            cfg_env["STRATA_EXL3_RESIDENT"] = resident_sel[1]
            notes.append(f"STRATA_EXL3_RESIDENT={resident_sel[1]} ({resident_sel[0]} "
                         f"wins; the engine's resident gate validates the mode)")
        if resolved("STRATA_EXL3_CPU_TIER") == "0":
            notes.append("CPU assistance skipped: STRATA_EXL3_CPU_TIER=0 is explicit - "
                         "no hybrid/cache/swizzle/worker keys were added")
        elif probe_env["avx512bw"] is True:
            arm(QWEN_AUTO_CPU_FLAGS)
            if peer_armed_in_args(cfg["args"]):
                # STRATA_EXL3_RESIDENT=peer (RAM complement, 2026-11-20): the hybrid's
                # PEER arm computes peer-RESIDENT rows on the CPU, and under the
                # complement those experts have no host bytes (the engine refuses the
                # arm at startup).  Never arm it beside the complement, and treat an
                # explicit pairing as conflicting intent - the same rule as the packed
                # guards above.  The plain CPU tier stays available either way.
                resident_mode = explicit_env_value(cfg_env, "STRATA_EXL3_RESIDENT")
                hybrid_peer = explicit_env_value(cfg_env, "STRATA_EXL3_CPU_HYBRID_PEER")
                if resident_mode is not None and resident_mode[1] == "peer":
                    if hybrid_peer is not None and hybrid_peer[1] not in ("0", ""):
                        raise Refused(
                            f"STRATA_EXL3_CPU_HYBRID_PEER={hybrid_peer[1]} ({hybrid_peer[0]}) "
                            f"cannot run with STRATA_EXL3_RESIDENT=peer "
                            f"({resident_mode[0]}): the peer-complement arena keeps "
                            f"peer-owned experts off the host, so the CPU cannot compute "
                            f"their rows. Unset STRATA_EXL3_CPU_HYBRID_PEER (the plain "
                            f"CPU tier still works) or use STRATA_EXL3_RESIDENT=all. "
                            f"Nothing was started.")
                    notes.append("STRATA_EXL3_CPU_HYBRID_PEER skipped: "
                                 "STRATA_EXL3_RESIDENT=peer keeps peer-owned experts off "
                                 "the host (the plain CPU tier is unaffected)")
                elif resolved("STRATA_EXL3_CPU_HYBRID_PEER") is None:
                    cfg_env["STRATA_EXL3_CPU_HYBRID_PEER"] = "1"
                    notes.append("STRATA_EXL3_CPU_HYBRID_PEER=1 (the peer-complement "
                                 "rows are the supported CPU-assist surface with a peer "
                                 "armed; an explicit 0 wins)")
            threads = str(cpu_worker_cap())
            if resolved("STRATA_EXL3_CPU_THREADS") is None:
                cfg_env["STRATA_EXL3_CPU_THREADS"] = threads
                notes.append(f"STRATA_EXL3_CPU_THREADS={threads} (half the logical "
                             f"CPUs, capped at 8 - machine topology; explicit wins)")
        else:
            notes.append("CPU assistance skipped: no AVX-512BW on this CPU (probe or "
                         "STRATA_EXPERTS_AVX512BW=1|0)")
    else:
        notes.append("auto profile skipped: model family unknown (no --exl3 model dir "
                     "to detect, or detection refused); the config is unchanged")


def cmd_launch(args, root: Path) -> int:
    cfg = load_server_config(args.config)
    cfg_env = cfg.setdefault("env", {})
    if not isinstance(cfg_env, dict):
        raise Refused(f"--config {args.config}: the 'env' section must be an object")
    model_dir = config_exl3_dir(cfg)
    gpus = normalize_gpu_list(cfg, args.config)
    notes: list[str] = []

    # ---- the two packed-expert selectors are mutually exclusive; say it before either
    #      does any work (a descriptor validation error would otherwise mask the conflict)
    if args.repack_in_memory and args.experts:
        raise Refused("--repack-in-memory and --experts are alternative packed-expert "
                      "modes - pick one (the artifact reads a packed cache from disk; "
                      "--repack-in-memory builds the same bytes in RAM at startup)")

    # ---- --single-gpu: strip the peer tier and the later card(s) from the config
    if args.single_gpu:
        cfg["args"], removed = strip_peer_from_args(cfg["args"])
        if removed:
            notes.append("removed --peer-device from the engine args (--single-gpu)")
        if gpus:
            kept = gpus[:1]
            if kept != gpus:
                cfg["gpu"] = kept
                notes.append(f"config gpu {gpus} -> {kept} (--single-gpu)")
        for key in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
            v = cfg_env.get(key)
            if isinstance(v, str) and "," in v:
                cfg_env[key] = v.split(",")[0]
                notes.append(f"config env {key}={v} -> {cfg_env[key]} (--single-gpu)")

    # ---- --experts: validate the artifact against the model THIS config serves
    descriptor = None
    if args.experts:
        if model_dir is None:
            raise Refused(f"--config {args.config} has no --exl3 <model dir> argument to "
                          f"validate the artifact against")
        if not os.path.isdir(model_dir):
            raise Refused(f"--config {args.config} points at model dir {model_dir}, which "
                          f"does not exist (edit the config's --exl3 path first)")
        try:
            descriptor = validate_descriptor(args.experts, model_dir, names_expected=True)
        except Refused as e:
            raise Refused(f"--experts {args.experts} did not validate against {model_dir}:\n"
                          f"  {e}\nnot starting (rerun the repack for this model)")
        for key, value in descriptor["engine_env"].items():
            got = explicit_env_value(cfg_env, key)
            if got is not None and got[1] != value:
                raise Refused(f"--experts selects {key}={value} but the {got[0]} says "
                              f"{key}={got[1]} - conflicting intent; unset one of them")
            cfg_env[key] = value
        notes.append(f"packed routed experts from {args.experts} (generation "
                     f"{descriptor['generation']}, {descriptor['payload_bytes'] / 2**30:.2f} "
                     f"GiB payload; the model dir above stays required for dense, ngram, "
                     f"MTP and drafter tensors)")

    # ---- --repack-in-memory: the runtime RAM repack (no packed disk artifact) -----------
    # The alternative to a repacked artifact: at engine startup the routed experts are
    # permuted into the canonical band-8 order DIRECTLY in the pinned resident RAM arena.
    # No descriptor, no payload file, no cache root - the model dir stays the single
    # source.  Mutually exclusive with --experts and with every explicit band8/cache env
    # (a hidden disk build is refused, never silently armed).
    memory_repack_family = None
    if args.repack_in_memory:
        if model_dir is None:
            raise Refused(f"--config {args.config} has no --exl3 <model dir> argument to "
                          f"detect the family of (the runtime repack needs it)")
        if not os.path.isdir(model_dir):
            raise Refused(f"--config {args.config} points at model dir {model_dir}, which "
                          f"does not exist (edit the config's --exl3 path first)")
        for key in (FAMILY_ENV_SWITCH["mimo_v2"], FAMILY_ENV_SWITCH["qwen4exp"],
                    "STRATA_EXL3_BAND8_CACHE"):
            got = explicit_env_value(cfg_env, key)
            if got is not None:
                raise Refused(f"--repack-in-memory never builds or reads a packed disk "
                              f"cache, but the {got[0]} sets {key}={got[1]} - unset it (or "
                              f"drop --repack-in-memory for the cache mode)")
        weight_map = read_weight_map(model_dir)
        family, geometry = detect_family(model_dir, weight_map)
        prefixes, summary = expert_prefixes(family, geometry)
        payload_estimate(model_dir, weight_map, prefixes)   # the same per-projection validation
        arena = arena_estimate(model_dir, weight_map, prefixes)
        cfg_env[FAMILY_ENV_SWITCH[family]] = "memory"
        memory_repack_family = family
        notes.append(f"runtime in-memory repack ({family}): the routed experts are permuted "
                     f"into the canonical band-8 order straight into the pinned resident RAM "
                     f"arena at startup - no descriptor, no cache root, no packed disk file; "
                     f"the model dir stays the single source")
        notes.append(f"RAM expectation: about {gig(arena)} of routed-expert bytes pinned "
                     f"page-rounded per expert ({summary['moe_layers']} MoE layer(s) x "
                     f"{summary['experts_per_layer']} experts x 3 projections = "
                     f"{len(prefixes):,} triples), plus the engine's own guard budgets "
                     f"(4 GiB lock headroom, a 256 MiB exchange reserve, a leave-4-GiB floor "
                     f"during paced registration); the engine refuses a machine that cannot "
                     f"pin it - no packed expert files are written to disk")
        resident = explicit_env_value(cfg_env, "STRATA_EXL3_RESIDENT")
        if resident is not None and resident[1] != "all":
            raise Refused(f"the runtime repack needs STRATA_EXL3_RESIDENT=all but the "
                          f"{resident[0]} says {resident[1]} - conflicting intent; unset it "
                          f"or drop --repack-in-memory")
        if resident is None:
            cfg_env["STRATA_EXL3_RESIDENT"] = "all"
            notes.append("STRATA_EXL3_RESIDENT=all (required by the runtime repack: the "
                         "pinned arena is the single host copy and the only packed source)")

    # ---- dual-GPU packed Qwen guard (same rule the peer presets apply)
    family = descriptor["family"] if descriptor else memory_repack_family
    if family == "qwen4exp" and peer_armed_in_args(cfg["args"]):
        resident = explicit_env_value(cfg_env, "STRATA_EXL3_RESIDENT")
        resident_all = resident is None or resident[1] == "all"
        experimental_ok = (os.environ.get("STRATA_QWEN_BAND8_PEER_EXPERIMENTAL") == "1"
                           and resident_all)
        if not experimental_ok:
            raise Refused(
                "packed Qwen experts on two GPUs is EXPERIMENTAL: a deterministic "
                "first-token numeric divergence from the native twin is unresolved and no "
                "parity is claimed (notes/QWEN-BAND8-INTEGRATION-2026-11-08.md). The "
                "validated packed configuration is single GPU + CPU tier: pass "
                "--single-gpu (it strips --peer-device), or drop --experts / "
                "--repack-in-memory to stay native. To run the dual-GPU experiment anyway, "
                "set BOTH STRATA_QWEN_BAND8_PEER_EXPERIMENTAL=1 and "
                "STRATA_EXL3_RESIDENT=all (the engine requires the resident arena for this "
                "mode). Nothing was started and no cache byte was written.")

    # ---- consequences of selecting a Qwen artifact (the engine's own composition,
    #      armed here so the user does not have to know them).  Per key: the engine
    #      REQUIRES exactly RESIDENT=all (a different explicit value is a conflict),
    #      while the CPU tier is the validated default that an explicit 0 legitimately
    #      overrides (GPU-only packed run; the engine then needs no AVX-512BW).
    if family == "qwen4exp":
        resident = explicit_env_value(cfg_env, "STRATA_EXL3_RESIDENT")
        if resident is not None and resident[1] != "all":
            raise Refused(f"a Qwen packed artifact needs STRATA_EXL3_RESIDENT=all but the "
                          f"{resident[0]} says {resident[1]} - conflicting intent; unset it "
                          f"or drop --experts")
        if resident is None:
            cfg_env["STRATA_EXL3_RESIDENT"] = "all"
            notes.append("STRATA_EXL3_RESIDENT=all (required with the packed Qwen backing: "
                         "the engine needs the single canonical pinned RAM arena - 56.72 GiB "
                         "were locked in the 4 bpw single-GPU smoke - and its own memory "
                         "guard refuses a machine that cannot pin it)")
        tier = explicit_env_value(cfg_env, "STRATA_EXL3_CPU_TIER")
        if tier is not None and tier[1] not in ("0", "1"):
            raise Refused(f"STRATA_EXL3_CPU_TIER={tier[1]} ({tier[0]}) is not valid for the "
                          f"packed Qwen backing - use 1 (single GPU + CPU tier, the "
                          f"validated composition) or 0 (GPU-only packed)")
        if tier is None:
            cfg_env["STRATA_EXL3_CPU_TIER"] = "1"
            notes.append("STRATA_EXL3_CPU_TIER=1 (the validated packed composition is "
                         "single GPU + CPU tier; an explicit 0 in the shell or config env "
                         "keeps a GPU-only packed run instead)")
        elif tier[1] == "0":
            cfg_env["STRATA_EXL3_CPU_TIER"] = "0"
            notes.append("STRATA_EXL3_CPU_TIER kept at 0 "
                         f"({tier[0]}) - GPU-only packed run")

    # ---- --performance: 'auto' (the default) arms the coherent measured profile for
    #      the detected family where this machine supports it; 'baseline' is a genuine
    #      opt-out that leaves the config byte-for-byte as the user wrote it.
    if args.performance == "baseline":
        notes.append("baseline: the config is used exactly as written (no profile keys "
                     "added or changed)")
    else:
        apply_auto_profile(cfg, cfg_env, notes, family, model_dir,
                           {"gfx11": probe_gfx11(), "avx512bw": probe_avx512bw()})

    server = args.server or str(root / "serve" / "server.py")
    python = args.python or sys.executable
    for note in notes:
        print(f"[strata-experts] {note}", file=sys.stderr)
    if args.dry_run:
        command = [python, server, "--engine", "strata", "--config",
                   "<temporary derived config>", *args.server_args]
        print(json.dumps({"command": command, "config": cfg}, indent=2))
        print("[strata-experts] dry run: nothing started, nothing written", file=sys.stderr)
        return 0
    # The derived config is a UNIQUE temporary file (never a predictable shared name):
    # written once, passed to the server, removed when the server exits - no clobber
    # window between concurrent launches and no stale file left behind.  The server is
    # a CHILD process: SIGTERM/SIGINT to this wrapper are forwarded and the child is
    # reaped before cleanup, so a stopped wrapper never orphans a running server.
    run_dir = tempfile.mkdtemp(prefix="strata-experts-launch.")
    try:
        derived = Path(run_dir) / "config.json"
        derived.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
        command = [python, server, "--engine", "strata", "--config", str(derived),
                   *args.server_args]
        print(f"[strata-experts] derived config: {derived}", file=sys.stderr)
        import signal as _signal
        proc = subprocess.Popen(command)
        forwarded = {}

        def _forward(signum, _frame):
            proc.send_signal(signum)

        for sig in (_signal.SIGTERM, _signal.SIGINT):
            try:
                forwarded[sig] = _signal.signal(sig, _forward)
            except (ValueError, OSError):
                pass  # not the main thread: the default disposition already covers it
        try:
            return proc.wait()
        except KeyboardInterrupt:
            proc.send_signal(_signal.SIGINT)
            return proc.wait()
        finally:
            for sig, old in forwarded.items():
                try:
                    _signal.signal(sig, old)
                except (ValueError, OSError):
                    pass
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


# ---- config: create a small portable server config -------------------------------------------
def cmd_config(args, root: Path) -> int:
    """Write a minimal, complete server config the launch command (and serve/server.py)
    accepts - so nobody has to invent the required fields by hand."""
    model_dir = str(Path(args.model).expanduser().resolve())
    if not os.path.isdir(model_dir):
        raise Refused(f"--model {args.model!r} is not a directory")
    output = Path(args.output).expanduser()
    if not output.is_absolute():
        output = Path.cwd() / output
    if output.exists() and not args.force:
        raise Refused(f"{output} already exists - pass --force to replace it explicitly")
    try:
        gpus = [int(t) for t in args.gpu.split(",") if t != ""]
    except ValueError:
        raise Refused(f"--gpu {args.gpu!r}: expected a comma list of device numbers "
                      f"(for example 0 or 0,1)")
    weight_map = read_weight_map(model_dir)
    family, _ = detect_family(model_dir, weight_map)
    if args.family != "auto":
        want = {"mimo": "mimo_v2", "qwen": "qwen4exp"}[args.family]
        if family != want:
            raise Refused(f"--family {args.family} refused: {model_dir} detects as {family}")
    engine = args.engine or str(root / "build-hip-exl3" / "strata")
    if not (os.path.isfile(engine) and os.access(engine, os.X_OK)):
        print(f"[strata-experts] note: engine {engine} is not an executable file yet - "
              f"build it or edit the 'exe' field (--engine overrides)", file=sys.stderr)
    engine_args = ["--exl3", model_dir]
    if args.peer_device is not None:
        engine_args += ["--peer-device", str(args.peer_device)]
    if family == "mimo_v2":
        engine_args += (["--dense-q8"] if args.dense_q8 else []) + \
                       (["--cpu-tier"] if args.cpu_tier else [])
        if args.drafter:
            engine_args += ["--dflash", str(Path(args.drafter).expanduser())]
    else:
        for flag, meaning in (("--dense-q8", "dense q8_0"), ("--cpu-tier", "the engine arg")):
            if (flag == "--dense-q8" and args.dense_q8) or \
               (flag == "--cpu-tier" and args.cpu_tier):
                print(f"[strata-experts] note: {flag} is a MiMo engine flag; not written "
                      f"into this {family} config ({meaning})", file=sys.stderr)
        if args.drafter:
            print("[strata-experts] note: --drafter maps to MiMo's --dflash; Qwen MTP/"
                  "drafter settings are config-specific and left to you (untouched)",
                  file=sys.stderr)
    cfg = {"exe": engine, "cwd": str(root),
           **({"arch": "mimo_v2"} if family == "mimo_v2" else {}),
           "backend": "hip", "tokenizer": str(Path(args.tokenizer).expanduser()),
           "lib_dirs": [], "gpu": gpus,
           "env": {"HIP_VISIBLE_DEVICES": ",".join(str(g) for g in gpus)},
           "args": engine_args}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output} (family {family}, model {model_dir})")
    print(f"start with:  python tools/strata_experts.py launch --config {output}"
          f"{' --experts <artifact>' if args.experts_hint else ''}   # add --single-gpu"
          f" for one card; --performance baseline keeps the config as written")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="strata_experts.py",
        description="Pack a model's routed experts into a named artifact (CPU-only, "
                    "opt-in; the engine's native expert path is unchanged without it).",
        epilog="Examples:\n"
               "  %(prog)s repack --model /mnt/models/MiMo2.6_EXL3_2.27bpw \\\n"
               "      --output /mnt/bigssd/mimo-2.27.strata-experts.json --dry-run\n"
               "  %(prog)s repack --model /mnt/models/MiMo2.6_EXL3_2.27bpw \\\n"
               "      --output /mnt/bigssd/mimo-2.27.strata-experts.json\n"
               "  %(prog)s config --model /mnt/models/MiMo2.6_EXL3_2.27bpw \\\n"
               "      --tokenizer /mnt/models/mimo-tokenizer --output my-mimo.json \\\n"
               "      --gpu 0,1 --peer-device 1 --dense-q8 --cpu-tier\n"
               "  %(prog)s launch --config my-mimo.json --experts \\\n"
               "      /mnt/bigssd/mimo-2.27.strata-experts.json\n"
               "  %(prog)s launch --config my-qwen.json --experts \\\n"
               "      /mnt/bigssd/qwen-4bpw.strata-experts.json --single-gpu\n"
               "  %(prog)s launch --config my-mimo.json --repack-in-memory --dry-run\n"
               "      # the same packed bytes, built in RAM at startup (no disk artifact)\n"
               "  %(prog)s validate /mnt/bigssd/mimo-2.27.strata-experts.json "
               "--model /mnt/models/MiMo2.6_EXL3_2.27bpw\n",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_repack = sub.add_parser(
        "repack", help="build (or reuse) a packed-expert generation and publish the "
                       "named descriptor")
    p_repack.add_argument("--model", required=True,
                          help="the EXL3 model dir (config.json + shards + index.json)")
    p_repack.add_argument("--output", required=True,
                          help=f"the artifact file YOU name, ending {OUTPUT_SUFFIX} "
                               f"(e.g. /mnt/bigssd/my-model{OUTPUT_SUFFIX}); the packed "
                               f"payload goes into the managed directory next to it")
    p_repack.add_argument("--cache-root",
                          help="pack into this existing cache root instead of the directory "
                               "next to --output (reuse an earlier generation by fingerprint; "
                               "a mismatching model builds a NEW generation there)")
    p_repack.add_argument("--family", choices=("auto", "mimo", "qwen"), default="auto",
                          help="override the auto-detected family (refuses a mismatch)")
    p_repack.add_argument("--helper", help="path to the exl3_band8_build binary "
                                           "(default: $STRATA_EXPERTS_HELPER, then "
                                           "build-hip-exl3/, then build/)")
    p_repack.add_argument("--progress", type=int, default=0,
                          help="helper progress line every N projection triples")
    p_repack.add_argument("--dry-run", action="store_true",
                          help="print the plan (family, sizes, free space, reuse) and exit "
                               "without writing anything")
    p_repack.add_argument("--force", action="store_true",
                          help="replace an existing descriptor file (the payload is never "
                               "clobbered: generations are fingerprint-addressed)")
    p_repack.set_defaults(func=cmd_repack)

    p_val = sub.add_parser("validate", help="check a descriptor against its generation "
                                            "and source model")
    p_val.add_argument("descriptor", help=f"the {OUTPUT_SUFFIX} file")
    p_val.add_argument("--model", help="also verify the descriptor matches THIS model dir "
                                       "(family, fingerprint, sizes)")
    p_val.set_defaults(func=cmd_validate)

    p_env = sub.add_parser("env", help="print the engine environment lines the descriptor "
                                       "selects (used by the launchers)")
    p_env.add_argument("descriptor")
    p_env.add_argument("--model", help="verify the source model before printing")
    p_env.set_defaults(func=cmd_env)

    p_launch = sub.add_parser(
        "launch", help="start the server from a config file (the tracked, portable "
                       "interface; without --experts the run stays native)")
    p_launch.add_argument("--config", required=True,
                          help="a server config JSON (any model/tokenizer/drafter paths "
                               "are yours; a UNIQUE temporary derived config is passed to "
                               "the server and cleaned up when it exits)")
    p_launch.add_argument("--experts",
                          help="select a packed-expert artifact (.strata-experts.json); "
                               "it is validated against the config's --exl3 model first")
    p_launch.add_argument("--repack-in-memory", action="store_true",
                          help="ALTERNATIVE packed mode, no disk artifact: at engine startup "
                               "the routed experts are permuted into the canonical band-8 "
                               "order directly in the pinned resident RAM arena (the same "
                               "bytes a repack writes, built in RAM every launch; needs RAM "
                               "for the whole routed-expert set and arms "
                               "STRATA_EXL3_RESIDENT=all). Mutually exclusive with --experts "
                               "and with any explicit band8/cache environment")
    p_launch.add_argument("--single-gpu", action="store_true",
                          help="strip the peer tier: --peer-device removed from the args, "
                               "the config's gpu list and HIP/CUDA_VISIBLE_DEVICES cut to "
                               "the first card")
    p_launch.add_argument("--performance", choices=PERFORMANCE_PROFILES, default="auto",
                          help="'auto' (default): arm the coherent measured profile for "
                               "the detected family where THIS machine supports it (HIP "
                               "backend + confirmed gfx11 for the GPU flags, AVX-512BW "
                               "for CPU assistance; absent evidence keeps the config "
                               "unchanged with a logged reason). 'baseline': genuine "
                               "opt-out - the derived config is byte-identical to yours "
                               "(the server still inherits your shell STRATA_* for keys "
                               "the config does not set). Explicit values (shell or "
                               "config env) always win, including 0, and a dependency "
                               "that is explicitly off makes auto decline the dependent "
                               "feature with an explanation")
    p_launch.add_argument("--server", help="alternative server entrypoint (advanced; "
                                           "default: serve/server.py of this checkout)")
    p_launch.add_argument("--python", help="Python interpreter for the server "
                                           "(default: the one running this tool)")
    p_launch.add_argument("--dry-run", action="store_true",
                          help="print the derived config and command as JSON and exit; "
                               "nothing is started, nothing is written")
    p_launch.add_argument("server_args", nargs="*",
                          help="extra serve/server.py options, after a '--' separator "
                               "(for example -- --port 8081 --lazy)")
    p_launch.set_defaults(func=cmd_launch)

    p_cfg = sub.add_parser(
        "config", help="write a minimal, complete server config (no required fields to "
                       "invent by hand)")
    p_cfg.add_argument("--model", required=True, help="the EXL3 model dir")
    p_cfg.add_argument("--tokenizer", required=True, help="the tokenizer pack path")
    p_cfg.add_argument("--output", required=True,
                       help="where to write the config JSON (refused if it exists, "
                            "unless --force)")
    p_cfg.add_argument("--family", choices=("auto", "mimo", "qwen"), default="auto",
                       help="override the auto-detected family (refuses a mismatch)")
    p_cfg.add_argument("--engine", help="the engine executable "
                                        "(default: build-hip-exl3/strata of this checkout)")
    p_cfg.add_argument("--gpu", default="0",
                       help="GPU list, comma-separated (default 0; e.g. 0,1 for the peer "
                            "tier)")
    p_cfg.add_argument("--peer-device", type=int,
                       help="arm the second-GPU peer tier with this device")
    p_cfg.add_argument("--dense-q8", action="store_true",
                       help="MiMo: serve the dense projections as q8_0 (measured)")
    p_cfg.add_argument("--cpu-tier", action="store_true",
                       help="MiMo: pass --cpu-tier (the engine arg the CPU tier needs)")
    p_cfg.add_argument("--drafter", help="MiMo: the dflash drafter dir (--dflash)")
    p_cfg.add_argument("--experts-hint", action="store_true",
                       help="mention --experts in the printed start command")
    p_cfg.add_argument("--force", action="store_true", help="replace an existing output")
    p_cfg.set_defaults(func=cmd_config)

    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    try:
        rc = args.func(args, root)
    except Refused as e:
        print(f"strata-experts: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
    return rc


if __name__ == "__main__":
    sys.exit(main())
