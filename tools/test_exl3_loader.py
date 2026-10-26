"""Exercise the C++ model loader with small valid and malformed safetensors files.

STRATA_EXL3_MODEL_CHECK=build-cuda12/strata-exl3-model python tools/test_exl3_loader.py
"""
import copy
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest


@unittest.skipUnless(os.environ.get("STRATA_EXL3_MODEL_CHECK"), "needs the C++ model checker")
class LoaderTests(unittest.TestCase):
    def run_model(self, edit=None, bits=3, pad=False, header_length=None):
        tensors = {
            "linear.trellis": {"dtype": "I16", "shape": [8, 8, bits * 16],
                               "data_offsets": [0, 8 * 8 * bits * 32]},
        }
        offset = tensors["linear.trellis"]["data_offsets"][1]
        for name in ("suh", "svh"):
            tensors["linear." + name] = {"dtype": "F16", "shape": [128],
                                         "data_offsets": [offset, offset + 256]}
            offset += 256
        payload = bytes(offset)
        if edit:
            edit(tensors)
        if pad:
            payload = payload.ljust(max(t["data_offsets"][1] for t in tensors.values()), b"\0")
        header = json.dumps(tensors).encode()
        with tempfile.TemporaryDirectory(prefix="strata-exl3-loader-") as tmp:
            root = Path(tmp)
            (root / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": {key: "model.safetensors" for key in tensors}}))
            (root / "model.safetensors").write_bytes(struct.pack("<Q", len(header) if header_length is None else header_length) + header + payload)
            return subprocess.run([os.environ["STRATA_EXL3_MODEL_CHECK"], tmp, "linear"],
                                  capture_output=True, text=True)

    def test_integer_bitrates(self):
        for bits in range(1, 9):
            with self.subTest(bits=bits):
                result = self.run_model(bits=bits)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_unsupported_geometry(self):
        for shape in ([8, 8, 24], [8, 8, 0], [8, 8, 144], [1, 8, 48], [0, 8, 48], [8, 8]):
            with self.subTest(shape=shape):
                def edit(tensors):
                    t = tensors["linear.trellis"]
                    t["shape"] = shape
                    # Give the malformed geometry its correct byte count; fail in linear(), not file parsing.
                    count = 2
                    for dim in shape:
                        count *= dim
                    t["data_offsets"][1] = count
                result = self.run_model(edit, pad=True)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn("integer K=1..8", result.stderr)

    def test_bad_scales(self):
        for key, value in (("dtype", "I16"), ("shape", [64, 2])):
            with self.subTest(key=key):
                result = self.run_model(lambda t: t["linear.suh"].update({key: value}))
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn("F16 suh[in]", result.stderr)

    def test_file_bounds_and_sizes(self):
        for end in (-1, 100000, 1):
            with self.subTest(end=end):
                result = self.run_model(lambda t: t["linear.trellis"].update(data_offsets=[0, end]))
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn("safetensors:", result.stderr)
        for shape in ([-1, -1], [9223372036854775807, 3], [9223372036854775808]):
            with self.subTest(shape=shape):
                result = self.run_model(lambda t: t["linear.trellis"].update(shape=shape))
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn("safetensors", result.stderr)

    def test_header_length_overflow(self):
        for length in (0, (1 << 64) - 1):
            with self.subTest(length=length):
                result = self.run_model(header_length=length)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn("bad header length", result.stderr)

    def test_bias_is_rejected(self):
        def edit(tensors):
            tensors["linear.bias"] = copy.deepcopy(tensors["linear.svh"])
        result = self.run_model(edit)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("biased EXL3", result.stderr)


@unittest.skipUnless(os.environ.get("STRATA_EXL3_WAVE_CHECK"), "needs the C++ wave checker")
class WaveSyncTests(unittest.TestCase):
    """A synthetic one-layer EXL3 model (80 routed experts, K4 mul1, 2560/640) for the expert
    stream's prefill path: STRATA_EXL3_WAVESYNC=1 and STRATA_EXL3_PIPE=1 must each produce
    byte-identical run_device output to the default per-tile barrier (same kernels, same order -
    only scheduling differs).  Every expert and projection gets DISTINCT weights and scales so a
    slot/expert mixup - or the pipeline's slot reuse before the slot's last reader - cannot pass.
    The checker covers cached and scratch (cache 0) streams, several waves with eviction and slot
    reuse, a partial final wave, one group over two tiles, a larger group after a smaller one,
    warm repeats and the prefill -> run() -> decode-window phase sequence."""

    def test_wave_sync_matches_per_tile(self):
        experts, bits, ki_gu, nj_gu = 80, 4, 160, 40   # gate/up 2560->640, down 640->2560
        tensors, offset, payload = {}, 0, bytearray()

        def add(name, dtype, shape, nbytes):
            nonlocal offset
            tensors[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + nbytes]}
            offset += nbytes

        def stream(seed, n):
            """Deterministic bytes distinct per seed: one 64 KiB LCG chunk per seed, tiled."""
            chunk = bytearray(65536)
            state = seed & 0xFFFFFFFF
            for i in range(0, len(chunk), 4):
                state = (1103515245 * state + 12345) & 0xFFFFFFFF
                chunk[i:i + 4] = state.to_bytes(4, "little")
            return (bytes(chunk) * (n // len(chunk) + 1))[:n]

        def scales(seed, count):
            """Distinct finite f16 scales in +/-[0.0156, 0.0312): random sign+mantissa, exponent
            fixed to the 0x2800 band - calibrated so the random trellis' f16 intermediates stay
            finite (larger bands overflow the gate/up packing, smaller underflow the output)."""
            raw = stream(seed, count * 2)
            out = bytearray(count * 2)
            for i in range(count):
                word = (int.from_bytes(raw[2 * i:2 * i + 2], "little") & 0x83FF) | 0x2800
                out[2 * i:2 * i + 2] = word.to_bytes(2, "little")
            return bytes(out)

        for e in range(experts):
            for p, (proj, ki, nj) in enumerate((("gate_proj", ki_gu, nj_gu), ("up_proj", ki_gu, nj_gu),
                                                ("down_proj", nj_gu, ki_gu))):
                seed = 7919 * e + 104729 * p + 12345   # distinct per (expert, projection)
                base = f"model.language_model.layers.0.mlp.experts.{e}.{proj}"
                n = ki * nj * bits * 32   # uint16 words: ki*nj*(bits*16) * 2 bytes
                add(base + ".trellis", "I16", [ki, nj, bits * 16], n)
                payload += stream(seed, n)
                add(base + ".suh", "F16", [ki * 16], ki * 32); payload += scales(seed + 1, ki * 16)
                add(base + ".svh", "F16", [nj * 16], nj * 32); payload += scales(seed + 2, nj * 16)
                add(base + ".mul1", "I16", [1], 2); payload += b"\x00\x00"
        # A second layer with a different K and FFN width exercises rank/layer offsets.
        for e in range(16):
            for p, (proj, ki, nj) in enumerate((("gate_proj", 160, 8), ("up_proj", 160, 8),
                                               ("down_proj", 8, 160))):
                seed = 7919 * e + 104729 * p + 777
                base = f"model.language_model.layers.1.mlp.experts.{e}.{proj}"
                n = ki * nj * 3 * 32
                add(base + ".trellis", "I16", [ki, nj, 48], n); payload += stream(seed, n)
                add(base + ".suh", "F16", [ki * 16], ki * 32); payload += scales(seed + 1, ki * 16)
                add(base + ".svh", "F16", [nj * 16], nj * 32); payload += scales(seed + 2, nj * 16)
                add(base + ".mul1", "I16", [1], 2); payload += b"\x00\x00"
        header = json.dumps(tensors).encode()
        with tempfile.TemporaryDirectory(prefix="strata-exl3-wave-") as tmp:
            root = Path(tmp)
            (root / "model.safetensors.index.json").write_text(json.dumps(
                {"weight_map": {key: "model.safetensors" for key in tensors}}))
            (root / "model.safetensors").write_bytes(
                struct.pack("<Q", len(header)) + header + bytes(payload))
            result = subprocess.run([os.environ["STRATA_EXL3_WAVE_CHECK"], str(root)],
                                    capture_output=True, text=True)
            if result.returncode == 77:
                self.skipTest("no CUDA device (checker exited 77)")
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            self.assertIn("PASS", result.stdout)

    def test_read_path_arms_byte_identical(self):
        """STRATA_EXL3_NO_PREAD unset vs present: the read-path choice is a once-per-process
        static, so equivalence is proven ACROSS processes.  Both checker runs hash every
        compared output (STRATA_EXL3_WAVE_CHECK_HASH); the whole battery - duplicate ids, all
        hit/miss mixes, cache 0, eviction, PIPE/OVERLAP/BATCH arms, the loan interplay - must
        produce the SAME SEQUENCEHASH with the env unset and set to 1."""
        import tempfile
        experts, bits, ki_gu, nj_gu = 80, 4, 160, 40
        tensors, offset, payload = {}, 0, bytearray()

        def add(name, dtype, shape, nbytes):
            nonlocal offset
            tensors[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + nbytes]}
            offset += nbytes

        def stream(seed, n):
            chunk = bytearray(65536)
            state = seed & 0xFFFFFFFF
            for i in range(0, len(chunk), 4):
                state = (1103515245 * state + 12345) & 0xFFFFFFFF
                chunk[i:i + 4] = state.to_bytes(4, "little")
            return (bytes(chunk) * (n // len(chunk) + 1))[:n]

        def scales(seed, count):
            raw = stream(seed, count * 2)
            out = bytearray(count * 2)
            for i in range(count):
                word = (int.from_bytes(raw[2 * i:2 * i + 2], "little") & 0x83FF) | 0x2800
                out[2 * i:2 * i + 2] = word.to_bytes(2, "little")
            return bytes(out)

        for e in range(experts):
            for p, (proj, ki, nj) in enumerate((("gate_proj", ki_gu, nj_gu), ("up_proj", ki_gu, nj_gu),
                                                ("down_proj", nj_gu, ki_gu))):
                seed = 7919 * e + 104729 * p + 12345
                base = f"model.language_model.layers.0.mlp.experts.{e}.{proj}"
                n = ki * nj * bits * 32
                add(base + ".trellis", "I16", [ki, nj, bits * 16], n)
                payload += stream(seed, n)
                add(base + ".suh", "F16", [ki * 16], ki * 32); payload += scales(seed + 1, ki * 16)
                add(base + ".svh", "F16", [nj * 16], nj * 32); payload += scales(seed + 2, nj * 16)
                add(base + ".mul1", "I16", [1], 2); payload += b"\x00\x00"
        for e in range(16):
            for p, (proj, ki, nj) in enumerate((("gate_proj", 160, 8), ("up_proj", 160, 8),
                                               ("down_proj", 8, 160))):
                seed = 7919 * e + 104729 * p + 777
                base = f"model.language_model.layers.1.mlp.experts.{e}.{proj}"
                n = ki * nj * 3 * 32
                add(base + ".trellis", "I16", [ki, nj, 48], n); payload += stream(seed, n)
                add(base + ".suh", "F16", [ki * 16], ki * 32); payload += scales(seed + 1, ki * 16)
                add(base + ".svh", "F16", [nj * 16], nj * 32); payload += scales(seed + 2, nj * 16)
                add(base + ".mul1", "I16", [1], 2); payload += b"\x00\x00"
        header = json.dumps(tensors).encode()
        with tempfile.TemporaryDirectory(prefix="strata-exl3-wave-") as tmp:
            root = Path(tmp)
            (root / "model.safetensors.index.json").write_text(json.dumps(
                {"weight_map": {key: "model.safetensors" for key in tensors}}))
            (root / "model.safetensors").write_bytes(
                struct.pack("<Q", len(header)) + header + bytes(payload))
            hashes = []
            for no_pread in (None, "1"):
                env = dict(os.environ)
                env.pop("STRATA_EXL3_NO_PREAD", None)
                env["STRATA_EXL3_WAVE_CHECK_HASH"] = "1"
                if no_pread is not None:
                    env["STRATA_EXL3_NO_PREAD"] = no_pread
                result = subprocess.run([os.environ["STRATA_EXL3_WAVE_CHECK"], str(root)],
                                        capture_output=True, text=True, env=env)
                if result.returncode == 77:
                    self.skipTest("no CUDA device (checker exited 77)")
                self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
                lines = [l for l in result.stdout.splitlines() if l.startswith("SEQUENCEHASH ")]
                self.assertEqual(len(lines), 1, "one SEQUENCEHASH line expected")
                hashes.append(lines[0].split()[1])
            self.assertEqual(hashes[0], hashes[1],
                             "the two read-path arms must be byte-identical over the whole battery")


@unittest.skipUnless(os.environ.get("STRATA_EXL3_SMALL_M_SEAM"), "needs the small-M seam checker")
class SmallMSeamTests(unittest.TestCase):
    """The OPT-IN grouped small-M FFN inside the host-IO expert service (run_multi - the verifier
    decode windows' path) on the same synthetic two-geometry model: STRATA_EXL3_SMALL_M=1, alone
    and with STRATA_EXL3_OVERLAP=1, must be byte-identical to the untouched default over a window
    battery (mixed hit/miss, cold/warm, duplicate ids, zero routes, M1/M2 stay-indexed regression,
    M3..M6 grouped mixes, a 7-row-group fallback, repeated windows and both FFN geometries), and
    the checker asserts the dispatch really happened and that the workspace never grew after
    warmup (the pre-reserve-under-queued-work contract)."""

    def test_small_m_seam_matches_default(self):
        experts, bits, ki_gu, nj_gu = 80, 4, 160, 40
        tensors, offset, payload = {}, 0, bytearray()

        def add(name, dtype, shape, nbytes):
            nonlocal offset
            tensors[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + nbytes]}
            offset += nbytes

        def stream(seed, n):
            chunk = bytearray(65536)
            state = seed & 0xFFFFFFFF
            for i in range(0, len(chunk), 4):
                state = (1103515245 * state + 12345) & 0xFFFFFFFF
                chunk[i:i + 4] = state.to_bytes(4, "little")
            return (bytes(chunk) * (n // len(chunk) + 1))[:n]

        def scales(seed, count):
            raw = stream(seed, count * 2)
            out = bytearray(count * 2)
            for i in range(count):
                word = (int.from_bytes(raw[2 * i:2 * i + 2], "little") & 0x83FF) | 0x2800
                out[2 * i:2 * i + 2] = word.to_bytes(2, "little")
            return bytes(out)

        for e in range(experts):
            for p, (proj, ki, nj) in enumerate((("gate_proj", ki_gu, nj_gu), ("up_proj", ki_gu, nj_gu),
                                                ("down_proj", nj_gu, ki_gu))):
                seed = 7919 * e + 104729 * p + 12345
                base = f"model.language_model.layers.0.mlp.experts.{e}.{proj}"
                n = ki * nj * bits * 32
                add(base + ".trellis", "I16", [ki, nj, bits * 16], n)
                payload += stream(seed, n)
                add(base + ".suh", "F16", [ki * 16], ki * 32); payload += scales(seed + 1, ki * 16)
                add(base + ".svh", "F16", [nj * 16], nj * 32); payload += scales(seed + 2, nj * 16)
                add(base + ".mul1", "I16", [1], 2); payload += b"\x00\x00"
        for e in range(16):
            for p, (proj, ki, nj) in enumerate((("gate_proj", 160, 8), ("up_proj", 160, 8),
                                               ("down_proj", 8, 160))):
                seed = 7919 * e + 104729 * p + 777
                base = f"model.language_model.layers.1.mlp.experts.{e}.{proj}"
                n = ki * nj * 3 * 32
                add(base + ".trellis", "I16", [ki, nj, 48], n); payload += stream(seed, n)
                add(base + ".suh", "F16", [ki * 16], ki * 32); payload += scales(seed + 1, ki * 16)
                add(base + ".svh", "F16", [nj * 16], nj * 32); payload += scales(seed + 2, nj * 16)
                add(base + ".mul1", "I16", [1], 2); payload += b"\x00\x00"
        header = json.dumps(tensors).encode()
        with tempfile.TemporaryDirectory(prefix="strata-exl3-smallm-") as tmp:
            root = Path(tmp)
            (root / "model.safetensors.index.json").write_text(json.dumps(
                {"weight_map": {key: "model.safetensors" for key in tensors}}))
            (root / "model.safetensors").write_bytes(
                struct.pack("<Q", len(header)) + header + bytes(payload))
            result = subprocess.run([os.environ["STRATA_EXL3_SMALL_M_SEAM"], str(root)],
                                    capture_output=True, text=True)
            if result.returncode == 77:
                self.skipTest("no CUDA device (checker exited 77)")
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            self.assertIn("engine seam PASS", result.stdout)


@unittest.skipUnless(os.environ.get("STRATA_EXL3_ENGINE"), "needs the engine")
class CommandLineTests(unittest.TestCase):
    def engine(self, *args):
        return subprocess.run([os.environ["STRATA_EXL3_ENGINE"], "--tokens", "1", *args],
                              capture_output=True, text=True)

    def test_ple_sources_are_exclusive(self):
        for args in (("--ple-exl3", "fake", "--no-ple"),
                     ("--ple-exl3", "fake", "--ple-gguf", "fake")):
            with self.subTest(args=args):
                result = self.engine(*args)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("choose one", result.stderr)

    def test_unsupported_modes_fail_before_loading(self):
        for args in (("--layer-split", "auto"),
                     ("--native", "fake.gguf")):
            with self.subTest(args=args):
                result = self.engine("--exl3", "fake", "--no-ple", *args)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("--exl3 does not yet accept", result.stderr)

    def test_help_lists_exl3(self):
        result = self.engine("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--exl3 DIR", result.stderr + result.stdout)
        self.assertIn("--ple-exl3 FILE", result.stderr + result.stdout)

    def test_diagnostic_dense_modes_are_rejected_for_windows(self):
        for dense in ("exl3", "only:output"):
            with self.subTest(dense=dense):
                result = subprocess.run([os.environ["STRATA_EXL3_ENGINE"], "--exl3", "fake",
                                         "--no-ple", "--tokens", "1", "--spec", "4"],
                    env={**os.environ, "STRATA_EXL3_DENSE": dense}, capture_output=True, text=True)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("unset STRATA_EXL3_DENSE", result.stderr)


if __name__ == "__main__":
    unittest.main()
