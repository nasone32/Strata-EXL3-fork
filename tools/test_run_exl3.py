"""Portable launcher checks: synthetic model headers, no GPU or downloads."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_exl3 as run
from test_strata_experts import make_qwen_model, make_mimo_model


class Launch(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.model = Path(self.tmp.name) / "model with spaces"
        make_qwen_model(self.model)
        (self.model / "tokenizer").mkdir()
        (self.model / "tokenizer/vocab.json").write_text('{}')
        (self.model / "ngram_embedding.safetensors").touch()
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def config(self, *args):
        return run.make_config(run.parser().parse_args(["--model-dir", str(self.model), *args]))

    def test_native_single_gpu_defaults(self):
        c = self.config()
        self.assertEqual(c['gpu'], [0])
        self.assertEqual(c['env']['STRATA_EXL3_CPU_TIER'], '1')
        self.assertEqual(c['env']['STRATA_EXL3_RESIDENT'], 'all')
        self.assertNotIn('STRATA_QWEN_BAND8', c['env'])
        self.assertNotIn('STRATA_EXL3_RDNA', c['env'])  # the ENGINE selects its defaults

    def test_peer_defaults(self):
        c = self.config('--peer-device', '1')
        self.assertEqual(c['gpu'], [0, 1])
        self.assertEqual(c['env']['STRATA_EXL3_CPU_TIER'], '0')
        self.assertIn('--peer-device', c['args'])
        self.assertEqual(c['args'][c['args'].index('--peer-slots') + 1], '65535')

    def test_cpu_and_affinity(self):
        c = self.config('--peer-device', '1', '--cpu', 'on', '--cpu-threads', '3', '--aux-cpus', 'off')
        self.assertEqual(c['env']['STRATA_EXL3_CPU_THREADS'], '3')
        self.assertEqual(c['env']['STRATA_AUX_CPUS'], 'off')
        self.assertEqual(c['env']['STRATA_EXL3_CPU_HYBRID_PEER'], '1')

    def test_explicit_zero(self):
        with patch.dict(os.environ, {'STRATA_EXL3_CPU_TIER': '0', 'STRATA_EXL3_RESIDENT': '0'}):
            c = self.config()
            self.assertEqual(c['env']['STRATA_EXL3_CPU_TIER'], '0')
            self.assertEqual(c['env']['STRATA_EXL3_RESIDENT'], '0')
            self.assertEqual(self.config('--cpu', 'on')['env']['STRATA_EXL3_CPU_TIER'], '1')

    def test_qwen_peer_residency_refused(self):
        with self.assertRaisesRegex(run.experts.Refused, 'low-RAM Qwen'):
            self.config('--resident', 'peer', '--peer-device', '1')

    def test_topology_refused(self):
        for opts in [('--gpu', '0,1'), ('--gpu', '0', '--peer-device', '1'), ('--gpu', '0,0')]:
            with self.subTest(opts=opts), self.assertRaises(run.experts.Refused):
                self.config(*opts)

    def test_remote_requires_key(self):
        with self.assertRaisesRegex(run.experts.Refused, '--api-key'):
            self.config('--host', '0.0.0.0')
        self.config('--host', '0.0.0.0', '--api-key', 'test-only')

    def test_dry_run_no_engine_needed(self):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(run.main(['--model-dir', str(self.model), '--dry-run']), 0)
        payload = json.loads(out.getvalue())
        self.assertEqual(payload['config']['gpu'], [0])
        self.assertIn('8085', payload['command'])

    def test_missing_tokenizer_refused(self):
        (self.model / 'tokenizer/vocab.json').unlink()
        with self.assertRaisesRegex(run.experts.Refused, 'tokenizer'):
            self.config()

    def test_mimo(self):
        other = Path(self.tmp.name) / 'mimo'
        make_mimo_model(other)
        (other / 'vocab.json').write_text('{}')
        a = run.parser().parse_args(['--family', 'mimo', '--model-dir', str(other), '--peer-device', '1'])
        c = run.make_config(a)
        self.assertIn('--cpu-tier', c['args'])
        self.assertEqual(c['env']['STRATA_MIMO_PEER_DEVICE_RESULTS'], '1')
        self.assertEqual(c['env']['STRATA_MIMO_WINDOW_DEVICE_MISSES'], '1')
        self.assertNotIn('--peer-reserve-mib', c['args'])
        self.assertNotIn('STRATA_MIMO_BAND8', c['env'])


if __name__ == '__main__':
    unittest.main()
