#!/usr/bin/env python3
"""One short release comparison: CODE x2, PROSE x2, full 32768-token prefill x2.

Two GPUs, native experts, CPU help off, identical cache/context/MTP settings.
Old published profile versus new engine defaults + full RAM residency. This is a
release-profile comparison, not attribution to any individual kernel or flag.
No prompt/checkpoint reuse; expert/OS caches remain warm between requests.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('exl3_bench', ROOT / 'tools/exl3/bench.py')
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('baseline', 'candidate', 'model-3', 'model-4', 'tokenizer', 'mtp', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--max-new', type=int, default=512)
    p.add_argument('--single-gpu-only', action='store_true',
                   help='candidate 4.05 bpw: GPU-only native vs CPU assistance + RAM repack')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)  # never clobber a result
    tk = bench.load_tokenizer(a.tokenizer)
    short = lambda text: tk.encode('<|im_start|>user\n' + text +
        '<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n\n', parse_special=True)
    prompts = {'CODE': short('write a bubblesort algorithm in C code'),
               'PROSE': short('explain a PID controller ELI5'),
               'PREFILL32K': bench.prompt(tk, 32768, 0, False)[0]}
    assert len(prompts['PREFILL32K']) == 32768
    (a.output / 'prompts.json').write_text(json.dumps(prompts))
    summary = []
    try:
        cases = [('3.05', a.model_3, 165, 10200), ('4.05', a.model_4, 104, 7200)]
        if a.single_gpu_only:
            cases = [('4.05', a.model_4, 104, 0)]
        for quant, model, slots, peer_slots in cases:
            arms = [('published', a.baseline), ('candidate', a.candidate)]
            if a.single_gpu_only:
                arms = [('native-gpu-only', a.candidate), ('cpu-ram-repack', a.candidate)]
            for arm, binary in arms:
                out = a.output / (quant + '-' + arm)
                out.mkdir()
                # Discard inherited tuning, not the runtime library search path.
                env = {k: v for k, v in os.environ.items() if not k.startswith('STRATA_')}
                env.update(HIP_VISIBLE_DEVICES='0' if a.single_gpu_only else '0,1', STRATA_AUX_CPUS='off',
                           STRATA_EXL3_CPU_TIER='0', STRATA_EXL3_PEER_ADAPT='0',
                           STRATA_EXL3_RESIDENT='0' if arm == 'published' else 'all')
                if arm == 'cpu-ram-repack':
                    env.update(STRATA_QWEN_BAND8='memory', STRATA_EXL3_CPU_TIER='1',
                               STRATA_EXL3_CPU_THREADS='8', STRATA_EXL3_CPU_HYBRID='2',
                               STRATA_EXL3_CPU_HYBRID_MIN_MISSES='1',
                               STRATA_EXL3_CPU_CACHE_MIB='0', STRATA_EXL3_CPU_SWIZZLE='0')
                if arm == 'published':
                    env.update({k: v for k, v in [
                        ('STRATA_EXL3_RDNA', '1'), ('STRATA_EXL3_FUSE', '3'),
                        ('STRATA_EXL3_WMMA', '1'), ('STRATA_WMMA_GEMM', '1'),
                        ('STRATA_HIP_WMMA', '1'), ('STRATA_EXL3_PRIMARY_MULTIROW', '2'),
                        ('STRATA_EXL3_PEER_MULTIROW', '4'), ('STRATA_EXL3_EXPERT_BATCH', '1'),
                        ('STRATA_EXL3_PEER_BATCH', '1'), ('STRATA_EXL3_PEER_PREFILL', '1')]})
                args = ['--exl3', str(model), '--ple-exl3', str(model / 'ngram_embedding.safetensors'),
                        '--expert-profile', str(ROOT / 'data/expert-profile.bin'),
                        '--expert-cache', str(slots), '--expert-ram-cache', '0',
                        *([] if a.single_gpu_only else ['--peer-device', '1', '--peer-slots', str(peer_slots)]),
                        '--vram-reserve-mib', '2048', '--peer-reserve-mib', '4096',
                        '--prefill', '8192', '--max-context', '65536', '--kv', 'int8',
                        '--mtp', str(a.mtp), '--spec', '4', '--spec-min-p', '0.5',
                        '--ple-io', 'direct', '--prompt-cache', '0',
                        '--conversation-cache-slots', '0', '--prompt-cache-every', '0']
                recorded_env = {k: v for k, v in env.items()
                                if k.startswith('STRATA_') or k == 'HIP_VISIBLE_DEVICES'}
                cfg = {'cwd': str(ROOT), 'env': recorded_env, 'args': args, 'lib_dirs': [],
                       'tokenizer': str(a.tokenizer)}
                (out / 'config.json').write_text(json.dumps(cfg, indent=2))
                command = [str(binary.resolve()), '--serve', *args]
                metadata = {'arm': arm, 'quant': quant, 'binary_sha256': sha(binary),
                            'command': command, 'started': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
                (out / 'manifest.json').write_text(json.dumps(metadata, indent=2))
                # Engine merges env onto inherited env; keep it sanitized during construction.
                inherited = os.environ.copy()
                try:
                    os.environ.clear(); os.environ.update(env)
                    eng = bench.Engine(command, cfg, out)
                finally:
                    os.environ.clear(); os.environ.update(inherited)
                try:
                    deadline = time.monotonic() + 900
                    while not eng.receive(deadline).startswith('READY '):
                        pass
                    print(f'{quant} {arm}: ready', flush=True)
                    for name, ids in prompts.items():
                        for run in (1, 2):
                            limit = 16 if name == 'PREFILL32K' else a.max_new
                            start = time.monotonic()
                            eng.send(f'GEN {limit} temperature=0 seed=1234 ckpt=0 ' + ','.join(map(str, ids)))
                            generated = []
                            deadline = time.monotonic() + 600
                            while True:
                                line = eng.receive(deadline)
                                if line.startswith('T '):
                                    generated.append(int(line.split()[1]))
                                if line.startswith('DONE '):
                                    break
                            f = line.split()
                            count, prompt_n, pp_ms, dec_ms = int(f[1]), int(f[2]), float(f[3]), float(f[4])
                            reused, read = int(f[8]), int(f[14])
                            if reused != 0 or read != len(ids) or count <= 0 or count != len(generated):
                                raise RuntimeError(f'Invalid benchmark counts/cache reuse: {line}; ids={len(ids)} outputs={len(generated)}')
                            row = {**metadata, 'prompt': name, 'run': run, 'input_tokens': prompt_n,
                                   'read_tokens': read, 'reused_tokens': reused,
                                   'generated_tokens': count, 'prefill_ms': pp_ms, 'decode_ms': dec_ms,
                                   'prefill_tokens_s': read * 1000 / pp_ms if pp_ms else None,
                                   'decode_tokens_s': count * 1000 / dec_ms if dec_ms else None,
                                   'wall_s': time.monotonic() - start, 'finish': f[5],
                                   'accepted': int(f[6]), 'offered': int(f[7]),
                                   'output_ids': generated, 'text': tk.decode(generated), 'done': line}
                            summary.append(row)
                            (out / f'{name}-{run}.json').write_text(json.dumps(row, indent=2))
                            print(f'{quant} {arm} {name} {run}: PP {row["prefill_tokens_s"]:.1f} tok/s; '
                                  f'decode {row["decode_tokens_s"]:.2f} tok/s; {count} tokens', flush=True)
                finally:
                    eng.close()
    finally:
        (a.output / 'results.json').write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
