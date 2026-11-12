# Optional Qwen draft model

Qwen runs without a drafter. The release benchmarks use an MTP runtime and a
four-token draft window; do not expect those decode rates from a no-drafter run.
This is separate from repacking EXL3 experts.

If you already have a matching Strata MTP runtime, pass its directory:

```sh
./run-peer.sh --model-dir /path/to/qwen-exl3 --mtp-dir /path/to/mtp/rt --spec 4
```

The directory contains `experts.bin`, `dense.bin` and `dense.txt`. Do not pass the
EXL3 model's quantized `mtp.layers.0` weights as a runtime directory.

To build a runtime, use the tools already in Strata. This downloads several GB of
draft tensors from the original BF16 checkpoint, **not** the whole checkpoint.
It needs extra disk space for the fetched tensors, an intermediate pack and the
runtime. Ask before downloading on somebody else's machine.

Activate `.venv`. The pack tools need `numpy` and the **pinned llama.cpp checkout's**
`gguf-py`, not an arbitrary version from pip. If that checkout is not at
`third_party/llama.cpp`, point `STRATA_GGUF_PY` at its `gguf-py` directory (the same
dependency used by the engine build):

```sh
export STRATA_GGUF_PY=/path/to/pinned/llama.cpp/gguf-py
python tools/mtp_fetch.py fetch --out /path/to/mtp/raw
python tools/mtp_fetch.py verify --out /path/to/mtp/raw
python tools/mtp_pack.py --src /path/to/mtp/raw --experts q2_0 --out /path/to/mtp/mtp-q2_0.gguf
python tools/mtp_rt.py --gguf /path/to/mtp/mtp-q2_0.gguf --out /path/to/mtp/rt
```

Fetching checks HTTP ranges and tensor checksums. Keep a failed download for
resume rather than deleting the whole directory. The runtime builder above uses
Q2_0 draft experts; changing the draft format or `--spec` changes the performance
comparison. CPU and GPU settings for the **main EXL3 model** are still selected by
the normal launcher.
