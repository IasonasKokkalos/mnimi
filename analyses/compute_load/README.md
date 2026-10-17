# Compute-load evidence (2026-10-17, the README's "Compute load" table)

The README's per-profile table is read from these files. One fresh process per cell on an AMD Ryzen 7 PRO 8845HS
(8 cores, 16 logical, 32 GiB, Windows 11); the CPU cells in a venv with Python 3.12.13, `mnimi` 3.1.1 (editable,
commit `66d13ad`), numpy 2.5.4, sqlite-vec 0.1.9, onnxruntime 1.28.0, tokenizers 0.23.1 and the CPU wheel of
llama-cpp-python 0.3.35; the GPU cells in the run environment (Python 3.14.4, numpy 2.4.6, the CUDA 13.2 build of
0.3.35, DECISIONS "Extractor runtime") on the RTX 1000 Ada (6 GB, driver 595.95).

- `compute_load.py`: the cell script. `prep` writes `rounds50.json` once (not committed: it holds dataset text):
  the first 50 rounds, in session order, of the LongMemEval-S questions outside the 100-question stratified slice
  (the dev pool of `evals.probes.extractor_bench`; all 50 come from question `e47becba`), each round the library's
  own `_messages_to_rounds` pairing with the session date as `ts`, mean 1,829 characters; and the 20 question
  texts of the 20-row stratified slice as queries. The other cells read that file, so the process never holds the
  parsed dataset (a first pass that loaded it in-process peaked at 2.5 GiB from the dataset alone and was discarded).
- Cells: `base` (import only), `hash` (`HashingEmbedder`), `bge` (`BgeSmallEmbedder`), `extract --profile cpu|gpu`
  (`BgeSmallEmbedder` + `CachedExtractor(QwenLlamaExtractor(...))` with a fresh cache; the CPU profile is
  `{**DECODE, "n_gpu_layers": 0, "max_tokens": 2048}`, the GPU profile `DECODE`). Each cell constructs `Memory` on a
  fresh store, does one `add()` per round (50 timings), one warm-up `recall()`, then 3 passes over the 20 queries of
  `recall()` and `get_context()` (60 timings each). Timings are `time.perf_counter`; the README quotes medians.
  Resident memory is the process's peak working set from `K32GetProcessMemoryInfo`; the JSON also carries the working
  set after import, after the embedder, after the extractor, after the writes and at the end. VRAM is the
  device-used delta of `nvidia-smi --query-gpu=memory.used` with the model loaded (`vram_probe.txt`; the per-process
  query returns `[N/A]` under WDDM).
- `*_run1.json`, `*_run2.json`: the cells' raw output. `extract_gpu_run1.json` ran under Python 3.14 (the only
  environment with the CUDA build), so its read timings are not paired with the venv's: the README quotes the venv's
  read figures for every row and says so; `hash_py314.json` and `bge_py314.json` are the same two CPU cells under
  3.14 for that comparison.
- `bench40.py`, `bench40_cpu.txt`, `bench40_gpu.txt`, `dev40_*_seconds.json`: the extractor alone on the 40 dev
  rounds of `evals.probes.extractor_bench --dev-set 40` (seed 0, limit 100: the same sampling as the 2026-10-10 T5
  measurement in `examples/README.md`), reusing the bench's sampling, loop and summary with the profile's decode
  dict. The per-round seconds are kept; the emitted JSONL (round text and raw outputs) is not committed.
- Installed sizes in the README's "on disk" column are `du` over the venv's `site-packages` (onnxruntime 41,329,664 B,
  tokenizers 7,929,856 B, llama_cpp 10,406,912 B, numpy 25,064,448 B, sqlite_vec 299,008 B); the BGE download is the
  two files `embeddings.py` pulls by revision (`onnx/model.onnx` 133,093,490 B + `tokenizer.json` 711,396 B); the
  GGUF is 1,834,426,016 B.

Diagnostic only: nothing here enters a run's pins or a published row.
