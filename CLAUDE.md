# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Read `AGENTS.md` first. It is the upstream working contract: scope control, evidence standards,
numerical-correctness rules, and the product definition. This file records only what is specific to
this checkout and does not repeat it.

## This checkout differs from AGENTS.md

`AGENTS.md` describes the upstream target: RTX 5090, `sm_120a`, CUDA 13.1, a `build/` directory.
This machine is an RTX 3090 Ti, `sm_86`, with GCC 15 as the host compiler. Build directories are
`build-sm86` and `build-test`, not `build/`. Anything in `AGENTS.md` that names `sm_120a`, the 5090,
or `build/` needs translating before you act on it.

The FP8 KV-cache profile does not run here. Its tests report SKIP with exit code 77, which CTest is
configured to treat as a skip rather than a failure. That is expected, not a regression.

This checkout also carries a local TCQ3 KV-cache profile that upstream does not have. See below.

## Build and test

```bash
# serving and CLI binaries, which is what serve.sh points at
cmake -S . -B build-sm86 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-15 -DCMAKE_C_COMPILER=/usr/bin/gcc-15
cmake --build build-sm86 -j

# tests. BUILD_TESTING is OFF by default, so they need their own directory
cmake -S . -B build-test -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-15 -DCMAKE_C_COMPILER=/usr/bin/gcc-15 \
  -DBUILD_TESTING=ON -DNINFER_BUILD_APPS=OFF
cmake --build build-test -j
```

Tests are separate executables registered with CTest, so one test means one target:

```bash
cmake --build build-test --target ninfer_softmax_attention_test -j
./build-test/tests/ninfer_softmax_attention_test     # runs the whole binary
ctest --test-dir build-test -R softmax_attention     # same, through CTest
ctest --test-dir build-test                          # everything
```

A test binary covers several suites. `ninfer_softmax_attention_test` runs causal-cache, packed,
and context attention. There is no finer selection flag, so narrowing below one binary means
editing its case list. `ninfer_add_op_test` compiles with `-fno-fast-math -ffp-contract=off`, which
is why host oracles reproduce device arithmetic closely enough to compare.

## Serving

`serve.sh` runs the full 262144 context with the TCQ3 cache, MTP, and vision, which is the
configuration this checkout is tuned for. It binds the Tailscale IP because `ninfer-serve` accepts
one `--host`, then adds 127.0.0.1 through a socat forwarder. Do not change it to `0.0.0.0`: the
firewall zone leaves 1025-65535/tcp open on the wired interface.

Perplexity, used as the quality gate for KV-cache work:

```bash
./build-sm86/apps/ninfer-perplexity models/qwen3_8_27b.ninfer \
  --kv-dtype tcq3 --context 2048 --stride 1024 --text <corpus.txt>
```

`--stride` must be strictly less than `--context`, and it defaults to 2048.

## The TCQ3 KV cache

A local port of the `turbo3_tcq` cache type from the llama.cpp TurboQuant fork. It is what makes
262144 context fit on 24 GiB, which INT8 cannot. It codes 3.25 bits for each value. A group of 128
values becomes an FP16 norm plus a 49-byte bitstream of six initial-state bits and 128 three-bit
symbols. Decode is random access, nine bits at a symbol boundary into a 512-entry codebook. Encode
is a 128-step Viterbi search, which is why it lives in the append Op and not in a kernel that has
to be fast.

`tcq3` is `DType::U8` throughout the ops layer. A 256-wide row is two independent 128-groups, so
the code plane has a leading extent of 98 bytes for each token and head, not 256. Three places
depend on that and must stay consistent: `d256_profile.h`, `decoder_state.cpp`, and any test that
builds a cache by hand.

| Concern | File |
|---|---|
| codebooks, decode, bit addressing | `src/ops/kv_cache/tcq3_g128_codec.cuh` (generated) |
| rotation and its sign pattern | `src/ops/kv_cache/tcq3_rotation.cuh`, `tcq3_sign_words.h` |
| Viterbi encoder | `src/ops/kv_cache/append/tcq3_encode.cuh` |
| decode attention, T of at most 6 | `.../causal_cache/small_t_tcq3.cuh` |
| prefill attention, T of 7 or more | `.../causal_cache/prompt_tcq3.cuh` |
| host oracle and 30 cases | `tests/ops/softmax_attention/causal_cache.cpp` |
| generators and CPU reference | `tools/tcq3/` |

`tcq3_g128_codec.cuh` and `tests/ops/softmax_attention/tcq3_host_tables.h` are generated. Edit
`tools/tcq3/gen-ninfer-codec-header.py` and re-run it. Do not hand-edit either header. The fitted
tables come from the fork through `tools/tcq3/reference/extract-constants.py`, which reads
`~/code/llama-turboquant/llama-cpp-turboquant` unless `TURBOQUANT_FORK` says otherwise. One
generator writes both headers so the kernels and the test oracle cannot disagree about a codebook
value.

### Rotation

Both K and V are stored rotated, which differs from INT8 and FP8 where only K is. The transform is
R = H . D, a sign flip followed by NInfer's `normalized_hadamard_d256_inplace`. R is orthogonal, so
scores are unchanged once Q takes the same R, but R is not its own inverse. Undoing it on the output
accumulator is H then D, which is why `kv_cache_tcq3_unrotate` exists separately from
`kv_cache_tcq3_rotate`. Anything touching Q preparation, V storage, or the output epilogue has to
move all of them together.

The signs are not decoration. The first Hadamard row is all ones. Without a sign flip the
per-channel mean of the keys therefore lands entirely in output coordinate 0, where it clips
against the outermost centroid. Perplexity is 6.352256 with randomization and 6.368235 without.
INT8 is 6.331016.

### Things that have bitten this code

An unguarded fallthrough to the BF16 kernel writes 512 bytes through a 98-byte row. It corrupts
memory without crashing, and the model still emits semi-coherent text, so it reads like poor codec
quality rather than a bug. Both the append path and the prompt path had this. When adding a KV
profile, check every dispatch, not just the one the failing case exercises.

Codebook lookups diverge across a warp. In `__constant__` memory that serializes up to 32 ways, and
it was the single largest cost in both attention kernels. The shared copy in the prompt kernel is
FP16 rather than FP32, because the dynamic arena is already full at 96 KiB and an FP32 copy misses
the 99 KiB opt-in limit by 1 KiB. The FP16 copy costs 0.086 percent perplexity.

Prefill is bandwidth bound, not math bound. The barrier structure already lets both decodes overlap
the matmuls. With the decode stubbed out, the whole remaining decode is worth 1.9 percent at 4060
prompt tokens and 10.2 percent at 16060. Retiling for more overlap has little left to win.
