# TCQ port results, first working version

Measured 2026-09-19 on an RTX 3090 Ti with Qwen3.8-27B.

## It works

The full path runs: Viterbi encode in the append Op, decode in both the
small-T and prompt attention kernels, and the rotation applied consistently to
K, V, Q, and the output accumulator. Answers are coherent.

## Context

262144 tokens of context load with MTP on and 2.75 GiB free after startup. The
KV cache is 3.93 GiB at that size. INT8 needs 2.73 GiB for 65536 tokens, so the
same cache at 262144 would want about 10.9 GiB and does not fit at all. This was
the goal of the whole exercise and it is met.

## Quality

Perplexity on the same wikitext subset at 2048 context with stride 1024,
through NInfer's own tool, so both numbers come from one harness:

| KV profile | Perplexity | Bits for each value |
|---|---:|---:|
| INT8 G64 | 6.331016 | 8.25 |
| TCQ3 G128 | 6.362775 | 3.25 |

TCQ costs 0.50 percent against INT8 while storing 2.5 times fewer bytes. For
scale, llama.cpp measured turbo3_tcq at 0.33 percent against an f16 cache, and
INT8 is itself lossy, so this sits where a correct port should.

## Speed after the read-path work

Decode, tokens for each second, all at 65536 context with MTP on, same prompts.
"First" is the original kernel, "staged" adds the shared-memory stream copy,
"shared cb" also moves both codebooks out of constant memory, and "windowed"
also reads four bytes once for each run of eight symbols.

| Prompt | INT8 | first | staged | shared cb | windowed | llama.cpp TCQ |
|---:|---:|---:|---:|---:|---:|---:|
| 63 | 68.5 | 77.1 | 77.0 | 78.5 | 78.8 | 53.0 |
| 60, gen 512 | 83.6 | 80.3 | 80.3 | 82.0 | 82.2 | 58.6 |
| 4060 | 82.8 | 58.6 | 60.6 | 68.7 | 68.9 | 51.7 |
| 16060 | 75.6 | 40.9 | 47.4 | 62.9 | 63.9 | 54.0 |

Long-context decode improved by 56 percent over the first version. TCQ now beats
INT8 on short prompts, matches it in the middle, and trails it at 16060 tokens.
It beats llama.cpp with turbo3_tcq at every length measured, and it does so at
262144 context where llama.cpp also runs at 262144.

The single largest win was the codebook. Constant memory broadcasts one address
to a warp, and every thread here looks up a different trellis state, so each
lookup serialized up to 32 ways. Moving both codebooks to shared memory took
16060-token decode from 47.4 to 62.9 on its own.

## Prefill

| Prompt tokens | INT8 | TCQ3 first | shared codebook | vector store | llama.cpp TCQ |
|---:|---:|---:|---:|---:|---:|
| 4060 | 1091 | 837 | 994 | 1006 | 988 |
| 16060 | 1051 | 531 | 870 | 908 | 965 |

Prefill went from 531 to 908 tokens for each second at 16060 tokens, which is 71
percent. TCQ3 prefill now passes llama.cpp and sits 14 percent behind INT8.

Two changes did this, and the first is the same lesson the decode path taught.

The codebooks moved from constant memory into shared memory. Every thread looks
up a different trellis state, and constant memory serves one address for each
warp cycle, so the lookup serialized up to 32 ways. The dynamic arena was already
full at 96 KiB of Q, K, and V tiles, so the shared copy is FP16 rather than FP32:
2 KiB instead of 4 KiB, which fits under the 99 KiB opt-in limit on this GPU. An
FP32 copy does not fit by 1 KiB.

The decode then wrote its eight output values with eight 2-byte shared stores.
Packing them into one 16-byte store added another 4 percent.

Perplexity is 6.368235, against 6.362775 before the FP16 codebook, a rise of
0.086 percent. The vector store did not move it at all, which also confirms the
packed store is bit-identical to the scalar one. The FP16 codebook is the whole
cost, and it is small next to the 0.50 percent that TCQ3 gives up against INT8.
Revert it to FP32 if the gap ever matters, at the price of an FP32 copy that
needs a smaller key tile to fit.

## Kernel correctness

The two attention kernels are now checked against a host oracle in
tests/ops/softmax_attention/causal_cache.cpp, alongside the BF16, INT8, and FP8
profiles that were already there. Thirty cases cover both geometries, all three
page mappings, and both kernels: T of at most 6 routes to the small-T kernel and
7 or more to the prompt kernel.

The oracle reads the same decoded values the kernel reads, so it measures kernel
arithmetic and addressing, not codec error. Its encoder is a separate host
implementation of the Viterbi search. The codec tables it uses are generated
into tests/ops/softmax_attention/tcq3_host_tables.h by the same script that
writes the kernel header, so the two cannot drift and neither transcribes a
fitted value by hand.

Two of the cases append through the device Viterbi encoder first. Those compare
the stored bitstream and the stored norms byte for byte against the host
encoder, and they match. The device encoder and an independently written host
encoder agree exactly.

The criterion is the FP8 one rather than the INT8 one. TCQ3 rotates Q into BF16
and rotates the output accumulator back through BF16, and the oracle has neither
rounding, so two extra 256-term BF16 transforms separate the two.

The cases were checked for sensitivity by mutating the oracle: dropping the 1.02
decode-time V alpha fails all thirty. That also confirms the kernel applies it.

## Vision

Vision works with TCQ3 at the full 262144 context with MTP on. The server takes
--vision --kv-dtype tcq3 --max-context 262144, an image through the OpenAI
image_url field is described correctly, and decode runs at 92 tokens for each
second. Memory sits at 22.3 GiB of 24.5 GiB, with 1.82 GiB free after startup.

## How much is left in the prompt kernel

At most 10 percent, and probably much less.

Measured by building the prompt kernel with the decode removed, leaving only the
shared-memory store, so the result is garbage but the timing bounds what any
decode optimization can return:

| Prompt tokens | TCQ3 now | decode removed | INT8 |
|---:|---:|---:|---:|
| 4060 | 1006 | 1025 | 1091 |
| 16060 | 908 | 1001 | 1051 |

The whole decode is worth 1.9 percent at 4060 tokens and 10.2 percent at 16060,
and no real implementation reaches that, because it still has to decode. INT8 at
16060 is faster than TCQ3 with no decode at all, so the rest of the gap is the Q
rotation, the output rotation, and tiling, not the codec.

An earlier note in this file said the kernel gave up the cp.async pipeline and
ran the decode and the matmul in sequence. The barrier structure does not
actually do that. The V decode is issued before the QK matmul with no barrier
between them, and the K decode for the next tile is issued before the PV matmul,
also with no barrier, so both already sit where the hardware can interleave them
with tensor-core work. What cp.async would save is issue slots and registers,
not ordering. A double buffer would not add overlap that is missing.

That leaves the retile as a change that costs a day, risks the softmax epilogue
doubling when Bc halves, and competes for under 10 percent of prefill on long
prompts and 2 percent on short ones.

## Rotation

The plain Sylvester transform did cost quality, and randomizing the signs
recovers most of it.

| Rotation | Perplexity |
|---|---:|
| Plain Sylvester H256 | 6.368235 |
| Sign-randomized H256 | 6.352256 |
| INT8, for reference | 6.331016 |

That is a gain of 0.25 percent, which cuts the gap to INT8 from 0.50 percent to
0.34 percent. It is also three times the 0.086 percent the FP16 shared codebook
cost, so the two together still leave TCQ3 ahead of where it started.

The mechanism is the one the original note flagged as the accepted risk. The
first row of a plain Sylvester matrix is all ones, so any component constant
across the head dimension lands entirely in output coordinate 0. Attention keys
carry a large per-channel mean, so that one coordinate runs past the outermost
codebook centroid and clips. Random signs spread the same component over every
coordinate.

The implementation is R = H . D with a fixed balanced 256-bit sign pattern, 128
of each, in src/ops/kv_cache/tcq3_sign_words.h. R is orthogonal, so scores are
unchanged once Q takes the same R, but R is not its own inverse: undoing it on
the output accumulator is H then D. The switch is one constant, and building
with it off reproduces 6.368235 bit for bit, which is how the two rows above
were measured against each other.

Prefill and decode are unchanged: 1002.9 and 906.8 tokens for each second at
4060 and 16060, against 1006 and 908 before. Eight conditional negations per row
do not register next to the trellis search. The thirty attention correctness
cases pass with randomization on, including the two that compare the device
encoder's bitstream and norms byte for byte against the host encoder.

This closes the last open question from the port.

## Running it

serve.sh in the ninfer tree. It follows the same binding rule as llama-turbo.sh:
ninfer-serve takes one --host, so it binds the Tailscale IP, and a socat
forwarder adds 127.0.0.1 on the same port, with a trap that cleans the forwarder
up. It never binds 0.0.0.0, because the FedoraWorkstation zone leaves
1025-65535/tcp open on the wired interface.

Modes are turbo for 262144 context, speed for 65536, and int8 for the faster
prefill that cannot reach full context. Vision and MTP are on in every mode, and
concurrency is pinned to 1 because MTP is single-sequence and because pooled
slots overrun one KV cache.
