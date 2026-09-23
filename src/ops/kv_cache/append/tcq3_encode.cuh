#pragma once

// TCQ 3-bit KV append: Viterbi encoder.
//
// One block of 512 threads encodes one group of 128 values, which is half of one row.
// The thread index is the trellis state index, so the add-compare-select step is one
// thread for each state. The search runs 128 sequential steps with two block barriers
// in each, which is why the block is wide and the grid is small.
//
// The layout of a block is a 6-bit initial state followed by 128 symbols of 3 bits,
// packed into 49 bytes, plus an FP16 norm in the scale plane. Decoding reads 9 bits at
// a symbol boundary, so the initial-state bits and the first symbols share a byte.
//
// Both K and V are rotated before quantization, because the codebooks were fitted on
// rotated data.
//
// The rotation is a sign-randomized H256, not the fork's sign-randomized H128. The width
// follows NInfer: the append Op's contract already names an H256 preparation for K, the
// INT8 profile already quantizes K after exactly that transform, and Q and the output
// accumulator need the matching transform in the attention kernel, where the H256 helper
// exists and the fork's does not. The sign randomization follows the fork, because it
// measures better than a plain Sylvester transform. See tcq3_rotation.cuh.
//
// The codebook stays valid because each group of 128 is normalized to unit length after
// the rotation, which makes the coordinates N(0, 1/128) whichever orthogonal transform
// produced them. The ordering differs from the fork, which normalizes before rotating.

#include "ops/kernel/paged_kv_address.cuh"
#include "ops/kv_cache/tcq3_rotation.cuh"
#include "ops/kv_cache/tcq3_g128_codec.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kTcq3EncodeThreads = 512;

// Block-wide scratch. The kernel owns it so that encoding K and V reuses one
// allocation instead of two.
struct Tcq3EncodeShared {
    float x[128];
    float cost_a[kKVCacheTcq3States];
    float cost_b[kKVCacheTcq3States];
    float pred_min[64];
    std::uint8_t backtrace[128 * 64];
    std::uint8_t symbols[128];
    float reduce[kTcq3EncodeThreads / 32];
    int reduce_idx[kTcq3EncodeThreads / 32];
    float group_norm;
    int initial_state;
};

// Encode one group. src holds 128 raw values, one for each dimension of the group.
// Writes 49 stream bytes and one FP16 norm.
__device__ inline void tcq3_encode_group_block(const float* src, bool is_key,
                                               Tcq3EncodeShared& shared, std::uint8_t* stream,
                                               __half* norm_out) {
    float* x = shared.x;
    float* cost_a = shared.cost_a;
    float* cost_b = shared.cost_b;
    float* pred_min = shared.pred_min;
    std::uint8_t* backtrace = shared.backtrace;
    std::uint8_t* symbols = shared.symbols;
    float* reduce = shared.reduce;
    int* reduce_idx = shared.reduce_idx;

    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const float* codebook = is_key ? kKVCacheTcq3CodebookK : kKVCacheTcq3CodebookV;

    if (tid < 128) { x[tid] = src[tid]; }
    __syncthreads();

    // L2 norm across the group.
    float partial = 0.0f;
    if (tid < 128) { partial = x[tid] * x[tid]; }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        partial += __shfl_down_sync(0xffffffffu, partial, offset);
    }
    if (lane == 0) { reduce[warp] = partial; }
    __syncthreads();
    if (tid == 0) {
        float total = 0.0f;
        for (int w = 0; w < kTcq3EncodeThreads / 32; ++w) { total += reduce[w]; }
        shared.group_norm = sqrtf(total);
    }
    __syncthreads();

    const float group_norm = shared.group_norm;
    const float inv_norm = group_norm > 1e-10f ? 1.0f / group_norm : 0.0f;
    if (tid < 128) { x[tid] *= inv_norm; }
    __syncthreads();

    // Viterbi forward pass over the right-shift trellis: next = (prev >> 3) | (out << 6).
    // The best predecessor depends only on the low six bits of the next state, so the
    // eight-way scan runs once for each of 64 predecessor classes instead of once for
    // each of 512 states.
    cost_a[tid] = 0.0f;
    __syncthreads();

    for (int t = 0; t < 128; ++t) {
        float* read = (t & 1) ? cost_b : cost_a;
        float* write = (t & 1) ? cost_a : cost_b;
        if (tid < 64) {
            const int base = tid << 3;
            float best = read[base];
            int best_p = 0;
#pragma unroll
            for (int p = 1; p < 8; ++p) {
                const float c = read[base | p];
                if (c < best) {
                    best = c;
                    best_p = p;
                }
            }
            pred_min[tid] = best;
            backtrace[t * 64 + tid] = static_cast<std::uint8_t>(best_p);
        }
        __syncthreads();
        const float xt = x[t];
        const float d = xt - codebook[tid];
        write[tid] = pred_min[tid & 0x3F] + d * d;
        __syncthreads();
    }

    // Step 127 wrote cost_a, because the loop ran an even number of steps.
    float my_cost = cost_a[tid];
    int my_idx = tid;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float other = __shfl_down_sync(0xffffffffu, my_cost, offset);
        const int other_idx = __shfl_down_sync(0xffffffffu, my_idx, offset);
        if (other < my_cost) {
            my_cost = other;
            my_idx = other_idx;
        }
    }
    if (lane == 0) {
        reduce[warp] = my_cost;
        reduce_idx[warp] = my_idx;
    }
    __syncthreads();
    if (tid == 0) {
        float best = reduce[0];
        int best_idx = reduce_idx[0];
        for (int w = 1; w < kTcq3EncodeThreads / 32; ++w) {
            if (reduce[w] < best) {
                best = reduce[w];
                best_idx = reduce_idx[w];
            }
        }
        // Backtrack. This is sequential by construction.
        int state = best_idx;
        for (int t = 127; t >= 0; --t) {
            symbols[t] = static_cast<std::uint8_t>(state >> 6);
            const int p = backtrace[t * 64 + (state & 0x3F)];
            state = ((state & 0x3F) << 3) | p;
        }
        shared.initial_state = state;
    }
    __syncthreads();

    // Norm correction. For t of at least 2 the state depends only on the previous three
    // symbols, so every step but the first two is independent of the initial state.
    const int initial_state = shared.initial_state;
    float recon = 0.0f;
    if (tid < 128) {
        int state;
        if (tid < 2) {
            state = initial_state;
            for (int t = 0; t <= tid; ++t) { state = (state >> 3) | (static_cast<int>(symbols[t]) << 6); }
        } else {
            state = (static_cast<int>(symbols[tid - 2]) & 0x7) |
                    ((static_cast<int>(symbols[tid - 1]) & 0x7) << 3) |
                    ((static_cast<int>(symbols[tid]) & 0x7) << 6);
        }
        const float c = codebook[state];
        recon = c * c;
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        recon += __shfl_down_sync(0xffffffffu, recon, offset);
    }
    if (lane == 0) { reduce[warp] = recon; }
    __syncthreads();
    if (tid == 0) {
        float total = 0.0f;
        for (int w = 0; w < kTcq3EncodeThreads / 32; ++w) { total += reduce[w]; }
        const float recon_norm = sqrtf(total);
        const float corrected = (recon_norm > 1e-10f) ? group_norm / recon_norm : group_norm;
        *norm_out = __float2half(corrected);
    }

    // Pack the bitstream. Each output byte is independent, so one thread owns each.
    if (tid < kKVCacheTcq3StreamBytes) {
        const int init_bits = (initial_state >> 3) & 0x3F;
        std::uint8_t packed = 0;
#pragma unroll
        for (int bit = 0; bit < 8; ++bit) {
            const int pos = tid * 8 + bit;
            int value = 0;
            if (pos < 6) {
                value = (init_bits >> pos) & 1;
            } else {
                const int sym_bit = pos - 6;
                const int sym = sym_bit / 3;
                if (sym < 128) { value = (symbols[sym] >> (sym_bit % 3)) & 1; }
            }
            packed |= static_cast<std::uint8_t>(value << bit);
        }
        stream[tid] = packed;
    }
    __syncthreads();
}

// One block for each (token, KV head, group). The block encodes K and then V for its
// group, reusing one scratch allocation.
template <typename Geometry, typename Metadata>
__launch_bounds__(kTcq3EncodeThreads) __global__
    void kv_cache_append_full_tcq3_kernel(const __nv_bfloat16* __restrict__ k,
                                          const __nv_bfloat16* __restrict__ v,
                                          const std::int32_t* __restrict__ positions,
                                          Metadata metadata, std::uint8_t* __restrict__ cache_k,
                                          std::uint8_t* __restrict__ cache_v,
                                          __half* __restrict__ scale_k,
                                          __half* __restrict__ scale_v, std::int32_t width) {
    __shared__ Tcq3EncodeShared shared;
    __shared__ float src[128];

    const int tokens = metadata.valid_tokens(width);
    const int units = tokens * Geometry::KVHeads * kKVCacheTcq3Groups;
    const int unit = static_cast<int>(blockIdx.x);
    if (unit >= units) { return; }

    const int group = unit % kKVCacheTcq3Groups;
    const int head_token = unit / kKVCacheTcq3Groups;
    const int kv_head = head_token % Geometry::KVHeads;
    const int token = head_token / Geometry::KVHeads;

    const int position = positions[0] + token;
    const std::int32_t* block_table = metadata.block_table();
    const int page = paged_kv_physical_page(block_table, position);
    const int page_off = position & kPagedKVPageMask;

    const std::int64_t stream_off =
        kv_cache_tcq3_stream_index<Geometry>(page, kv_head, group, page_off);
    const std::int64_t norm_off =
        kv_cache_tcq3_norm_index<Geometry>(page, kv_head, group, page_off);

    const int tid = static_cast<int>(threadIdx.x);
    const int d0 = group * kKVCacheTcq3Group;

    // One warp rotates the whole 256-wide row and keeps the half this block owns. The
    // rotation is warp-local and cheap next to the trellis search, so doing it in both
    // group blocks costs less than coordinating between them.
    const auto load_rotated_group = [&](const __nv_bfloat16* source) {
        if (tid < 32) {
            const int lane = tid;
            float values[8];
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int d = lane + 32 * r;
                values[r] = __bfloat162float(
                    source[kv_cache_tcq3_src_index<Geometry>(kv_head, d, token)]);
            }
            kv_cache_tcq3_rotate(values, lane);
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int d = lane + 32 * r;
                if (d >= d0 && d < d0 + kKVCacheTcq3Group) { src[d - d0] = values[r]; }
            }
        }
        __syncthreads();
    };

    load_rotated_group(k);
    tcq3_encode_group_block(src, true, shared, cache_k + stream_off, scale_k + norm_off);

    load_rotated_group(v);
    tcq3_encode_group_block(src, false, shared, cache_v + stream_off, scale_v + norm_off);
}

} // namespace ninfer::ops
