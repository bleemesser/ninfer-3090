#pragma once

// TCQ 3-bit KV append: Viterbi encoder.
//
// One warp encodes one group of 128 values, which is half of one row of K or V. Lane l owns
// the 16 trellis states 16l..16l+15 when it reads costs, and the 16 states whose low six bits
// are 2l or 2l+1 when it writes them. Those are exactly the states that the two predecessor
// classes it reduced can reach, so the only exchange in a step is one pass of costs through
// shared memory behind a __syncwarp. Earlier versions used a 512-thread block for each group
// with two block barriers in each of the 128 steps.
//
// The arithmetic, the reduction order of both norms, and every tie-break match the block
// encoder this replaced, so the bitstream and the norm are unchanged bit for bit. The host
// encoder in the attention test compares both exactly.
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

inline constexpr int kTcq3EncodeWarps   = 4;
inline constexpr int kTcq3EncodeThreads = kTcq3EncodeWarps * 32;

// One padding float after every 16 states keeps both the contiguous reads and the strided
// writes of the cost exchange free of bank conflicts.
inline constexpr int kTcq3EncodeCostStride = kKVCacheTcq3States + kKVCacheTcq3States / 16;

__device__ __forceinline__ int tcq3_encode_cost_index(int state) { return state + (state >> 4); }

// Scratch for one warp.
struct Tcq3EncodeWarpShared {
    float x[kKVCacheTcq3Group];
    float cost[2][kTcq3EncodeCostStride];
    // Predecessor choices for classes 2l and 2l+1 in the low and high nibble of byte l.
    std::uint8_t backtrace[kKVCacheTcq3Group][32];
    std::uint8_t symbols[kKVCacheTcq3Group];
};

// Sum of v over the 128 values held four to a lane (value lane + 32w in slot w). Each slot
// is reduced with the shuffle tree the 128-thread encoder used for one warp, and the four
// slot sums are added in warp order, which reproduces that encoder's float result exactly.
__device__ __forceinline__ float tcq3_encode_sum128(const float (&v)[4]) {
    float total = 0.0f;
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        float partial = v[w];
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            partial += __shfl_down_sync(0xffffffffu, partial, offset);
        }
        total += __shfl_sync(0xffffffffu, partial, 0);
    }
    return total;
}

// Encode one group. The warp's shared.x must hold the 128 rotated values. Writes 49 stream
// bytes and one FP16 norm.
__device__ inline void tcq3_encode_group_warp(bool is_key, Tcq3EncodeWarpShared& shared,
                                              std::uint8_t* stream, __half* norm_out) {
    const int lane        = static_cast<int>(threadIdx.x) & 31;
    const float* codebook = is_key ? kKVCacheTcq3CodebookK : kKVCacheTcq3CodebookV;

    float squares[4];
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        const float value = shared.x[lane + 32 * w];
        squares[w]        = value * value;
    }
    const float group_norm = sqrtf(tcq3_encode_sum128(squares));
    const float inv_norm   = group_norm > 1e-10f ? 1.0f / group_norm : 0.0f;
#pragma unroll
    for (int w = 0; w < 4; ++w) { shared.x[lane + 32 * w] *= inv_norm; }

    // Written states: low six bits 2l + h, high three bits o, so state = 2l + h + 64o.
    float centroid[2][8];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
#pragma unroll
        for (int o = 0; o < 8; ++o) { centroid[h][o] = codebook[2 * lane + h + 64 * o]; }
    }
#pragma unroll
    for (int i = 0; i < 16; ++i) { shared.cost[0][tcq3_encode_cost_index(16 * lane + i)] = 0.0f; }
    __syncwarp();

    // Viterbi forward pass over the right-shift trellis: next = (prev >> 3) | (out << 6).
    // The best predecessor depends only on the low six bits of the next state, so the
    // eight-way scan runs once for each of 64 predecessor classes.
#pragma unroll 2
    for (int t = 0; t < kKVCacheTcq3Group; ++t) {
        const float* read = shared.cost[t & 1];
        float* write      = shared.cost[(t & 1) ^ 1];
        float pred_min[2];
        int pred_idx[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int base = 16 * lane + 8 * h;
            float best     = read[tcq3_encode_cost_index(base)];
            int best_p     = 0;
#pragma unroll
            for (int p = 1; p < 8; ++p) {
                const float c = read[tcq3_encode_cost_index(base | p)];
                if (c < best) {
                    best   = c;
                    best_p = p;
                }
            }
            pred_min[h] = best;
            pred_idx[h] = best_p;
        }
        shared.backtrace[t][lane] = static_cast<std::uint8_t>(pred_idx[0] | (pred_idx[1] << 4));
        const float xt = shared.x[t];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
#pragma unroll
            for (int o = 0; o < 8; ++o) {
                const float d = xt - centroid[h][o];
                write[tcq3_encode_cost_index(2 * lane + h + 64 * o)] = pred_min[h] + d * d;
            }
        }
        __syncwarp();
    }

    // Step 127 wrote cost[0]. Lowest total cost wins, and the lowest state index breaks ties.
    float my_cost = shared.cost[0][tcq3_encode_cost_index(16 * lane)];
    int my_idx    = 16 * lane;
#pragma unroll
    for (int i = 1; i < 16; ++i) {
        const float c = shared.cost[0][tcq3_encode_cost_index(16 * lane + i)];
        if (c < my_cost) {
            my_cost = c;
            my_idx  = 16 * lane + i;
        }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float other     = __shfl_down_sync(0xffffffffu, my_cost, offset);
        const int other_idx   = __shfl_down_sync(0xffffffffu, my_idx, offset);
        if (other < my_cost || (other == my_cost && other_idx < my_idx)) {
            my_cost = other;
            my_idx  = other_idx;
        }
    }

    // Backtrack. This is sequential by construction.
    int initial_state = 0;
    if (lane == 0) {
        int state = my_idx;
        for (int t = kKVCacheTcq3Group - 1; t >= 0; --t) {
            shared.symbols[t]  = static_cast<std::uint8_t>(state >> 6);
            const int cls      = state & 0x3F;
            const int p        = (shared.backtrace[t][cls >> 1] >> (4 * (cls & 1))) & 0x7;
            state              = (cls << 3) | p;
        }
        initial_state = state;
    }
    initial_state = __shfl_sync(0xffffffffu, initial_state, 0);
    __syncwarp();

    // Norm correction. For t of at least 2 the state depends only on the previous three
    // symbols, so every step but the first two is independent of the initial state.
    float recon[4];
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        const int t = lane + 32 * w;
        int state;
        if (t < 2) {
            state = initial_state;
            for (int u = 0; u <= t; ++u) {
                state = (state >> 3) | (static_cast<int>(shared.symbols[u]) << 6);
            }
        } else {
            state = (static_cast<int>(shared.symbols[t - 2]) & 0x7) |
                    ((static_cast<int>(shared.symbols[t - 1]) & 0x7) << 3) |
                    ((static_cast<int>(shared.symbols[t]) & 0x7) << 6);
        }
        const float c = codebook[state];
        recon[w]      = c * c;
    }
    const float recon_norm = sqrtf(tcq3_encode_sum128(recon));
    if (lane == 0) {
        const float corrected = (recon_norm > 1e-10f) ? group_norm / recon_norm : group_norm;
        *norm_out             = __float2half(corrected);
    }

    // Pack the bitstream. Each output byte is independent, so one lane owns each.
    const int init_bits = (initial_state >> 3) & 0x3F;
    for (int byte = lane; byte < kKVCacheTcq3StreamBytes; byte += 32) {
        std::uint8_t packed = 0;
#pragma unroll
        for (int bit = 0; bit < 8; ++bit) {
            const int pos = byte * 8 + bit;
            int value     = 0;
            if (pos < 6) {
                value = (init_bits >> pos) & 1;
            } else {
                const int sym_bit = pos - 6;
                const int sym     = sym_bit / 3;
                if (sym < kKVCacheTcq3Group) { value = (shared.symbols[sym] >> (sym_bit % 3)) & 1; }
            }
            packed |= static_cast<std::uint8_t>(value << bit);
        }
        stream[byte] = packed;
    }
    __syncwarp();
}

// One warp for each (token, KV head, group, plane). Consecutive warps take K then V of the
// same group, so both read the same rows of the source.
template <typename Geometry, typename Metadata>
__launch_bounds__(kTcq3EncodeThreads) __global__
    void kv_cache_append_full_tcq3_kernel(const __nv_bfloat16* __restrict__ k,
                                          const __nv_bfloat16* __restrict__ v,
                                          const std::int32_t* __restrict__ positions,
                                          Metadata metadata, std::uint8_t* __restrict__ cache_k,
                                          std::uint8_t* __restrict__ cache_v,
                                          __half* __restrict__ scale_k,
                                          __half* __restrict__ scale_v, std::int32_t width) {
    __shared__ Tcq3EncodeWarpShared shared[kTcq3EncodeWarps];

    const int warp   = static_cast<int>(threadIdx.x) >> 5;
    const int lane   = static_cast<int>(threadIdx.x) & 31;
    const int tokens = metadata.valid_tokens(width);
    const int units  = tokens * Geometry::KVHeads * kKVCacheTcq3Groups * 2;
    const int unit   = static_cast<int>(blockIdx.x) * kTcq3EncodeWarps + warp;
    if (unit >= units) { return; }

    const bool is_key    = (unit & 1) == 0;
    const int group_unit = unit >> 1;
    const int group      = group_unit % kKVCacheTcq3Groups;
    const int head_token = group_unit / kKVCacheTcq3Groups;
    const int kv_head    = head_token % Geometry::KVHeads;
    const int token      = head_token / Geometry::KVHeads;

    const int position              = positions[0] + token;
    const std::int32_t* block_table = metadata.block_table();
    const int page                  = paged_kv_physical_page(block_table, position);
    const int page_off              = position & kPagedKVPageMask;

    // The warp rotates the whole 256-wide row and keeps the half it owns.
    const __nv_bfloat16* source = is_key ? k : v;
    const int d0                = group * kKVCacheTcq3Group;
    Tcq3EncodeWarpShared& mine  = shared[warp];
    {
        float values[8];
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            values[r] = __bfloat162float(
                source[kv_cache_tcq3_src_index<Geometry>(kv_head, lane + 32 * r, token)]);
        }
        kv_cache_tcq3_rotate(values, lane);
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int d = lane + 32 * r;
            if (d >= d0 && d < d0 + kKVCacheTcq3Group) { mine.x[d - d0] = values[r]; }
        }
    }
    __syncwarp();

    const std::int64_t stream_off =
        kv_cache_tcq3_stream_index<Geometry>(page, kv_head, group, page_off);
    const std::int64_t norm_off = kv_cache_tcq3_norm_index<Geometry>(page, kv_head, group, page_off);
    tcq3_encode_group_warp(is_key, mine, (is_key ? cache_k : cache_v) + stream_off,
                           (is_key ? scale_k : scale_v) + norm_off);
}

} // namespace ninfer::ops
