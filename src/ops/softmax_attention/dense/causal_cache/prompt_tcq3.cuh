#pragma once

// TCQ3-cache causal prompt kernel.
//
// One CTA owns 64 query rows of one query head and walks the cached history in 32-key tiles.
// Sixteen warps split the work the same way the INT8 kernel does. Four producer warps compute
// S = Q K^T and the online softmax for 16 rows each while the twelve worker warps decode the V
// tile. After a barrier every warp accumulates O += P V for 16 rows and a 64-dimension slice and
// then decodes a slice of the next K tile.
//
// The compressed tiles are small (32 keys x 98 bytes for each plane), so they are prefetched
// with cp.async three tiles ahead and the trellis decode reads only shared memory. Decoding
// straight from global memory left the decode latency bound on four warps and cost more than
// the tensor-core work it fed.
//
// Q, K, and V are all stored in the rotated domain, so Q takes the forward rotation on load
// and each output row takes the inverse before it is written.

#include <math_constants.h>

#include "ops/kv_cache/tcq3_g128_codec.cuh"
#include "ops/kv_cache/tcq3_rotation.cuh"
#include "ops/softmax_attention/dense/causal_cache/prompt_common.cuh"

#include <cuda_fp16.h>

namespace ninfer::ops {

inline constexpr int kCausalPromptTcq3Warps         = 16;
inline constexpr int kCausalPromptTcq3Threads       = kCausalPromptTcq3Warps * 32;
inline constexpr int kCausalPromptTcq3Br            = 64;
inline constexpr int kCausalPromptTcq3Bc            = 32;
inline constexpr int kCausalPromptTcq3RowTiles      = kCausalPromptTcq3Br / 16;
inline constexpr int kCausalPromptTcq3DSlices       = kCausalPromptTcq3Warps / kCausalPromptTcq3RowTiles;
inline constexpr int kCausalPromptTcq3Producers     = kCausalPromptTcq3RowTiles;
inline constexpr int kCausalPromptTcq3RawStages     = 3;
inline constexpr int kCausalPromptTcq3OutRowStride  = kCausalPromptHeadDim + 8;

inline constexpr int kCausalPromptTcq3QBytes = kCausalPromptTcq3Br * kCausalPromptHeadDim * 2;
inline constexpr int kCausalPromptTcq3KBytes = kCausalPromptTcq3Bc * kCausalPromptHeadDim * 2;
inline constexpr int kCausalPromptTcq3VBytes = kCausalPromptTcq3Bc * kCausalPromptHeadDim * 2;
inline constexpr int kCausalPromptTcq3PBytes = kCausalPromptTcq3Br * kCausalPromptTcq3Bc * 2;
inline constexpr int kCausalPromptTcq3RawCodeBytes = kCausalPromptTcq3Bc * kKVCacheTcq3RowBytes;
inline constexpr int kCausalPromptTcq3RawNormBytes =
    kCausalPromptTcq3Bc * kKVCacheTcq3Groups * static_cast<int>(sizeof(__half));
inline constexpr int kCausalPromptTcq3RawStageBytes =
    2 * kCausalPromptTcq3RawCodeBytes + 2 * kCausalPromptTcq3RawNormBytes;
inline constexpr int kCausalPromptTcq3CodebookBytes =
    2 * kKVCacheTcq3States * static_cast<int>(sizeof(float));
inline constexpr int kCausalPromptTcq3StatsBytes =
    2 * kCausalPromptTcq3Br * static_cast<int>(sizeof(float));
inline constexpr int kCausalPromptTcq3SmemBytes =
    kCausalPromptTcq3QBytes + kCausalPromptTcq3KBytes + kCausalPromptTcq3VBytes +
    kCausalPromptTcq3PBytes + kCausalPromptTcq3RawStages * kCausalPromptTcq3RawStageBytes +
    kCausalPromptTcq3CodebookBytes + kCausalPromptTcq3StatsBytes;

static_assert(kCausalPromptTcq3DSlices == 4);
static_assert(kCausalPromptTcq3RawCodeBytes % 16 == 0 && kCausalPromptTcq3RawNormBytes % 16 == 0);
// The FP32 output staging reuses the contiguous Q, K, V, and P tiles once the key loop ends.
static_assert(kCausalPromptTcq3Br * kCausalPromptTcq3OutRowStride * 4 <=
              kCausalPromptTcq3QBytes + kCausalPromptTcq3KBytes + kCausalPromptTcq3VBytes +
                  kCausalPromptTcq3PBytes);
static_assert(kCausalPromptTcq3SmemBytes <= 99 * 1024);

// XOR swizzle for the [64][32] FP16 P tile. A row is four 16-byte chunks, so two rows share
// one 128-byte bank line; permuting by row pairs keeps an 8-row ldmatrix conflict-free.
__device__ __forceinline__ int causal_prompt_tcq3_p_swz(int row, int col) {
    return (((col >> 3) ^ ((row >> 1) & 3)) << 3) | (col & 7);
}

// Issue the cp.async copies for one 32-key tile of compressed K and V into a raw stage. The
// layout inside a page is [token][98 bytes] for the codes and [token][2] for the norms, so a
// half-page tile is one contiguous run in each plane.
template <typename Geometry>
__device__ __forceinline__ void
causal_prompt_tcq3_issue_raw(unsigned char* stage, const std::uint8_t* cache_k,
                             const std::uint8_t* cache_v, const __half* norm_k,
                             const __half* norm_v, const std::int32_t* block_table, int kv_head,
                             int k0, int tid) {
    constexpr int CodeChunks = kCausalPromptTcq3RawCodeBytes / 16;
    constexpr int NormChunks = kCausalPromptTcq3RawNormBytes / 16;
    const int physical_page  = block_table[k0 >> kPagedKVPageShift];
    const int page_off       = k0 & kPagedKVPageMask;
    if (tid < 2 * CodeChunks) {
        const bool is_v = tid >= CodeChunks;
        const int chunk = is_v ? tid - CodeChunks : tid;
        const std::uint8_t* base =
            (is_v ? cache_v : cache_k) +
            kv_cache_tcq3_stream_index<Geometry>(physical_page, kv_head, 0, page_off);
        cp_async<16, Cache::cg>(stage + (is_v ? kCausalPromptTcq3RawCodeBytes : 0) + chunk * 16,
                                base + chunk * 16);
    } else if (tid < 2 * CodeChunks + 2 * NormChunks) {
        const int item  = tid - 2 * CodeChunks;
        const bool is_v = item >= NormChunks;
        const int chunk = is_v ? item - NormChunks : item;
        const __half* base =
            (is_v ? norm_v : norm_k) +
            kv_cache_tcq3_norm_index<Geometry>(physical_page, kv_head, 0, page_off);
        cp_async<16, Cache::cg>(stage + 2 * kCausalPromptTcq3RawCodeBytes +
                                    (is_v ? kCausalPromptTcq3RawNormBytes : 0) + chunk * 16,
                                reinterpret_cast<const unsigned char*>(base) + chunk * 16);
    }
}

// Decode eight consecutive values of one key from a staged tile into a swizzled 16-bit smem
// tile. K is written as BF16 for the QK product, V as FP16 for the PV product.
template <bool IsKey>
__device__ __forceinline__ void
causal_prompt_tcq3_decode8(unsigned short* dst_tile, const unsigned char* stage,
                           const float* codebook, int task, int k0, int max_query_abs) {
    constexpr int D  = kCausalPromptHeadDim;
    const int key_l  = task >> 5;
    const int c      = task & 31;
    const int d      = c << 3;
    unsigned short* p = &dst_tile[key_l * D + causal_prompt_swz(key_l, d)];
    if (k0 + key_l > max_query_abs) {
        store_vec(p, make_int4(0, 0, 0, 0));
        return;
    }
    const int group = c >> 4;
    const unsigned char* codes =
        stage + (IsKey ? 0 : kCausalPromptTcq3RawCodeBytes);
    const __half* norms = reinterpret_cast<const __half*>(
        stage + 2 * kCausalPromptTcq3RawCodeBytes + (IsKey ? 0 : kCausalPromptTcq3RawNormBytes));
    // A run of eight symbols starts on a byte boundary and spans 30 bits.
    const int byte_off = key_l * kKVCacheTcq3RowBytes + group * kKVCacheTcq3StreamBytes +
                         3 * (c & 15);
    const std::uint32_t* words =
        reinterpret_cast<const std::uint32_t*>(codes + (byte_off & ~3));
    const std::uint32_t word = __funnelshift_r(words[0], words[1], (byte_off & 3) * 8);
    const float norm =
        kv_cache_tcq3_norm_scale(norms[key_l * kKVCacheTcq3Groups + group], IsKey);
    float dec[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) { dec[i] = codebook[(word >> (3 * i)) & 0x1FFu] * norm; }
    unsigned packed[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        if constexpr (IsKey) {
            packed[i] = pack_bf16x2(dec[2 * i], dec[2 * i + 1]);
        } else {
            const __half2 h = __floats2half2_rn(dec[2 * i], dec[2 * i + 1]);
            packed[i]       = *reinterpret_cast<const unsigned*>(&h);
        }
    }
    store_vec(p, make_int4(static_cast<int>(packed[0]), static_cast<int>(packed[1]),
                           static_cast<int>(packed[2]), static_cast<int>(packed[3])));
}

// FlashAttention-2 forward over the paged TCQ3 cache, one CTA for each (64-row query block,
// query head). Query row i sees keys [0, base_pos + i] (bottom-right causal alignment).
template <typename Geometry, typename Metadata>
__global__ __launch_bounds__(kCausalPromptTcq3Threads, 1) void causal_attention_prompt_tcq3_kernel(
    const __nv_bfloat16* __restrict__ q, const std::uint8_t* __restrict__ cache_k,
    const std::uint8_t* __restrict__ cache_v, const __half* __restrict__ cache_k_norm,
    const __half* __restrict__ cache_v_norm, Metadata metadata,
    const std::int32_t* __restrict__ positions, float scale, __nv_bfloat16* __restrict__ out,
    std::int32_t width) {
    constexpr int D             = kCausalPromptHeadDim;
    constexpr int Br            = kCausalPromptTcq3Br;
    constexpr int Bc            = kCausalPromptTcq3Bc;
    constexpr int Threads       = kCausalPromptTcq3Threads;
    constexpr int Producers     = kCausalPromptTcq3Producers;
    constexpr int Stages        = kCausalPromptTcq3RawStages;
    constexpr int QKNt          = Bc / 8;
    constexpr int QKKs          = D / 16;
    constexpr int PVNtPerWarp   = D / (kCausalPromptTcq3DSlices * 8);
    constexpr int PVKs          = Bc / 16;
    constexpr int TileTasks     = Bc * (D / 8);
    constexpr int WorkerThreads = Threads - Producers * 32;
    constexpr int OutStride     = kCausalPromptTcq3OutRowStride;
    constexpr float Log2E       = 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffu;

    static_assert(QKNt % 2 == 0 && PVNtPerWarp % 2 == 0);

    extern __shared__ __align__(16) unsigned char smem_raw[];
    auto* q_s  = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    auto* k_s  = reinterpret_cast<unsigned short*>(smem_raw + kCausalPromptTcq3QBytes);
    auto* v_s  = reinterpret_cast<unsigned short*>(smem_raw + kCausalPromptTcq3QBytes +
                                                   kCausalPromptTcq3KBytes);
    auto* p_s  = reinterpret_cast<__half*>(smem_raw + kCausalPromptTcq3QBytes +
                                          kCausalPromptTcq3KBytes + kCausalPromptTcq3VBytes);
    auto* raw  = smem_raw + kCausalPromptTcq3QBytes + kCausalPromptTcq3KBytes +
                kCausalPromptTcq3VBytes + kCausalPromptTcq3PBytes;
    auto* cb_k = reinterpret_cast<float*>(raw + Stages * kCausalPromptTcq3RawStageBytes);
    auto* cb_v = cb_k + kKVCacheTcq3States;
    auto* alpha_s   = cb_v + kKVCacheTcq3States;
    auto* final_l_s = alpha_s + Br;
    auto* out_s     = reinterpret_cast<float*>(smem_raw);

    const int q_block = static_cast<int>(blockIdx.x);
    const int q_head  = static_cast<int>(blockIdx.y);
    const int tid     = static_cast<int>(threadIdx.x);
    const int warp    = tid >> 5;
    const int lane    = tid & 31;
    const int q0      = q_block * Br;
    const int kv_head = q_head / Geometry::GroupSize;
    const int tokens  = metadata.valid_tokens(width);

    if (q_head >= Geometry::QHeads || q0 >= width) { return; }
    if (q0 >= tokens) {
        causal_prompt_zero_output_rows<Geometry>(out, q_head, q0, min(q0 + Br, width), tid,
                                                 Threads);
        return;
    }
    const int base_pos              = positions[0];
    const std::int32_t* block_table = metadata.block_table();
    const int tile_rows             = min(Br, tokens - q0);
    const int max_query_abs         = base_pos + q0 + tile_rows - 1;
    const int key_blocks            = max_query_abs / Bc + 1;

    auto stage_ptr = [&](int kb) { return raw + (kb % Stages) * kCausalPromptTcq3RawStageBytes; };
    auto issue_raw = [&](int kb) {
        if (kb < key_blocks) {
            causal_prompt_tcq3_issue_raw<Geometry>(stage_ptr(kb), cache_k, cache_v, cache_k_norm,
                                                   cache_v_norm, block_table, kv_head, kb * Bc,
                                                   tid);
        }
        cp_commit();
    };

#pragma unroll
    for (int s = 0; s < Stages; ++s) { issue_raw(s); }

    for (int i = tid; i < kKVCacheTcq3States; i += Threads) {
        cb_k[i] = kKVCacheTcq3CodebookK[i];
        cb_v[i] = kKVCacheTcq3CodebookV[i];
    }

    // Rotate Q into the stored domain. One warp owns each row, as in the append Op.
    for (int row = warp; row < Br; row += kCausalPromptTcq3Warps) {
        float values[8];
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            values[r] = row < tile_rows
                            ? __bfloat162float(
                                  q[causal_prompt_q_index<Geometry>(q_head, lane + 32 * r, q0 + row)])
                            : 0.0f;
        }
        kv_cache_tcq3_rotate(values, lane);
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int d                          = lane + 32 * r;
            q_s[row * D + causal_prompt_swz(row, d)] = __float2bfloat16(values[r]);
        }
    }

    cp_wait<Stages - 2>();
    __syncthreads();
    for (int task = tid; task < TileTasks; task += Threads) {
        causal_prompt_tcq3_decode8<true>(k_s, stage_ptr(0), cb_k, task, 0, max_query_abs);
    }
    __syncthreads();

    const int gid      = lane >> 2;
    const int lid      = lane & 3;
    const int a_mat    = lane >> 3;
    const int a_rin    = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int a_coloff = (a_mat >> 1) << 3;
    const int b_rin    = lane & 7;
    const int b_koff   = ((lane >> 3) & 1) << 3;

    // Producer ldmatrix bases; see causal_prompt_swz_addr.
    const int prod_row0        = (warp & (Producers - 1)) * 16;
    const unsigned q_lane_base = smem_addr(q_s) + static_cast<unsigned>((prod_row0 + a_rowoff) * 512);
    const unsigned q_as        = static_cast<unsigned>((a_mat >> 1) << 4);
    const unsigned ab_r        = static_cast<unsigned>(a_rin << 4);
    const unsigned k_lane_base = smem_addr(k_s) + static_cast<unsigned>(b_rin * 512) +
                                 (static_cast<unsigned>(lane >> 4) << 12);
    const unsigned k_as        = static_cast<unsigned>((b_koff >> 3) << 4);

    // PV ldmatrix bases. Row tile and dimension slice of this warp.
    const int row_tile = warp % kCausalPromptTcq3RowTiles;
    const int d_slice  = warp / kCausalPromptTcq3RowTiles;
    const int pv_row0  = row_tile * 16;
    const unsigned v_lane_base = smem_addr(v_s) + static_cast<unsigned>(((lane >> 3) & 1) * 4096) +
                                 static_cast<unsigned>(b_rin * 512) +
                                 static_cast<unsigned>(d_slice * PVNtPerWarp * 16);
    const unsigned v_as = static_cast<unsigned>((lane >> 4) << 4);

    float acc[PVNtPerWarp][4];
#pragma unroll
    for (int n = 0; n < PVNtPerWarp; ++n) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { acc[n][i] = 0.0f; }
    }
    float m0 = -CUDART_INF_F, m1 = -CUDART_INF_F, l0 = 0.0f, l1 = 0.0f;
    const float scale_l2 = scale * Log2E;

    for (int kb = 0; kb < key_blocks; ++kb) {
        const int k0 = kb * Bc;
        if (warp < Producers) {
            float score[QKNt][4];
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                score[nt][0] = score[nt][1] = score[nt][2] = score[nt][3] = 0.0f;
            }
            unsigned af[2][4];
            unsigned bf[2][QKNt][2];
            ldmatrix_x4(af[0][0], af[0][1], af[0][2], af[0][3],
                        causal_prompt_swz_addr(q_lane_base, 0u, q_as, ab_r));
#pragma unroll
            for (int nt2 = 0; nt2 < QKNt; nt2 += 2) {
                ldmatrix_x4(bf[0][nt2][0], bf[0][nt2][1], bf[0][nt2 + 1][0], bf[0][nt2 + 1][1],
                            causal_prompt_swz_addr(k_lane_base + static_cast<unsigned>(nt2 * 4096),
                                                   0u, k_as, ab_r));
            }
#pragma unroll
            for (int k = 0; k < QKKs; ++k) {
                const int cur = k & 1;
                const int nxt = cur ^ 1;
                if (k + 1 < QKKs) {
                    const unsigned ck = static_cast<unsigned>((k + 1) << 5);
                    ldmatrix_x4(af[nxt][0], af[nxt][1], af[nxt][2], af[nxt][3],
                                causal_prompt_swz_addr(q_lane_base, ck, q_as, ab_r));
#pragma unroll
                    for (int nt2 = 0; nt2 < QKNt; nt2 += 2) {
                        ldmatrix_x4(bf[nxt][nt2][0], bf[nxt][nt2][1], bf[nxt][nt2 + 1][0],
                                    bf[nxt][nt2 + 1][1],
                                    causal_prompt_swz_addr(
                                        k_lane_base + static_cast<unsigned>(nt2 * 4096), ck, k_as,
                                        ab_r));
                    }
                }
#pragma unroll
                for (int nt = 0; nt < QKNt; ++nt) {
                    mma_bf16(score[nt][0], score[nt][1], score[nt][2], score[nt][3], af[cur][0],
                             af[cur][1], af[cur][2], af[cur][3], bf[cur][nt][0], bf[cur][nt][1]);
                }
            }

            const int row0             = prod_row0 + gid;
            const int row1             = row0 + 8;
            const int qabs0            = row0 < tile_rows ? base_pos + q0 + row0 : -1;
            const int qabs1            = row1 < tile_rows ? base_pos + q0 + row1 : -1;
            const bool full_score_tile = q0 + Br <= tokens && k0 + Bc - 1 <= base_pos + q0;
            float bm0 = -CUDART_INF_F, bm1 = -CUDART_INF_F;
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                if (!full_score_tile) {
                    const int key0 = k0 + nt * 8 + 2 * lid;
                    const int key1 = key0 + 1;
                    score[nt][0]   = key0 <= qabs0 ? score[nt][0] : -CUDART_INF_F;
                    score[nt][1]   = key1 <= qabs0 ? score[nt][1] : -CUDART_INF_F;
                    score[nt][2]   = key0 <= qabs1 ? score[nt][2] : -CUDART_INF_F;
                    score[nt][3]   = key1 <= qabs1 ? score[nt][3] : -CUDART_INF_F;
                }
                bm0 = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
                bm1 = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
            }
            bm0 = warp_max<4>(bm0, FullMask);
            bm1 = warp_max<4>(bm1, FullMask);

            const float nm0 = fmaxf(m0, bm0);
            const float nm1 = fmaxf(m1, bm1);
            // A row with no visible key yet keeps m at -inf; use 0 so exp2 stays finite.
            const float nm0_scaled = nm0 == -CUDART_INF_F ? 0.0f : nm0 * scale_l2;
            const float nm1_scaled = nm1 == -CUDART_INF_F ? 0.0f : nm1 * scale_l2;
            const float alpha0 =
                m0 == -CUDART_INF_F ? 0.0f : exp2_approx(__fmaf_rn(m0, scale_l2, -nm0_scaled));
            const float alpha1 =
                m1 == -CUDART_INF_F ? 0.0f : exp2_approx(__fmaf_rn(m1, scale_l2, -nm1_scaled));
            float bl0 = 0.0f, bl1 = 0.0f;
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const float p00 = exp2_approx(__fmaf_rn(score[nt][0], scale_l2, -nm0_scaled));
                const float p01 = exp2_approx(__fmaf_rn(score[nt][1], scale_l2, -nm0_scaled));
                const float p10 = exp2_approx(__fmaf_rn(score[nt][2], scale_l2, -nm1_scaled));
                const float p11 = exp2_approx(__fmaf_rn(score[nt][3], scale_l2, -nm1_scaled));
                bl0 += p00 + p01;
                bl1 += p10 + p11;
                const int col = nt * 8 + 2 * lid;
                store_vec(&p_s[row0 * Bc + causal_prompt_tcq3_p_swz(row0, col)],
                          __floats2half2_rn(p00, p01));
                store_vec(&p_s[row1 * Bc + causal_prompt_tcq3_p_swz(row1, col)],
                          __floats2half2_rn(p10, p11));
            }
            // Row sums stay lane-partial until the epilogue; alpha is uniform across a row.
            l0 = __fmaf_rn(l0, alpha0, bl0);
            l1 = __fmaf_rn(l1, alpha1, bl1);
            m0 = nm0;
            m1 = nm1;
            if (lid == 0) {
                alpha_s[row0] = alpha0;
                alpha_s[row1] = alpha1;
            }
        } else {
            for (int task = tid - Producers * 32; task < TileTasks; task += WorkerThreads) {
                causal_prompt_tcq3_decode8<false>(v_s, stage_ptr(kb), cb_v, task, k0,
                                                  max_query_abs);
            }
        }
        cp_wait<Stages - 2>();
        __syncthreads();

        // Stage kb is fully consumed: its V was decoded above and its K one iteration ago.
        issue_raw(kb + Stages);

        // The tile's P V product accumulates in FP16 at twice the FP32-accumulate MMA rate. It
        // sums 32 keys of P <= 1 against FP16 V, so its extra rounding sits below the BF16
        // operand rounding, and the running accumulator stays FP32.
        unsigned tile_pv[PVNtPerWarp][2] = {};
#pragma unroll
        for (int k = 0; k < PVKs; ++k) {
            unsigned pf[4];
            const int prow = pv_row0 + a_rowoff;
            ldmatrix_x4(pf[0], pf[1], pf[2], pf[3],
                        smem_addr(&p_s[prow * Bc +
                                       causal_prompt_tcq3_p_swz(prow, k * 16 + a_coloff)]));
#pragma unroll
            for (int n2 = 0; n2 < PVNtPerWarp; n2 += 2) {
                unsigned vf[4];
                ldmatrix_x4_t(vf[0], vf[1], vf[2], vf[3],
                              causal_prompt_swz_addr(v_lane_base + static_cast<unsigned>(k * 8192),
                                                     static_cast<unsigned>(n2 << 4), v_as, ab_r));
                mma_f16_f16acc(tile_pv[n2][0], tile_pv[n2][1], pf[0], pf[1], pf[2], pf[3], vf[0],
                               vf[1]);
                mma_f16_f16acc(tile_pv[n2 + 1][0], tile_pv[n2 + 1][1], pf[0], pf[1], pf[2], pf[3],
                               vf[2], vf[3]);
            }
        }
        const float alpha0 = alpha_s[pv_row0 + gid];
        const float alpha1 = alpha_s[pv_row0 + gid + 8];
#pragma unroll
        for (int n = 0; n < PVNtPerWarp; ++n) {
            const float2 top = __half22float2(*reinterpret_cast<const __half2*>(&tile_pv[n][0]));
            const float2 bot = __half22float2(*reinterpret_cast<const __half2*>(&tile_pv[n][1]));
            acc[n][0]        = fmaf(acc[n][0], alpha0, top.x);
            acc[n][1]        = fmaf(acc[n][1], alpha0, top.y);
            acc[n][2]        = fmaf(acc[n][2], alpha1, bot.x);
            acc[n][3]        = fmaf(acc[n][3], alpha1, bot.y);
        }
        // Decode the next K tile after the PV MMAs are issued, so the decode fills issue slots
        // while the tensor pipe works through them.
        if (kb + 1 < key_blocks) {
            for (int task = tid; task < TileTasks; task += Threads) {
                causal_prompt_tcq3_decode8<true>(k_s, stage_ptr(kb + 1), cb_k, task, k0 + Bc,
                                                 max_query_abs);
            }
        }
        __syncthreads();
    }

    if (warp < Producers) {
        l0 = warp_sum<4>(l0, FullMask);
        l1 = warp_sum<4>(l1, FullMask);
        if (lid == 0) {
            final_l_s[prod_row0 + gid]     = l0;
            final_l_s[prod_row0 + gid + 8] = l1;
        }
    }
    __syncthreads();

    // V was rotated before quantization, so the accumulator is still in the rotated domain.
    // The inverse needs whole rows, which the MMA layout spreads over four warps, so stage the
    // normalized rows in FP32 over the tiles the key loop no longer uses.
    {
        const int row0     = pv_row0 + gid;
        const int row1     = row0 + 8;
        const float fl0    = final_l_s[row0];
        const float fl1    = final_l_s[row1];
        const float inv_l0 = fl0 > 0.0f ? __frcp_rn(fl0) : 0.0f;
        const float inv_l1 = fl1 > 0.0f ? __frcp_rn(fl1) : 0.0f;
#pragma unroll
        for (int n = 0; n < PVNtPerWarp; ++n) {
            const int d0 = (d_slice * PVNtPerWarp + n) * 8 + 2 * lid;
            store_vec(&out_s[row0 * OutStride + d0],
                      make_float2(acc[n][0] * inv_l0, acc[n][1] * inv_l0));
            store_vec(&out_s[row1 * OutStride + d0],
                      make_float2(acc[n][2] * inv_l1, acc[n][3] * inv_l1));
        }
    }
    __syncthreads();
    for (int row = warp; row < tile_rows; row += kCausalPromptTcq3Warps) {
        float values[8];
#pragma unroll
        for (int r = 0; r < 8; ++r) { values[r] = out_s[row * OutStride + lane + 32 * r]; }
        kv_cache_tcq3_unrotate(values, lane);
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            out[causal_prompt_q_index<Geometry>(q_head, lane + 32 * r, q0 + row)] =
                __float2bfloat16(values[r]);
        }
    }
    causal_prompt_zero_output_rows<Geometry>(out, q_head, tokens, min(q0 + Br, width), tid,
                                             Threads);
}

} // namespace ninfer::ops
