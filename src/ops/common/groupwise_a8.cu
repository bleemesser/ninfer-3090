#include "ops/common/groupwise_a8.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/math.h"
#include "ops/common/memory.cuh"

#include <cuda_fp16.h>

namespace ninfer::ops::detail {
namespace {

constexpr int kGroup        = 64;
constexpr int kBlockCols    = 128;
constexpr int kBlockK       = 128;
constexpr int kGroupsPerK   = kBlockK / kGroup;
constexpr int kStages       = 3;
constexpr int kWarpsM       = 2;
constexpr int kWarpsN       = 4;
constexpr int kThreads      = kWarpsM * kWarpsN * 32;
constexpr int kWarpRows     = kGroupwiseA8BlockRows / kWarpsM;
constexpr int kWarpCols     = kBlockCols / kWarpsN;
constexpr int kMmaRows      = kWarpRows / 16;
constexpr int kMmaCols      = kWarpCols / 8;
constexpr int kQuantThreads = 256;

// int32 accumulators start at the bit pattern of M = 1.5 * 2^23. Every partial sum of one group
// stays far inside +-2^22, so the float with the resulting bits is exactly M plus the sum, which
// avoids the quarter-rate I2F conversion. Activation scales keep a 22-bit significand, so
// M * scale is exact and fma(M + sum, scale, -M * scale) is sum * scale rounded once.
constexpr int kMagicBits    = 0x4B400000;
constexpr float kMagicValue = 12582912.0f;

constexpr int kCodeBytes  = kGroupwiseA8BlockRows * kGroupsPerK * 32;
constexpr int kHighBytes  = kGroupwiseA8BlockRows * kGroupsPerK * 8;
constexpr int kScaleBytes = kGroupwiseA8BlockRows * kGroupsPerK * 2;
constexpr int kActBytes   = kBlockCols * kBlockK;
constexpr int kMetaBytes  = kBlockCols * kGroupsPerK * static_cast<int>(sizeof(float2));

template <bool Q5>
inline constexpr int kStageBytes =
    kCodeBytes + (Q5 ? kHighBytes : 0) + kScaleBytes + kActBytes + kMetaBytes;

enum class Epilogue { Store, AddResidual, SwiGlu };

// Staged weight codes are [row][2 groups x 32 bytes] and activations [column][2 groups x 64
// bytes]. Swapping the two group halves on alternate row pairs and alternate columns spreads the
// fragment loads of one quarter-warp over all 32 banks.
__device__ __forceinline__ int code_offset(int row, int group, int byte) {
    return row * 64 + ((group ^ ((row >> 1) & 1)) << 5) + byte;
}

__device__ __forceinline__ int act_offset(int col, int group, int byte) {
    return col * kBlockK + ((group ^ (col & 1)) << 6) + byte;
}

struct Jobs {
    GroupwiseJob job[4];
    int count;
};

__device__ __forceinline__ void mma_u8s8(int& c0, int& c1, int& c2, int& c3, unsigned a0,
                                         unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                         unsigned b1) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// d = a * b + c with a separate accumulator input, which seeds a group without register moves.
__device__ __forceinline__ void mma_u8s8_seed(int& d0, int& d1, int& d2, int& d3, unsigned a0,
                                              unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                              unsigned b1, int c0, int c1) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%10,%11};\n"
                 : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(c0), "r"(c1));
}

// Byte i of the result carries bit 4 set when bit 2i of h is clear. The Q5 unsigned code is
// low nibble | (!high) << 4, which is the signed code plus 16. Multiplying by 0x410410 places
// bits 0, 2, 4, 6 at 4, 12, 20, 28; the cross terms and their carries land elsewhere.
__device__ __forceinline__ unsigned q5_spread_high(unsigned h) {
    return ((~h & 0x55u) * 0x410410u) & 0x10101010u;
}

// Quantize one 64-value group for each eight lanes. Lane j of a group reads values 8j..8j+7
// and writes them as two words: even offsets first, then odd. That is the order in which
// the GEMM unpacks low and high weight nibbles from the same four code bytes.
__global__ __launch_bounds__(kQuantThreads) void groupwise_a8_quantize_kernel(
    const __nv_bfloat16* __restrict__ x, std::int8_t* __restrict__ codes,
    float2* __restrict__ meta, std::int32_t k, std::int32_t tokens) {
    const int groups = k / kGroup;
    const int item   = static_cast<int>((blockIdx.x * blockDim.x + threadIdx.x) >> 3);
    const int sub    = static_cast<int>(threadIdx.x) & 7;
    if (item >= tokens * groups) { return; }
    const int token        = item / groups;
    const int group        = item - token * groups;
    const std::int64_t off = static_cast<std::int64_t>(token) * k + group * kGroup + sub * 8;

    const int4 raw = load_vec<int4>(&x[off]);
    const auto* v  = reinterpret_cast<const __nv_bfloat16*>(&raw);
    float f[8];
    float absmax = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        f[i]   = __bfloat162float(v[i]);
        absmax = fmaxf(absmax, fabsf(f[i]));
    }
    const unsigned mask = 0xffu << (threadIdx.x & 24);
#pragma unroll
    for (int o = 1; o < 8; o <<= 1) { absmax = fmaxf(absmax, __shfl_xor_sync(mask, absmax, o)); }
    // Clearing the lowest significand bit keeps kMagicValue * scale exact in the GEMM.
    const float scale = __int_as_float(__float_as_int(absmax / 127.0f) & ~1);
    const float inv   = scale > 0.0f ? 1.0f / scale : 0.0f;
    int q[8];
    int sum = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        q[i] = max(-127, min(127, __float2int_rn(f[i] * inv)));
        sum += q[i];
    }
#pragma unroll
    for (int o = 1; o < 8; o <<= 1) { sum += __shfl_xor_sync(mask, sum, o); }
    const unsigned even = (q[0] & 0xff) | ((q[2] & 0xff) << 8) | ((q[4] & 0xff) << 16) |
                          (static_cast<unsigned>(q[6] & 0xff) << 24);
    const unsigned odd = (q[1] & 0xff) | ((q[3] & 0xff) << 8) | ((q[5] & 0xff) << 16) |
                         (static_cast<unsigned>(q[7] & 0xff) << 24);
    store_vec(&codes[off], make_uint2(even, odd));
    if (sub == 0) {
        meta[static_cast<std::int64_t>(token) * groups + group] =
            make_float2(scale, __int_as_float(sum));
    }
}

// Weight row of the gate/up parent that local CTA row lr reads. Each warp's first two MMA row
// tiles are gate rows and its last two are the matching up rows, so the epilogue pairs them
// in registers.
__device__ __forceinline__ int swiglu_parent_row(int tile, int lr, int intermediate) {
    const int wm = lr / kWarpRows;
    const int mt = (lr % kWarpRows) / 16;
    return tile * (kGroupwiseA8BlockRows / 2) + wm * (kWarpRows / 2) + (mt & 1) * 16 + (lr & 15) +
           (mt >> 1) * intermediate;
}

// out[rows of the jobs, T] = W x with INT8 activations. Grid x walks the row tiles of every
// job in order and grid y the 128-token tiles. BK is two quant groups.
template <bool Q5, Epilogue Kind>
__global__ __launch_bounds__(kThreads, 1) void groupwise_a8_gemm_kernel(
    const std::int8_t* __restrict__ act_codes, const float2* __restrict__ act_meta, Jobs jobs,
    std::int32_t k, std::int32_t tokens) {
    constexpr int Stage = kStageBytes<Q5>;
    extern __shared__ __align__(16) unsigned char smem[];

    int tile = static_cast<int>(blockIdx.x);
    int j    = 0;
    if constexpr (Kind != Epilogue::SwiGlu) {
        while (j + 1 < jobs.count && tile >= jobs.job[j].rows / kGroupwiseA8BlockRows) {
            tile -= jobs.job[j].rows / kGroupwiseA8BlockRows;
            ++j;
        }
    }
    const GroupwiseJob job = jobs.job[j];

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wm   = warp / kWarpsN;
    const int wn   = warp % kWarpsN;
    const int gid  = lane >> 2;
    const int lid  = lane & 3;
    const int m0   = tile * kGroupwiseA8BlockRows;
    const int t0   = static_cast<int>(blockIdx.y) * kBlockCols;
    const int gpr  = k / kGroup;
    const int kts  = k / kBlockK;
    // SwiGLU jobs hold the whole parent, so half its rows is the intermediate width.
    const int intermediate = job.rows / 2;

    auto weight_row = [&](int lr) {
        if constexpr (Kind == Epilogue::SwiGlu) {
            return swiglu_parent_row(tile, lr, intermediate);
        } else {
            return m0 + lr;
        }
    };

    // Every thread copies the same rows and columns at every K step, so the source offsets are
    // formed once; a K step only adds its group or byte offset.
    static_assert(kThreads == 2 * kGroupwiseA8BlockRows && kBlockCols * (kBlockK / 16) == 4 * kThreads);
    const int code_row = tid >> 2;
    const int code_col = (tid & 3) * 16;
    const std::uint32_t code_src[2] = {
        static_cast<std::uint32_t>(weight_row(code_row) * gpr * 32 + code_col),
        static_cast<std::uint32_t>(weight_row(code_row + kGroupwiseA8BlockRows / 2) * gpr * 32 +
                                   code_col)};
    const bool row_copier       = tid < kGroupwiseA8BlockRows;
    const std::uint32_t row_src = static_cast<std::uint32_t>(weight_row(row_copier ? tid : 0) * gpr);
    const int act_chunk         = (tid & 7) * 16;
    std::uint32_t act_src[4];
    int act_live = 0;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const int col = t0 + (tid >> 3) + q * (kThreads / 8);
        act_live |= (col < tokens ? 1 : 0) << q;
        act_src[q] = static_cast<std::uint32_t>((col < tokens ? col : 0) * k + act_chunk);
    }
    const int meta_col           = t0 + (row_copier ? tid : 0);
    const bool meta_live         = row_copier && meta_col < tokens;
    const std::uint32_t meta_src = static_cast<std::uint32_t>((meta_live ? meta_col : 0) * gpr);

    auto load_stage = [&](int stage, int kt) {
        unsigned char* base = smem + stage * Stage;
        const int g0        = kt * kGroupsPerK;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            cp_async<16, Cache::cg>(
                base + code_offset(code_row + h * kGroupwiseA8BlockRows / 2, code_col >> 5,
                                   code_col & 31),
                job.codes + code_src[h] + g0 * 32);
        }
        unsigned char* high_dst  = base + kCodeBytes;
        unsigned char* scale_dst = high_dst + (Q5 ? kHighBytes : 0);
        unsigned char* act_dst   = scale_dst + kScaleBytes;
        unsigned char* meta_dst  = act_dst + kActBytes;
        if (row_copier) {
            if constexpr (Q5) {
                cp_async<16, Cache::cg>(high_dst + tid * 16, job.high + (row_src + g0) * 8);
            }
            cp_async<4>(scale_dst + tid * 4, job.scales + (row_src + g0) * 2);
            cp_async_zfill<16, Cache::cg>(meta_dst + tid * 16, act_meta + meta_src + g0,
                                          meta_live ? 16 : 0);
        }
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            cp_async_zfill<16, Cache::cg>(
                act_dst + act_offset((tid >> 3) + q * (kThreads / 8), act_chunk >> 6, act_chunk & 63),
                act_codes + act_src[q] + kt * kBlockK, ((act_live >> q) & 1) ? 16 : 0);
        }
    };

    float acc[kMmaRows][kMmaCols][4];
#pragma unroll
    for (int mt = 0; mt < kMmaRows; ++mt) {
#pragma unroll
        for (int nt = 0; nt < kMmaCols; ++nt) {
#pragma unroll
            for (int i = 0; i < 4; ++i) { acc[mt][nt][i] = 0.0f; }
        }
    }

#pragma unroll
    for (int s = 0; s < kStages - 1; ++s) {
        if (s < kts) { load_stage(s, s); }
        cp_commit();
    }

    // Unsigned weight codes carry a +8 (Q4) or +16 (Q5) offset; the group code sum removes it.
    constexpr int kOffset = Q5 ? 16 : 8;

    for (int kt = 0; kt < kts; ++kt) {
        cp_wait<kStages - 2>();
        __syncthreads();
        {
            const int next = kt + kStages - 1;
            if (next < kts) { load_stage(next % kStages, next); }
            cp_commit();
        }
        const unsigned char* base   = smem + (kt % kStages) * Stage;
        const unsigned char* codes  = base;
        const unsigned char* high   = base + kCodeBytes;
        const __half* scales        = reinterpret_cast<const __half*>(high + (Q5 ? kHighBytes : 0));
        const unsigned char* acts   = reinterpret_cast<const unsigned char*>(scales) + kScaleBytes;
        const float2* meta          = reinterpret_cast<const float2*>(acts + kActBytes);

#pragma unroll
        for (int gg = 0; gg < kGroupsPerK; ++gg) {
            uint4 b[kMmaCols];
#pragma unroll
            for (int nt = 0; nt < kMmaCols; ++nt) {
                const int c = wn * kWarpCols + nt * 8 + gid;
                b[nt]       = load_vec<uint4>(acts + act_offset(c, gg, lid * 16));
            }
            int init[kMmaCols][2];
            float act_scale[kMmaCols][2];
            float act_bias[kMmaCols][2];
#pragma unroll
            for (int nt = 0; nt < kMmaCols; ++nt) {
                const int c = wn * kWarpCols + nt * 8 + 2 * lid;
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const float2 m   = meta[(c + e) * kGroupsPerK + gg];
                    init[nt][e]      = kMagicBits - kOffset * __float_as_int(m.y);
                    act_scale[nt][e] = m.x;
                    act_bias[nt][e]  = -kMagicValue * m.x;
                }
            }
            // Slice mt's MMAs are issued before slice mt-1 is converted, so the tensor pipe works
            // through one slice while the FMA pipe scales the previous one.
            int ci[2][kMmaCols][4];
            auto issue_slice = [&](int mt, int (&out)[kMmaCols][4]) {
                const int r0   = wm * kWarpRows + mt * 16 + gid;
                const uint2 w0 = load_vec<uint2>(codes + code_offset(r0, gg, lid * 8));
                const uint2 w1 = load_vec<uint2>(codes + code_offset(r0 + 8, gg, lid * 8));
                unsigned h0 = 0, h1 = 0;
                if constexpr (Q5) {
                    h0 = load_vec<unsigned short>(high + r0 * 16 + gg * 8 + lid * 2);
                    h1 = load_vec<unsigned short>(high + (r0 + 8) * 16 + gg * 8 + lid * 2);
                }
#pragma unroll
                for (int s = 0; s < 2; ++s) {
                    const unsigned x0 = s ? w0.y : w0.x;
                    const unsigned x1 = s ? w1.y : w1.x;
                    unsigned a0, a1, a2, a3;
                    if constexpr (Q5) {
                        const unsigned hb0 = (h0 >> (8 * s)) & 0xffu;
                        const unsigned hb1 = (h1 >> (8 * s)) & 0xffu;
                        a0 = (x0 & 0x0f0f0f0fu) | q5_spread_high(hb0);
                        a1 = (x1 & 0x0f0f0f0fu) | q5_spread_high(hb1);
                        a2 = ((x0 >> 4) & 0x0f0f0f0fu) | q5_spread_high(hb0 >> 1);
                        a3 = ((x1 >> 4) & 0x0f0f0f0fu) | q5_spread_high(hb1 >> 1);
                    } else {
                        a0 = (x0 ^ 0x88888888u) & 0x0f0f0f0fu;
                        a1 = (x1 ^ 0x88888888u) & 0x0f0f0f0fu;
                        a2 = ((x0 ^ 0x88888888u) >> 4) & 0x0f0f0f0fu;
                        a3 = ((x1 ^ 0x88888888u) >> 4) & 0x0f0f0f0fu;
                    }
#pragma unroll
                    for (int nt = 0; nt < kMmaCols; ++nt) {
                        if (s == 0) {
                            mma_u8s8_seed(out[nt][0], out[nt][1], out[nt][2], out[nt][3], a0, a1,
                                          a2, a3, b[nt].x, b[nt].y, init[nt][0], init[nt][1]);
                        } else {
                            mma_u8s8(out[nt][0], out[nt][1], out[nt][2], out[nt][3], a0, a1, a2,
                                     a3, b[nt].z, b[nt].w);
                        }
                    }
                }
            };
            auto convert_slice = [&](int mt, const int (&in)[kMmaCols][4]) {
                const int r0    = wm * kWarpRows + mt * 16 + gid;
                const float ws0 = __half2float(scales[r0 * kGroupsPerK + gg]);
                const float ws1 = __half2float(scales[(r0 + 8) * kGroupsPerK + gg]);
#pragma unroll
                for (int nt = 0; nt < kMmaCols; ++nt) {
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const float partial = fmaf(__int_as_float(in[nt][i]), act_scale[nt][i & 1],
                                                   act_bias[nt][i & 1]);
                        acc[mt][nt][i] = fmaf(partial, i < 2 ? ws0 : ws1, acc[mt][nt][i]);
                    }
                }
            };
#pragma unroll
            for (int mt = 0; mt < kMmaRows; ++mt) {
                issue_slice(mt, ci[mt & 1]);
                if (mt > 0) { convert_slice(mt - 1, ci[(mt - 1) & 1]); }
            }
            convert_slice(kMmaRows - 1, ci[(kMmaRows - 1) & 1]);
        }
    }

    if constexpr (Kind == Epilogue::SwiGlu) {
        constexpr int Half = kMmaRows / 2;
#pragma unroll
        for (int mt = 0; mt < Half; ++mt) {
            const int row0 = tile * (kGroupwiseA8BlockRows / 2) + wm * (kWarpRows / 2) + mt * 16 + gid;
#pragma unroll
            for (int nt = 0; nt < kMmaCols; ++nt) {
                const int col0 = t0 + wn * kWarpCols + nt * 8 + 2 * lid;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int row = row0 + (i >> 1) * 8;
                    const int col = col0 + (i & 1);
                    if (col < tokens) {
                        job.out[static_cast<std::int64_t>(col) * job.out_ld + row] =
                            __float2bfloat16_rn(silu(acc[mt][nt][i]) * acc[mt + Half][nt][i]);
                    }
                }
            }
        }
    } else {
#pragma unroll
        for (int mt = 0; mt < kMmaRows; ++mt) {
            const int row0 = job.out_row_offset + m0 + wm * kWarpRows + mt * 16 + gid;
#pragma unroll
            for (int nt = 0; nt < kMmaCols; ++nt) {
                const int col0 = t0 + wn * kWarpCols + nt * 8 + 2 * lid;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int row = row0 + (i >> 1) * 8;
                    const int col = col0 + (i & 1);
                    if (col < tokens) {
                        __nv_bfloat16* dst =
                            &job.out[static_cast<std::int64_t>(col) * job.out_ld + row];
                        if constexpr (Kind == Epilogue::AddResidual) {
                            *dst = __float2bfloat16_rn(__bfloat162float(*dst) + acc[mt][nt][i]);
                        } else {
                            *dst = __float2bfloat16_rn(acc[mt][nt][i]);
                        }
                    }
                }
            }
        }
    }
}

template <bool Q5, Epilogue Kind>
void launch_gemm(const GroupwiseA8Activation& act, const Jobs& jobs, int row_tiles,
                 cudaStream_t stream) {
    constexpr int Smem = kStageBytes<Q5> * kStages;
    static const cudaError_t attr = cudaFuncSetAttribute(
        groupwise_a8_gemm_kernel<Q5, Kind>, cudaFuncAttributeMaxDynamicSharedMemorySize, Smem);
    CUDA_CHECK(attr);
    const dim3 grid(static_cast<unsigned>(row_tiles),
                    static_cast<unsigned>(div_up(act.tokens, kBlockCols)));
    groupwise_a8_gemm_kernel<Q5, Kind>
        <<<grid, kThreads, Smem, stream>>>(act.codes, act.meta, jobs, act.k, act.tokens);
    CUDA_CHECK(cudaGetLastError());
}

void check_weight(const Weight& weight, const GroupwiseA8Activation& act) {
    if (weight.qtype != QType::Q4G64_F16S && weight.qtype != QType::Q5G64_F16S) {
        throw std::invalid_argument("groupwise A8: weight must be Q4G64_F16S or Q5G64_F16S");
    }
    if (weight.padded_shape[1] != act.k || weight.k != act.k) {
        throw std::invalid_argument("groupwise A8: weight K must equal the activation K");
    }
}

} // namespace

void groupwise_a8_quantize(const Tensor& x, const GroupwiseA8Activation& act,
                           cudaStream_t stream) {
    if (x.dtype != DType::BF16 || x.ne[0] != act.k || x.ne[1] != act.tokens) {
        throw std::invalid_argument("groupwise A8: activation does not match its workspace");
    }
    const std::int64_t lanes =
        static_cast<std::int64_t>(act.tokens) * (act.k / kGroup) * 8;
    groupwise_a8_quantize_kernel<<<static_cast<unsigned>(div_up(lanes, static_cast<std::int64_t>(
                                                                           kQuantThreads))),
                                   kQuantThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), act.codes, act.meta, act.k, act.tokens);
    CUDA_CHECK(cudaGetLastError());
}

void groupwise_a8_store(const GroupwiseA8Activation& act, bool q5,
                        std::span<const GroupwiseJob> jobs, cudaStream_t stream) {
    if (jobs.empty() || jobs.size() > 4) {
        throw std::invalid_argument("groupwise A8: store takes one to four jobs");
    }
    Jobs packed{};
    packed.count  = static_cast<int>(jobs.size());
    int row_tiles = 0;
    for (std::size_t i = 0; i < jobs.size(); ++i) {
        if ((jobs[i].rows % kGroupwiseA8BlockRows) != 0) {
            throw std::invalid_argument("groupwise A8: job rows must be a multiple of 128");
        }
        packed.job[i] = jobs[i];
        row_tiles += jobs[i].rows / kGroupwiseA8BlockRows;
    }
    if (q5) {
        launch_gemm<true, Epilogue::Store>(act, packed, row_tiles, stream);
    } else {
        launch_gemm<false, Epilogue::Store>(act, packed, row_tiles, stream);
    }
}

void groupwise_a8_add_residual(const GroupwiseA8Activation& act, const Weight& weight,
                               Tensor& residual, cudaStream_t stream) {
    check_weight(weight, act);
    if (residual.ne[0] != weight.n || residual.ne[1] != act.tokens) {
        throw std::invalid_argument("groupwise A8: residual shape does not match");
    }
    Jobs packed{};
    packed.count  = 1;
    packed.job[0] = groupwise_job(weight, 0, weight.n, residual, 0);
    const int row_tiles = weight.n / kGroupwiseA8BlockRows;
    if (weight.qtype == QType::Q5G64_F16S) {
        launch_gemm<true, Epilogue::AddResidual>(act, packed, row_tiles, stream);
    } else {
        launch_gemm<false, Epilogue::AddResidual>(act, packed, row_tiles, stream);
    }
}

void groupwise_a8_swiglu(const GroupwiseA8Activation& act, const Weight& gate_up, Tensor& out,
                         cudaStream_t stream) {
    check_weight(gate_up, act);
    if (gate_up.qtype != QType::Q4G64_F16S || (gate_up.n % (2 * kGroupwiseA8BlockRows)) != 0 ||
        out.ne[0] != gate_up.n / 2 || out.ne[1] != act.tokens) {
        throw std::invalid_argument("groupwise A8: SwiGLU shape does not match");
    }
    Jobs packed{};
    packed.count  = 1;
    packed.job[0] = groupwise_job(gate_up, 0, gate_up.n, out, 0);
    launch_gemm<false, Epilogue::SwiGlu>(act, packed, gate_up.n / kGroupwiseA8BlockRows, stream);
}

} // namespace ninfer::ops::detail
