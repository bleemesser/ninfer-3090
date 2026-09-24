#include "ops/common/groupwise_small_t.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kGroup   = 64;
constexpr int kRows    = 16;
// Each warp streams its K slice in steps of four groups, which is one 128-byte run of codes for
// every row. SwiGLU stages twice the rows, so it runs two steps deep instead of three to stay
// inside static shared memory.
constexpr int kStepGroups = 4;

enum class Epilogue { Store, AddResidual, SwiGlu };

struct Jobs {
    GroupwiseJob job[4];
    int count;
};

// Four unsigned codes in the low nibbles of lo and four in hi (byte i of each is one value),
// as BF16 pairs (lo_i, hi_i) of 128 + code. The exponent byte 0x43 makes 128 + u exact for
// every u below 128.
__device__ __forceinline__ void biased_pairs(unsigned lo, unsigned hi, unsigned (&out)[4]) {
    constexpr unsigned kMagic = 0x43004300u;
    const unsigned even       = __byte_perm(lo, hi, 0x5140);
    const unsigned odd        = __byte_perm(lo, hi, 0x7362);
    out[0]                    = __byte_perm(even, kMagic, 0x7150);
    out[1]                    = __byte_perm(even, kMagic, 0x7372);
    out[2]                    = __byte_perm(odd, kMagic, 0x7150);
    out[3]                    = __byte_perm(odd, kMagic, 0x7372);
}

// Byte i carries bit 4 set when bit 2i of h is clear; see q5_spread_high in groupwise_a8.cu.
__device__ __forceinline__ unsigned q5_spread_high(unsigned h) {
    return ((~h & 0x55u) * 0x410410u) & 0x10101010u;
}

// A fragments for the four k16 steps of one row's group, from the lane's eight code bytes.
// Lane lid owns group values 16*lid..16*lid+15, and step s takes values 4s..4s+3 of them, so
// a[2s] is the (even, odd) pair of byte 2s and a[2s+1] the pair of byte 2s+1.
template <bool Q5>
__device__ __forceinline__ void decode_row(uint2 w, unsigned high, unsigned (&a)[8]) {
    // Unsigned Q4 codes carry +8 and Q5 codes +16, so the bias is 136 or 144 in both halves.
    constexpr unsigned kBias = Q5 ? 0x43104310u : 0x43084308u;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const unsigned word = half ? w.y : w.x;
        unsigned lo, hi;
        if constexpr (Q5) {
            const unsigned hb = (high >> (8 * half)) & 0xffu;
            lo                = (word & 0x0f0f0f0fu) | q5_spread_high(hb);
            hi                = ((word >> 4) & 0x0f0f0f0fu) | q5_spread_high(hb >> 1);
        } else {
            lo = (word ^ 0x88888888u) & 0x0f0f0f0fu;
            hi = ((word ^ 0x88888888u) >> 4) & 0x0f0f0f0fu;
        }
        unsigned pairs[4];
        biased_pairs(lo, hi, pairs);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const __nv_bfloat162 value = __hsub2(*reinterpret_cast<const __nv_bfloat162*>(&pairs[i]),
                                                 *reinterpret_cast<const __nv_bfloat162*>(&kBias));
            a[4 * half + i] = *reinterpret_cast<const unsigned*>(&value);
        }
    }
}

// Byte offset of 32-byte group chunk c of staged row r. Rotating the chunks by the row keeps
// the fragment loads of a half-warp, four rows at one group, on distinct banks.
__device__ __forceinline__ int staged_code_offset(int row, int chunk) {
    return row * (kStepGroups * 32) + ((chunk ^ (row & 3)) << 5);
}

// out rows of the jobs = W x for T in [2,8]. Each CTA owns 16 weight rows (32 for SwiGLU: the
// gate rows and their up rows) and its warps split K. Every warp streams its own rows and K
// slice through private shared buffers, so warps never wait on each other until the reduction.
// Lane (gid, lid) computes rows gid and gid+8 against token column gid.
// Warps per CTA and stage depth. SwiGLU stages twice the rows for each warp.
template <Epilogue Kind>
struct SmallTShape {
    static constexpr int kTiles   = Kind == Epilogue::SwiGlu ? 2 : 1;
    static constexpr int kWarps   = Kind == Epilogue::SwiGlu ? 4 : 8;
    static constexpr int kStages  = 2;
    static constexpr int kThreads = kWarps * 32;
};

template <bool Q5, Epilogue Kind>
__global__ __launch_bounds__(SmallTShape<Kind>::kThreads) void groupwise_small_t_kernel(
    const __nv_bfloat16* __restrict__ x, Jobs jobs, std::int32_t k, std::int32_t tokens) {
    constexpr int Tiles     = SmallTShape<Kind>::kTiles;
    constexpr int kWarps    = SmallTShape<Kind>::kWarps;
    constexpr int kStages   = SmallTShape<Kind>::kStages;
    constexpr int StageRows = Tiles * kRows;
    constexpr int CodeBytes = StageRows * kStepGroups * 32;
    constexpr int HighBytes = Q5 ? StageRows * kStepGroups * 8 : 0;
    constexpr int ScaleBytes = StageRows * kStepGroups * 2;
    constexpr int StageBytes = CodeBytes + HighBytes + ScaleBytes;
    __shared__ __align__(16) unsigned char staged[kWarps][kStages][StageBytes];
    __shared__ float4 partial[kWarps - 1][Tiles][32];

    int tile = static_cast<int>(blockIdx.x);
    int j    = 0;
    if constexpr (Kind != Epilogue::SwiGlu) {
        while (j + 1 < jobs.count && tile >= jobs.job[j].rows / kRows) {
            tile -= jobs.job[j].rows / kRows;
            ++j;
        }
    }
    const GroupwiseJob job = jobs.job[j];

    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int gid  = lane >> 2;
    const int lid  = lane & 3;
    const int gpr  = k / kGroup;

    // Staged row sr is weight row stage_row(sr): SwiGLU stages its 16 gate rows, then the 16
    // matching up rows.
    auto stage_row = [&](int sr) {
        if constexpr (Kind == Epilogue::SwiGlu) {
            return (sr < kRows ? 0 : job.rows / 2) + tile * kRows + (sr & (kRows - 1));
        } else {
            return tile * kRows + sr;
        }
    };

    // Warps take K steps round-robin, so any step count spreads within one step per warp.
    const int total_steps = gpr / kStepGroups;
    const int steps       = (total_steps - warp + kWarps - 1) / kWarps;
    auto step_group       = [&](int step) { return (warp + step * kWarps) * kStepGroups; };

    auto issue_step = [&](int step) {
        unsigned char* base = staged[warp][step % kStages];
        const int g0        = step_group(step);
        // Codes: eight 16-byte vectors for each row.
#pragma unroll
        for (int item = lane; item < StageRows * 8; item += 32) {
            const int sr = item >> 3;
            const int v  = item & 7;
            cp_async<16, Cache::cg>(
                base + staged_code_offset(sr, v >> 1) + (v & 1) * 16,
                job.codes + (static_cast<std::int64_t>(stage_row(sr)) * gpr + g0) * 32 + v * 16);
        }
        if constexpr (Q5) {
#pragma unroll
            for (int item = lane; item < StageRows * 2; item += 32) {
                const int sr = item >> 1;
                const int v  = item & 1;
                cp_async<16, Cache::cg>(
                    base + CodeBytes + sr * 32 + v * 16,
                    job.high + (static_cast<std::int64_t>(stage_row(sr)) * gpr + g0) * 8 + v * 16);
            }
        }
#pragma unroll
        for (int sr = lane; sr < StageRows; sr += 32) {
            cp_async<8>(base + CodeBytes + HighBytes + sr * 8,
                        job.scales + (static_cast<std::int64_t>(stage_row(sr)) * gpr + g0) * 2);
        }
    };

    const bool live_col        = gid < tokens;
    const __nv_bfloat16* x_col = x + static_cast<std::int64_t>(live_col ? gid : 0) * k + 16 * lid;

#pragma unroll
    for (int s = 0; s < kStages - 1; ++s) {
        if (s < steps) { issue_step(s); }
        cp_commit();
    }

    float acc[Tiles][4] = {};
    for (int step = 0; step < steps; ++step) {
        // Activations come from L1; issue them before waiting on the staged weights.
        uint4 xv[kStepGroups][2];
#pragma unroll
        for (int gs = 0; gs < kStepGroups; ++gs) {
            const int g = step_group(step) + gs;
            xv[gs][0]   = live_col ? load_ldg<uint4>(x_col + g * kGroup) : make_uint4(0, 0, 0, 0);
            xv[gs][1]   = live_col ? load_ldg<uint4>(x_col + g * kGroup + 8) : make_uint4(0, 0, 0, 0);
        }
        cp_wait<kStages - 2>();
        __syncwarp();
        if (step + kStages - 1 < steps) { issue_step(step + kStages - 1); }
        cp_commit();

        const unsigned char* base = staged[warp][step % kStages];
        const __half* scales      = reinterpret_cast<const __half*>(base + CodeBytes + HighBytes);
#pragma unroll
        for (int gs = 0; gs < kStepGroups; ++gs) {
            const unsigned b[8] = {xv[gs][0].x, xv[gs][0].y, xv[gs][0].z, xv[gs][0].w,
                                   xv[gs][1].x, xv[gs][1].y, xv[gs][1].z, xv[gs][1].w};
#pragma unroll
            for (int t = 0; t < Tiles; ++t) {
                const int r_lo = t * kRows + gid;
                const int r_hi = r_lo + 8;
                const uint2 w_lo = load_vec<uint2>(base + staged_code_offset(r_lo, gs) + lid * 8);
                const uint2 w_hi = load_vec<uint2>(base + staged_code_offset(r_hi, gs) + lid * 8);
                unsigned h_lo = 0, h_hi = 0;
                if constexpr (Q5) {
                    h_lo = load_vec<unsigned short>(base + CodeBytes + r_lo * 32 + gs * 8 + lid * 2);
                    h_hi = load_vec<unsigned short>(base + CodeBytes + r_hi * 32 + gs * 8 + lid * 2);
                }
                unsigned a_lo[8], a_hi[8];
                decode_row<Q5>(w_lo, h_lo, a_lo);
                decode_row<Q5>(w_hi, h_hi, a_hi);
                float sum[4] = {};
#pragma unroll
                for (int s = 0; s < 4; ++s) {
                    mma_bf16(sum[0], sum[1], sum[2], sum[3], a_lo[2 * s], a_hi[2 * s],
                             a_lo[2 * s + 1], a_hi[2 * s + 1], b[2 * s], b[2 * s + 1]);
                }
                const float s_lo = __half2float(scales[r_lo * kStepGroups + gs]);
                const float s_hi = __half2float(scales[r_hi * kStepGroups + gs]);
                acc[t][0] = fmaf(sum[0], s_lo, acc[t][0]);
                acc[t][1] = fmaf(sum[1], s_lo, acc[t][1]);
                acc[t][2] = fmaf(sum[2], s_hi, acc[t][2]);
                acc[t][3] = fmaf(sum[3], s_hi, acc[t][3]);
            }
        }
    }

    if (warp != 0) {
#pragma unroll
        for (int t = 0; t < Tiles; ++t) {
            partial[warp - 1][t][lane] = make_float4(acc[t][0], acc[t][1], acc[t][2], acc[t][3]);
        }
    }
    __syncthreads();
    if (warp != 0) { return; }
#pragma unroll
    for (int w = 0; w < kWarps - 1; ++w) {
#pragma unroll
        for (int t = 0; t < Tiles; ++t) {
            const float4 p = partial[w][t][lane];
            acc[t][0] += p.x;
            acc[t][1] += p.y;
            acc[t][2] += p.z;
            acc[t][3] += p.w;
        }
    }

    // Accumulator element i is row gid + 8 * (i >> 1), token column 2 * lid + (i & 1).
    const int row_base = tile * kRows;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int col = 2 * lid + (i & 1);
        if (col >= tokens) { continue; }
        const int row = row_base + gid + 8 * (i >> 1);
        __nv_bfloat16* dst =
            &job.out[static_cast<std::int64_t>(col) * job.out_ld + job.out_row_offset + row];
        if constexpr (Kind == Epilogue::SwiGlu) {
            *dst = __float2bfloat16_rn(silu(acc[0][i]) * acc[Tiles - 1][i]);
        } else if constexpr (Kind == Epilogue::AddResidual) {
            *dst = __float2bfloat16_rn(__bfloat162float(*dst) + acc[0][i]);
        } else {
            *dst = __float2bfloat16_rn(acc[0][i]);
        }
    }
}

template <bool Q5, Epilogue Kind>
void launch(const Tensor& x, const Jobs& jobs, int row_tiles, cudaStream_t stream) {
    groupwise_small_t_kernel<Q5, Kind>
        <<<static_cast<unsigned>(row_tiles), SmallTShape<Kind>::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), jobs, x.ne[0], x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

void check_input(const Tensor& x) {
    if (x.dtype != DType::BF16 || x.ne[1] < kGroupwiseSmallTMinTokens ||
        x.ne[1] > kGroupwiseSmallTMaxTokens || (x.ne[0] % (kGroup * kStepGroups)) != 0) {
        throw std::invalid_argument("groupwise small-T: x must be BF16 [K,T] with T in [2,8]");
    }
}

void check_weight(const Weight& weight, const Tensor& x) {
    if ((weight.qtype != QType::Q4G64_F16S && weight.qtype != QType::Q5G64_F16S) ||
        weight.padded_shape[1] != x.ne[0] || weight.k != x.ne[0]) {
        throw std::invalid_argument("groupwise small-T: weight must be Q4/Q5 with K equal to x");
    }
}

} // namespace

void groupwise_small_t_store(const Tensor& x, bool q5, std::span<const GroupwiseJob> jobs,
                             cudaStream_t stream) {
    check_input(x);
    if (jobs.empty() || jobs.size() > 4) {
        throw std::invalid_argument("groupwise small-T: store takes one to four jobs");
    }
    Jobs packed{};
    packed.count  = static_cast<int>(jobs.size());
    int row_tiles = 0;
    for (std::size_t i = 0; i < jobs.size(); ++i) {
        packed.job[i] = jobs[i];
        row_tiles += jobs[i].rows / kRows;
    }
    if (q5) {
        launch<true, Epilogue::Store>(x, packed, row_tiles, stream);
    } else {
        launch<false, Epilogue::Store>(x, packed, row_tiles, stream);
    }
}

void groupwise_small_t_add_residual(const Tensor& x, const Weight& weight, Tensor& residual,
                                    cudaStream_t stream) {
    check_input(x);
    check_weight(weight, x);
    if (residual.ne[0] != weight.n || residual.ne[1] != x.ne[1]) {
        throw std::invalid_argument("groupwise small-T: residual shape does not match");
    }
    Jobs packed{};
    packed.count  = 1;
    packed.job[0] = groupwise_job(weight, 0, weight.n, residual, 0);
    if (weight.qtype == QType::Q5G64_F16S) {
        launch<true, Epilogue::AddResidual>(x, packed, weight.n / kRows, stream);
    } else {
        launch<false, Epilogue::AddResidual>(x, packed, weight.n / kRows, stream);
    }
}

void groupwise_small_t_swiglu(const Tensor& x, const Weight& gate_up, Tensor& out,
                              cudaStream_t stream) {
    check_input(x);
    check_weight(gate_up, x);
    if (gate_up.qtype != QType::Q4G64_F16S || (gate_up.n % (2 * kRows)) != 0 ||
        out.ne[0] != gate_up.n / 2 || out.ne[1] != x.ne[1]) {
        throw std::invalid_argument("groupwise small-T: SwiGLU shape does not match");
    }
    Jobs packed{};
    packed.count  = 1;
    packed.job[0] = groupwise_job(gate_up, 0, gate_up.n, out, 0);
    launch<false, Epilogue::SwiGlu>(x, packed, gate_up.n / (2 * kRows), stream);
}

} // namespace ninfer::ops::detail
