#pragma once

// Small-T A16 route for the Q4G64 and Q5G64 RowSplit formats, for the 2..8 token widths of
// speculative verification.
//
// At these widths the projection is bound by weight bandwidth. Each lane loads its weight bytes
// straight from global memory into MMA fragment registers, so no staging stage sits between the
// load and the m16n8k16 BF16 product. The codes stay exact integers in BF16 and each group's
// FP16 scale is applied to the FP32 group sum. Semantic Ops own the route choice and the jobs.

#include "core/tensor.h"
#include "ops/common/groupwise_job.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <span>

namespace ninfer::ops::detail {

inline constexpr std::int32_t kGroupwiseSmallTMinTokens = 2;
inline constexpr std::int32_t kGroupwiseSmallTMaxTokens = 8;

// Stores BF16 projections for up to four jobs that share one format in a single launch.
// x is contiguous BF16 [K,T] with T in [2,8].
void groupwise_small_t_store(const Tensor& x, bool q5, std::span<const GroupwiseJob> jobs,
                             cudaStream_t stream);

// residual[:, t] += W x[:, t], rounded to BF16 once.
void groupwise_small_t_add_residual(const Tensor& x, const Weight& weight, Tensor& residual,
                                    cudaStream_t stream);

// out = SiLU(gate) * up for a Q4 gate/up parent whose first half of rows is the gate.
void groupwise_small_t_swiglu(const Tensor& x, const Weight& gate_up, Tensor& out,
                              cudaStream_t stream);

} // namespace ninfer::ops::detail
