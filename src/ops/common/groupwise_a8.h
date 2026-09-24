#pragma once

// A8 compute route for the Q4G64 and Q5G64 RowSplit formats.
//
// Activations are quantized to signed INT8 with one FP32 scale for each 64-value group, which
// is the weight quant group, and the product runs on m16n8k32 INT8 Tensor Cores. On sm_86 that
// instruction has four times the throughput of BF16 MMA with FP32 accumulation. Weight codes
// stay exact integers and scales stay FP32, so the only added rounding is the activation code.
//
// Semantic Ops own the route choice, the workspace scope, and the job set. This header owns
// the activation layout and the launch mechanism.

#include "core/layout.h"
#include "ops/common/groupwise_job.h"
#include "core/tensor.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>

namespace ninfer::ops::detail {

// Smallest T for which AllowA8 selects this route over the A16 Tensor Core routes.
inline constexpr std::int32_t kGroupwiseA8MinTokens = 17;

// Weight rows owned by one CTA. Every job and every SwiGLU half must be a multiple of it.
inline constexpr std::int32_t kGroupwiseA8BlockRows = 128;

// Quantized activations for one call. codes is [T][K] with each 64-value group permuted to
// the order the kernel reads weight nibbles in. meta is [T][K/64] of {scale, code sum}, the
// sum stored as int bits; the kernel needs it to remove the unsigned weight-code offset.
struct GroupwiseA8Activation {
    std::int8_t* codes  = nullptr;
    float2* meta        = nullptr;
    std::int32_t k      = 0;
    std::int32_t tokens = 0;
};

inline std::size_t groupwise_a8_checked_bytes(std::int32_t tokens, std::size_t bytes_per_token) {
    if (tokens <= 0) { throw std::invalid_argument("groupwise A8 workspace: T must be positive"); }
    const auto count = static_cast<std::size_t>(tokens);
    if (count > std::numeric_limits<std::size_t>::max() / bytes_per_token) {
        throw std::overflow_error("groupwise A8 workspace size overflow");
    }
    return count * bytes_per_token;
}

template <class Arena>
GroupwiseA8Activation allocate_groupwise_a8_activation(Arena& arena, std::int32_t tokens,
                                                       std::int32_t k) {
    if (k <= 0 || (k % 128) != 0) {
        throw std::invalid_argument("groupwise A8 workspace: K must be a positive multiple of 128");
    }
    const DeviceSpan codes =
        arena.alloc_bytes(groupwise_a8_checked_bytes(tokens, static_cast<std::size_t>(k)), 256);
    const DeviceSpan meta = arena.alloc_bytes(
        groupwise_a8_checked_bytes(tokens, static_cast<std::size_t>(k / 64) * sizeof(float2)),
        256);
    return {static_cast<std::int8_t*>(codes.data), static_cast<float2*>(meta.data), k, tokens};
}

inline std::size_t groupwise_a8_activation_capacity_bytes(std::int32_t tokens, std::int32_t k) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_groupwise_a8_activation(layout, tokens, k);
    return layout.peak_bytes(1);
}

// x is contiguous BF16 [K,T] with the K and T of act.
void groupwise_a8_quantize(const Tensor& x, const GroupwiseA8Activation& act,
                           cudaStream_t stream);

// Stores BF16 projections for up to four jobs that share one format in a single launch.
void groupwise_a8_store(const GroupwiseA8Activation& act, bool q5,
                        std::span<const GroupwiseJob> jobs, cudaStream_t stream);

// residual[:, t] += W x[:, t], rounded to BF16 once.
void groupwise_a8_add_residual(const GroupwiseA8Activation& act, const Weight& weight,
                               Tensor& residual, cudaStream_t stream);

// out = SiLU(gate) * up for a Q4 gate/up parent whose first half of rows is the gate.
void groupwise_a8_swiglu(const GroupwiseA8Activation& act, const Weight& gate_up, Tensor& out,
                         cudaStream_t stream);

} // namespace ninfer::ops::detail
