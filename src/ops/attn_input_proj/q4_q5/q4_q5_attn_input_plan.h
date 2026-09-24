#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

enum class Q4Q5AttnInputScheduleId {
    ParentSplitFixed,
    StreamSmallT,
    GroupedHomogeneousPairMmaR16C64S3,
    GroupedHomogeneousPairMmaR32C64S4,
    A8,
};

struct Q4Q5AttnInputProblem {
    std::int32_t input_rows;
    std::int32_t query_rows;
    std::int32_t kv_rows;
    std::int32_t padded_k;
    std::int32_t cols;
    LinearPolicy policy = LinearPolicy::A16Only;
};

struct Q4Q5AttnInputPlan {
    Q4Q5AttnInputScheduleId schedule;
    std::size_t workspace_bytes = 0;
};

const char* q4_q5_attn_input_schedule_name(Q4Q5AttnInputScheduleId schedule) noexcept;

bool q4_q5_attn_input_admits(const Q4Q5AttnInputProblem& problem) noexcept;
Q4Q5AttnInputPlan q4_q5_attn_input_resolve_plan(const Q4Q5AttnInputProblem& problem);

std::size_t q4_q5_attn_input_workspace_capacity_bytes(LinearPolicy policy, std::int32_t min_cols,
                                                      std::int32_t max_cols);

void q4_q5_attn_input_execute_plan(const Q4Q5AttnInputPlan& plan, const Tensor& x,
                                   const Weight& query_key_weight, const Weight& gate_value_weight,
                                   Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                                   WorkspaceArena* workspace, cudaStream_t stream);
void q4_q5_attn_input_dispatch(const Tensor& x, const Weight& query_key_weight,
                               const Weight& gate_value_weight, Tensor& q, Tensor& gate, Tensor& k,
                               Tensor& v, LinearPolicy policy, WorkspaceArena* workspace,
                               cudaStream_t stream);

} // namespace ninfer::ops::detail
