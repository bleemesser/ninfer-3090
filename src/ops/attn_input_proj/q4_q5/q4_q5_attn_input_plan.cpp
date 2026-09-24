#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_plan.h"

#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_kernels.h"
#include "ops/common/groupwise_a8.h"
#include "ops/common/groupwise_small_t.h"

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kAnyCols = std::numeric_limits<std::int32_t>::max();

struct ColsSet {
    std::int32_t first;
    std::int32_t last;

    constexpr bool contains(std::int32_t cols) const noexcept {
        return cols >= first && cols <= last;
    }
};

struct RouteSpec {
    ColsSet cols;
    Q4Q5AttnInputScheduleId schedule;
};

constexpr std::array<RouteSpec, 5> kRoutes{{
    {{1, 1}, Q4Q5AttnInputScheduleId::ParentSplitFixed},
    {{2, 8}, Q4Q5AttnInputScheduleId::StreamSmallT},
    {{9, 16}, Q4Q5AttnInputScheduleId::ParentSplitFixed},
    {{17, 20}, Q4Q5AttnInputScheduleId::GroupedHomogeneousPairMmaR16C64S3},
    {{21, kAnyCols}, Q4Q5AttnInputScheduleId::GroupedHomogeneousPairMmaR32C64S4},
}};

constexpr bool catalog_is_closed() noexcept {
    std::int64_t expected = 1;
    for (const RouteSpec& route : kRoutes) {
        if (route.cols.first != expected || route.cols.last < route.cols.first) { return false; }
        expected = static_cast<std::int64_t>(route.cols.last) + 1;
    }
    return kRoutes.back().cols.last == kAnyCols;
}

static_assert(catalog_is_closed(), "attention input routes must be exact and closed");

bool supported_shape(const Q4Q5AttnInputProblem& problem) noexcept {
    return problem.input_rows == 5120 && problem.query_rows == 6144 && problem.kv_rows == 1024 &&
           problem.padded_k == 5120;
}

} // namespace

const char* q4_q5_attn_input_schedule_name(Q4Q5AttnInputScheduleId schedule) noexcept {
    switch (schedule) {
    case Q4Q5AttnInputScheduleId::ParentSplitFixed:
        return "attn_input_proj.q4_q5.parent_split_fixed";
    case Q4Q5AttnInputScheduleId::StreamSmallT:
        return "attn_input_proj.q4_q5.stream_small_t";
    case Q4Q5AttnInputScheduleId::GroupedHomogeneousPairMmaR16C64S3:
        return "attn_input_proj.q4_q5.grouped_homogeneous_pair.mma.r16.c64.s3";
    case Q4Q5AttnInputScheduleId::GroupedHomogeneousPairMmaR32C64S4:
        return "attn_input_proj.q4_q5.grouped_homogeneous_pair.mma.r32.c64.s4";
    case Q4Q5AttnInputScheduleId::A8:
        return "attn_input_proj.q4_q5.a8";
    }
    return "attn_input_proj.q4_q5.unknown";
}

bool q4_q5_attn_input_admits(const Q4Q5AttnInputProblem& problem) noexcept {
    return supported_shape(problem) && problem.cols >= 1 &&
           (problem.policy == LinearPolicy::A16Only || problem.policy == LinearPolicy::AllowA8);
}

Q4Q5AttnInputPlan q4_q5_attn_input_resolve_plan(const Q4Q5AttnInputProblem& problem) {
    if (!q4_q5_attn_input_admits(problem)) {
        throw std::invalid_argument(
            "Q4/Q5 attention input: exact problem or column count is not admitted");
    }
    if (problem.policy == LinearPolicy::AllowA8 && problem.cols >= kGroupwiseA8MinTokens) {
        return {Q4Q5AttnInputScheduleId::A8,
                groupwise_a8_activation_capacity_bytes(problem.cols, problem.input_rows)};
    }

    for (const RouteSpec& route : kRoutes) {
        if (!route.cols.contains(problem.cols)) { continue; }
        return {route.schedule, 0};
    }
    throw std::logic_error("Q4/Q5 attention input: admitted problem has no covering route");
}

std::size_t q4_q5_attn_input_workspace_capacity_bytes(LinearPolicy policy, std::int32_t min_cols,
                                                      std::int32_t max_cols) {
    if (min_cols <= 0 || max_cols < min_cols) {
        throw std::invalid_argument("Q4/Q5 attention input: invalid column interval");
    }
    const auto at = [&](std::int32_t cols) {
        return q4_q5_attn_input_resolve_plan({5120, 6144, 1024, 5120, cols, policy})
            .workspace_bytes;
    };
    return std::max(at(min_cols), at(max_cols));
}

void q4_q5_attn_input_execute_plan(const Q4Q5AttnInputPlan& plan, const Tensor& x,
                                   const Weight& query_key_weight, const Weight& gate_value_weight,
                                   Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                                   WorkspaceArena* workspace, cudaStream_t stream) {
    const Q4Q5AttnInputProblem problem{
        x.ne[0],
        q.ne[0],
        k.ne[0],
        query_key_weight.padded_shape[1],
        x.ne[1],
        plan.schedule == Q4Q5AttnInputScheduleId::A8 ? LinearPolicy::AllowA8
                                                     : LinearPolicy::A16Only};
    const Q4Q5AttnInputPlan resolved = q4_q5_attn_input_resolve_plan(problem);
    if (resolved.schedule != plan.schedule) {
        throw std::invalid_argument("Q4/Q5 attention input: plan does not match exact problem");
    }

    switch (plan.schedule) {
    case Q4Q5AttnInputScheduleId::ParentSplitFixed:
        q4_q5_attn_input_small_t_launch(x, query_key_weight, gate_value_weight, q, gate, k, v,
                                        stream);
        return;
    case Q4Q5AttnInputScheduleId::StreamSmallT: {
        const GroupwiseJob q4_jobs[] = {
            groupwise_job(query_key_weight, 0, problem.query_rows, q, 0),
            groupwise_job(query_key_weight, problem.query_rows, problem.kv_rows, k, 0),
        };
        groupwise_small_t_store(x, false, q4_jobs, stream);
        const GroupwiseJob q5_jobs[] = {
            groupwise_job(gate_value_weight, 0, problem.query_rows, gate, 0),
            groupwise_job(gate_value_weight, problem.query_rows, problem.kv_rows, v, 0),
        };
        groupwise_small_t_store(x, true, q5_jobs, stream);
        return;
    }
    case Q4Q5AttnInputScheduleId::GroupedHomogeneousPairMmaR16C64S3:
        q4_q5_attn_input_grouped_mma_r16_c64_s3_launch(x, query_key_weight, gate_value_weight, q,
                                                       gate, k, v, stream);
        return;
    case Q4Q5AttnInputScheduleId::GroupedHomogeneousPairMmaR32C64S4:
        q4_q5_attn_input_grouped_mma_r32_c64_s4_launch(x, query_key_weight, gate_value_weight, q,
                                                       gate, k, v, stream);
        return;
    case Q4Q5AttnInputScheduleId::A8: {
        if (workspace == nullptr) {
            throw std::invalid_argument("Q4/Q5 attention input: A8 requires workspace");
        }
        auto scratch_scope = workspace->scope();
        const GroupwiseA8Activation act =
            allocate_groupwise_a8_activation(*workspace, problem.cols, problem.input_rows);
        groupwise_a8_quantize(x, act, stream);
        const GroupwiseJob q4_jobs[] = {
            groupwise_job(query_key_weight, 0, problem.query_rows, q, 0),
            groupwise_job(query_key_weight, problem.query_rows, problem.kv_rows, k, 0),
        };
        groupwise_a8_store(act, false, q4_jobs, stream);
        const GroupwiseJob q5_jobs[] = {
            groupwise_job(gate_value_weight, 0, problem.query_rows, gate, 0),
            groupwise_job(gate_value_weight, problem.query_rows, problem.kv_rows, v, 0),
        };
        groupwise_a8_store(act, true, q5_jobs, stream);
        return;
    }
    }
    throw std::logic_error("Q4/Q5 attention input: unknown schedule");
}

void q4_q5_attn_input_dispatch(const Tensor& x, const Weight& query_key_weight,
                               const Weight& gate_value_weight, Tensor& q, Tensor& gate, Tensor& k,
                               Tensor& v, LinearPolicy policy, WorkspaceArena* workspace,
                               cudaStream_t stream) {
    const Q4Q5AttnInputProblem problem{
        x.ne[0], q.ne[0], k.ne[0], query_key_weight.padded_shape[1], x.ne[1], policy};
    const Q4Q5AttnInputPlan plan = q4_q5_attn_input_resolve_plan(problem);
    q4_q5_attn_input_execute_plan(plan, x, query_key_weight, gate_value_weight, q, gate, k, v,
                                  workspace, stream);
}

} // namespace ninfer::ops::detail
