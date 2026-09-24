#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_plan.h"

#include "ops/common/groupwise_a8.h"
#include "ops/common/groupwise_small_t.h"
#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_kernels.h"

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
    Q4Q5GdnInputScheduleId schedule;
};

constexpr std::array<RouteSpec, 4> kRoutes{{
    {{1, 1}, Q4Q5GdnInputScheduleId::IndependentDirectFixed},
    {{2, 8}, Q4Q5GdnInputScheduleId::StreamSmallT},
    {{9, 16}, Q4Q5GdnInputScheduleId::IndependentDirectFixed},
    {{17, kAnyCols}, Q4Q5GdnInputScheduleId::GroupedMixedMmaR64C128},
}};

constexpr bool catalog_is_closed() noexcept {
    std::int64_t expected = 1;
    for (const RouteSpec& route : kRoutes) {
        if (route.cols.first != expected || route.cols.last < route.cols.first) { return false; }
        expected = static_cast<std::int64_t>(route.cols.last) + 1;
    }
    return kRoutes.back().cols.last == kAnyCols;
}

static_assert(catalog_is_closed(), "GDN input routes must be exact and closed");

bool supported_shape(const Q4Q5GdnInputProblem& problem) noexcept {
    return problem.input_rows == 5120 && problem.qk_rows == 4096 && problem.value_z_rows == 12288 &&
           problem.qkv_rows == 10240 && problem.z_rows == 6144 && problem.padded_k == 5120;
}

} // namespace

const char* q4_q5_gdn_input_schedule_name(Q4Q5GdnInputScheduleId schedule) noexcept {
    switch (schedule) {
    case Q4Q5GdnInputScheduleId::IndependentDirectFixed:
        return "gdn_input_proj.q4_q5.independent_direct_fixed";
    case Q4Q5GdnInputScheduleId::StreamSmallT:
        return "gdn_input_proj.q4_q5.stream_small_t";
    case Q4Q5GdnInputScheduleId::GroupedMixedMmaR64C128:
        return "gdn_input_proj.q4_q5.grouped_mixed.mma.r64.c128";
    case Q4Q5GdnInputScheduleId::A8:
        return "gdn_input_proj.q4_q5.a8";
    }
    return "gdn_input_proj.q4_q5.unknown";
}

const char* q4_q5_gdn_input_conv_schedule_name(Q4Q5GdnInputConvScheduleId schedule) noexcept {
    switch (schedule) {
    case Q4Q5GdnInputConvScheduleId::ProjectionEpilogueFused:
        return "gdn_input_proj_conv.q4_q5.projection_epilogue_fused";
    case Q4Q5GdnInputConvScheduleId::Materialized:
        return "gdn_input_proj_conv.q4_q5.materialized";
    }
    return "gdn_input_proj_conv.q4_q5.unknown";
}

bool q4_q5_gdn_input_admits(const Q4Q5GdnInputProblem& problem) noexcept {
    return supported_shape(problem) && problem.cols >= 1 &&
           (problem.policy == LinearPolicy::A16Only || problem.policy == LinearPolicy::AllowA8);
}

Q4Q5GdnInputPlan q4_q5_gdn_input_resolve_plan(const Q4Q5GdnInputProblem& problem) {
    if (!q4_q5_gdn_input_admits(problem)) {
        throw std::invalid_argument(
            "Q4/Q5 GDN input: exact problem or column count is not admitted");
    }
    if (problem.policy == LinearPolicy::AllowA8 && problem.cols >= kGroupwiseA8MinTokens) {
        return {Q4Q5GdnInputScheduleId::A8,
                groupwise_a8_activation_capacity_bytes(problem.cols, problem.input_rows)};
    }

    for (const RouteSpec& route : kRoutes) {
        if (!route.cols.contains(problem.cols)) { continue; }
        return {route.schedule, 0};
    }
    throw std::logic_error("Q4/Q5 GDN input: admitted problem has no covering route");
}

std::size_t q4_q5_gdn_input_workspace_capacity_bytes(LinearPolicy policy, std::int32_t min_cols,
                                                     std::int32_t max_cols) {
    if (min_cols <= 0 || max_cols < min_cols) {
        throw std::invalid_argument("Q4/Q5 GDN input: invalid column interval");
    }
    const auto at = [&](std::int32_t cols) {
        return q4_q5_gdn_input_resolve_plan({5120, 4096, 12288, 10240, 6144, 5120, cols, policy})
            .workspace_bytes;
    };
    return std::max(at(min_cols), at(max_cols));
}

Q4Q5GdnInputConvPlan q4_q5_gdn_input_conv_resolve_plan(const Q4Q5GdnInputProblem& problem,
                                                       std::int32_t batch_size) {
    if (!q4_q5_gdn_input_admits(problem) || batch_size <= 0 || batch_size > 8) {
        throw std::invalid_argument(
            "Q4/Q5 GDN input conv: exact problem or column count is not admitted");
    }
    if (batch_size > 1) { return {Q4Q5GdnInputConvScheduleId::Materialized}; }
    switch (problem.cols) {
    case 1:
    case 2:
    case 3:
    case 5:
    case 6:
        return {Q4Q5GdnInputConvScheduleId::ProjectionEpilogueFused};
    default:
        return {Q4Q5GdnInputConvScheduleId::Materialized};
    }
}

void q4_q5_gdn_input_execute_plan(const Q4Q5GdnInputPlan& plan, const Tensor& x,
                                  const Weight& qk_weight, const Weight& value_z_weight,
                                  Tensor& qkv, Tensor& z, WorkspaceArena* workspace,
                                  cudaStream_t stream) {
    const Q4Q5GdnInputProblem problem{
        x.ne[0],
        qk_weight.n,
        value_z_weight.n,
        qkv.ne[0],
        z.ne[0],
        qk_weight.padded_shape[1],
        x.ne[1],
        plan.schedule == Q4Q5GdnInputScheduleId::A8 ? LinearPolicy::AllowA8
                                                    : LinearPolicy::A16Only};
    const Q4Q5GdnInputPlan resolved = q4_q5_gdn_input_resolve_plan(problem);
    if (resolved.schedule != plan.schedule) {
        throw std::invalid_argument("Q4/Q5 GDN input: plan does not match exact problem");
    }

    switch (plan.schedule) {
    case Q4Q5GdnInputScheduleId::IndependentDirectFixed: {
        Tensor qk    = qkv.slice(0, 0, problem.qk_rows);
        Tensor value = qkv.slice(0, problem.qk_rows, problem.z_rows);
        q4_q5_gdn_input_independent_launch(x, qk_weight, value_z_weight, qk, value, z, stream);
        return;
    }
    case Q4Q5GdnInputScheduleId::StreamSmallT: {
        const GroupwiseJob q4_jobs[] = {
            groupwise_job(qk_weight, 0, problem.qk_rows, qkv, 0),
        };
        groupwise_small_t_store(x, false, q4_jobs, stream);
        const std::int32_t value_rows  = problem.qkv_rows - problem.qk_rows;
        const GroupwiseJob q5_jobs[] = {
            groupwise_job(value_z_weight, 0, value_rows, qkv, problem.qk_rows),
            groupwise_job(value_z_weight, value_rows, problem.z_rows, z, 0),
        };
        groupwise_small_t_store(x, true, q5_jobs, stream);
        return;
    }
    case Q4Q5GdnInputScheduleId::GroupedMixedMmaR64C128:
        q4_q5_gdn_input_grouped_mma_launch(x, qk_weight, value_z_weight, qkv, z, stream);
        return;
    case Q4Q5GdnInputScheduleId::A8: {
        if (workspace == nullptr) {
            throw std::invalid_argument("Q4/Q5 GDN input: A8 requires workspace");
        }
        auto scratch_scope = workspace->scope();
        const GroupwiseA8Activation act =
            allocate_groupwise_a8_activation(*workspace, problem.cols, problem.input_rows);
        groupwise_a8_quantize(x, act, stream);
        const GroupwiseJob q4_jobs[] = {
            groupwise_job(qk_weight, 0, problem.qk_rows, qkv, 0),
        };
        groupwise_a8_store(act, false, q4_jobs, stream);
        const std::int32_t value_rows  = problem.qkv_rows - problem.qk_rows;
        const GroupwiseJob q5_jobs[] = {
            groupwise_job(value_z_weight, 0, value_rows, qkv, problem.qk_rows),
            groupwise_job(value_z_weight, value_rows, problem.z_rows, z, 0),
        };
        groupwise_a8_store(act, true, q5_jobs, stream);
        return;
    }
    }
    throw std::logic_error("Q4/Q5 GDN input: unknown schedule");
}

void q4_q5_gdn_input_dispatch(const Tensor& x, const Weight& qk_weight,
                              const Weight& value_z_weight, Tensor& qkv, Tensor& z,
                              LinearPolicy policy, WorkspaceArena* workspace, cudaStream_t stream) {
    const Q4Q5GdnInputProblem problem{x.ne[0],   qk_weight.n, value_z_weight.n,
                                      qkv.ne[0], z.ne[0],     qk_weight.padded_shape[1],
                                      x.ne[1],   policy};
    const Q4Q5GdnInputPlan plan = q4_q5_gdn_input_resolve_plan(problem);
    q4_q5_gdn_input_execute_plan(plan, x, qk_weight, value_z_weight, qkv, z, workspace, stream);
}

} // namespace ninfer::ops::detail
