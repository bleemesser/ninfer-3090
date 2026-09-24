#pragma once

// One row range of a Q4G64 or Q5G64 RowSplit weight and where its projection lands. Shared by
// the groupwise projection kernels that serve several output tensors from one launch.

#include "core/tensor.h"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

// Row r of the range lands in out[t * out_ld + out_row_offset + r].
struct GroupwiseJob {
    const std::uint8_t* codes   = nullptr;
    const std::uint8_t* high    = nullptr;
    const std::uint8_t* scales  = nullptr;
    std::int32_t rows           = 0;
    __nv_bfloat16* out          = nullptr;
    std::int32_t out_ld         = 0;
    std::int32_t out_row_offset = 0;
};

// Rows [row_begin, row_begin + row_count) of weight, stored into out from out_row_offset.
GroupwiseJob groupwise_job(const Weight& weight, std::int32_t row_begin, std::int32_t row_count,
                           Tensor& out, std::int32_t out_row_offset);

} // namespace ninfer::ops::detail
