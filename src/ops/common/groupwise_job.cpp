#include "ops/common/groupwise_job.h"

#include <stdexcept>

namespace ninfer::ops::detail {

GroupwiseJob groupwise_job(const Weight& weight, std::int32_t row_begin, std::int32_t row_count,
                           Tensor& out, std::int32_t out_row_offset) {
    if (row_begin < 0 || row_count <= 0 || row_begin + row_count > weight.n ||
        (row_count % 16) != 0) {
        throw std::invalid_argument("groupwise job: row range is invalid");
    }
    const std::int64_t groups = weight.padded_shape[1] / 64;
    const auto* codes         = static_cast<const std::uint8_t*>(weight.qdata) +
                                static_cast<std::int64_t>(row_begin) * groups * 32;
    const auto* high          = weight.qtype == QType::Q5G64_F16S
                                    ? static_cast<const std::uint8_t*>(weight.qhigh) +
                                          static_cast<std::int64_t>(row_begin) * groups * 8
                                    : nullptr;
    const auto* scales        = static_cast<const std::uint8_t*>(weight.scales) +
                                static_cast<std::int64_t>(row_begin) * groups * 2;
    return {codes,     high,          scales, row_count, static_cast<__nv_bfloat16*>(out.data),
            out.ne[0], out_row_offset};
}

} // namespace ninfer::ops::detail
