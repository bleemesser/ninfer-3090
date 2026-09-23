#pragma once

#include "core/dtype.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops {

inline constexpr std::int32_t kD256KVCacheHeadDim = 256;

// code_leading_extent is the size of one token's storage for one head in the code
// plane. It equals the head dimension for every profile that stores one code for each
// value. TCQ packs a bitstream instead, so its extent is 49 bytes for each group of
// 128 values.
struct D256KVCacheProfile {
    DType code_dtype;
    std::int32_t quant_group;
    std::int32_t scale_leading_extent;
    std::int32_t code_leading_extent;
};

inline constexpr std::int32_t kD256KVCacheTcq3StreamBytes = 49;

inline D256KVCacheProfile d256_kv_cache_profile(DType dtype) {
    switch (dtype) {
    case DType::BF16:
        return {DType::BF16, 0, 0, kD256KVCacheHeadDim};
    case DType::I8:
        return {DType::I8, 64, 4, kD256KVCacheHeadDim};
    case DType::FP8_E4M3FN:
        return {DType::FP8_E4M3FN, 256, 1, kD256KVCacheHeadDim};
    case DType::U8:
        return {DType::U8, 128, 2, 2 * kD256KVCacheTcq3StreamBytes};
    default:
        throw std::invalid_argument("unsupported D256 KV-cache dtype");
    }
}

} // namespace ninfer::ops
