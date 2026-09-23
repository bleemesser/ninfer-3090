#pragma once

// Orthogonal preparation for the TCQ3 KV cache.
//
// The default is NInfer's plain Sylvester H256, which is what the INT8 profile already
// applies to keys. The llama.cpp fork the codec came from instead uses a sign-randomized
// 128-wide transform, and the difference matters for one reason: the first row of a plain
// Sylvester matrix is all ones, so any component that is constant across the head
// dimension lands entirely in output coordinate 0. Attention keys are known to carry a
// large per-channel mean, so that coordinate can run past the outermost codebook centroid
// and clip. Random signs spread the same component across every coordinate.
//
// kTcq3SignRandomized in tcq3_sign_words.h switches between the two, so the question was
// measured rather than argued: randomized gives perplexity 6.352256, plain 6.368235. The signs are a fixed balanced 256-bit pattern, 128 of each, held as
// eight words in tcq3_sign_words.h. Lane l of a warp carries dimension l + 32r, so word r
// is uniform across the warp and only the bit select varies: no divergent constant-memory
// read, unlike a table of 256 floats indexed by dimension.
//
// With randomization on, the stored transform is R = H . D. That is still orthogonal, so
// scores are unchanged when Q takes the same R, but R is no longer its own inverse: the
// output accumulator needs D . H, which is why unrotate is a separate function.

#include "ops/kv_cache/hadamard_d256.cuh"
#include "ops/kv_cache/tcq3_sign_words.h"

#include <cstdint>

namespace ninfer::ops {

// Every lane of a warp reads the same word, so constant memory broadcasts it in one
// cycle. static gives each translation unit its own copy rather than a link-time clash.
static __constant__ std::uint32_t kKVCacheTcq3SignWords[8] = {
    kTcq3SignWords[0], kTcq3SignWords[1], kTcq3SignWords[2], kTcq3SignWords[3],
    kTcq3SignWords[4], kTcq3SignWords[5], kTcq3SignWords[6], kTcq3SignWords[7],
};

__device__ __forceinline__ void kv_cache_tcq3_apply_signs(float (&values)[8], int lane) {
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const bool negative = ((kKVCacheTcq3SignWords[r] >> lane) & 1u) != 0u;
        values[r]           = negative ? -values[r] : values[r];
    }
}

// Forward preparation, for stored K and V and for the Q that must match them.
__device__ __forceinline__ void kv_cache_tcq3_rotate(float (&values)[8], int lane) {
    if constexpr (kTcq3SignRandomized) { kv_cache_tcq3_apply_signs(values, lane); }
    normalized_hadamard_d256_inplace(values, lane);
}

// Inverse, for the output accumulator, which is still in the rotated domain.
__device__ __forceinline__ void kv_cache_tcq3_unrotate(float (&values)[8], int lane) {
    normalized_hadamard_d256_inplace(values, lane);
    if constexpr (kTcq3SignRandomized) { kv_cache_tcq3_apply_signs(values, lane); }
}

} // namespace ninfer::ops
