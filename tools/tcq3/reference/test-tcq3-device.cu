// Checks that NInfer's device decode agrees with the CPU reference, value for value.
//
// The CPU reference is itself validated bit for bit against the fork's encoder, so
// agreement here chains the device path back to the fork.

#include "tcq3_reference.h"

#include "ops/kv_cache/tcq3_g128_codec.cuh"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

using ninfer::ops::kKVCacheTcq3Group;

__global__ void decode_kernel(const std::uint8_t* streams, const __half* norms, int groups,
                              int is_key, float* out) {
    const int g = blockIdx.x;
    const int t = threadIdx.x;
    if (g >= groups || t >= kKVCacheTcq3Group) { return; }
    const std::uint8_t* stream = streams + (std::size_t)g * 49;
    const float norm = ninfer::ops::kv_cache_tcq3_norm_scale(norms[g], is_key != 0);
    out[(std::size_t)g * kKVCacheTcq3Group + t] =
        ninfer::ops::kv_cache_tcq3_decode(stream, t, norm, is_key != 0);
}

int main() {
    const int groups = 256;
    unsigned int seed = 7;

    std::vector<float> src((std::size_t)groups * 128);
    for (auto& v : src) {
        const double u1 = (rand_r(&seed) + 1.0) / (RAND_MAX + 2.0);
        const double u2 = (rand_r(&seed) + 1.0) / (RAND_MAX + 2.0);
        v = (float)(sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2));
    }

    int rc = 0;
    for (int is_key = 1; is_key >= 0; is_key--) {
        std::vector<std::uint8_t> streams((std::size_t)groups * 49);
        std::vector<__half> norms(groups);
        std::vector<float> expected((std::size_t)groups * 128);

        for (int g = 0; g < groups; g++) {
            tcq3_block blk;
            tcq3_encode_group(src.data() + (std::size_t)g * 128, is_key, nullptr, nullptr, &blk);
            memcpy(streams.data() + (std::size_t)g * 49, blk.qs, 49);
            memcpy(&norms[g], &blk.norm, sizeof(std::uint16_t));
            for (int t = 0; t < 128; t++) {
                expected[(std::size_t)g * 128 + t] = tcq3_decode_value(&blk, t, is_key);
            }
        }

        std::uint8_t* d_streams = nullptr;
        __half* d_norms = nullptr;
        float* d_out = nullptr;
        cudaMalloc(&d_streams, streams.size());
        cudaMalloc(&d_norms, norms.size() * sizeof(__half));
        cudaMalloc(&d_out, expected.size() * sizeof(float));
        cudaMemcpy(d_streams, streams.data(), streams.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_norms, norms.data(), norms.size() * sizeof(__half), cudaMemcpyHostToDevice);

        decode_kernel<<<groups, 128>>>(d_streams, d_norms, groups, is_key, d_out);
        cudaDeviceSynchronize();

        std::vector<float> got(expected.size());
        cudaMemcpy(got.data(), d_out, got.size() * sizeof(float), cudaMemcpyDeviceToHost);

        int mismatches = 0;
        double worst = 0.0;
        for (std::size_t i = 0; i < got.size(); i++) {
            const double d = fabs((double)got[i] - expected[i]);
            if (d > worst) { worst = d; }
            if (got[i] != expected[i]) { mismatches++; }
        }
        printf("%s: %d groups, %d of %zu values differ, worst %.3e\n", is_key ? "K" : "V", groups,
               mismatches, got.size(), worst);
        if (mismatches != 0) {
            printf("%s FAILED: device decode does not match the reference\n", is_key ? "K" : "V");
            rc = 1;
        }

        cudaFree(d_streams);
        cudaFree(d_norms);
        cudaFree(d_out);
    }
    return rc;
}
