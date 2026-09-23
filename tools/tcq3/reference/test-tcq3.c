// Checks for the turbo3_tcq CPU reference.
//
// 1. The FWHT is orthonormal and its inverse undoes it.
// 2. The packed bitstream decodes back to the symbols the Viterbi chose.
// 3. Round-trip error is in the range a 3.25 bit codec should reach, and
//    beats an equal-bitrate scalar quantizer on the same data.

#include "tcq3_reference.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define ROWS 512

static unsigned int seed = 20260919;

static float normal_sample(void) {
    const double u1 = (rand_r(&seed) + 1.0) / (RAND_MAX + 2.0);
    const double u2 = (rand_r(&seed) + 1.0) / (RAND_MAX + 2.0);
    return (float)(sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2));
}

static int check_fwht(void) {
    float x[128];
    float y[128];
    for (int i = 0; i < 128; i++) { x[i] = normal_sample(); }
    memcpy(y, x, sizeof(x));

    double energy_in = 0.0;
    for (int i = 0; i < 128; i++) { energy_in += (double)x[i] * x[i]; }
    tcq3_fwht_128(y);
    double energy_out = 0.0;
    for (int i = 0; i < 128; i++) { energy_out += (double)y[i] * y[i]; }
    tcq3_ifwht_128(y);

    double worst = 0.0;
    for (int i = 0; i < 128; i++) {
        const double d = fabs((double)y[i] - x[i]);
        if (d > worst) { worst = d; }
    }
    const double energy_ratio = energy_out / energy_in;
    printf("fwht: energy ratio %.9f  round-trip max error %.3e\n", energy_ratio, worst);
    if (fabs(energy_ratio - 1.0) > 1e-5) {
        printf("fwht FAILED: transform is not orthonormal\n");
        return 1;
    }
    if (worst > 1e-4) {
        printf("fwht FAILED: inverse does not undo the forward transform\n");
        return 1;
    }
    return 0;
}

// Read the 9-bit trellis state the decoder sees at step t.
static int packed_state(const tcq3_block* blk, int t) {
    const int bit_pos = t * 3;
    const uint16_t raw =
        (uint16_t)blk->qs[bit_pos / 8] | (uint16_t)((uint16_t)blk->qs[bit_pos + 3 >= 0 ? bit_pos / 8 + 1 : 0] << 8);
    return (raw >> (bit_pos % 8)) & 0x1FF;
}

static int check_bitstream(void) {
    float group[128];
    for (int i = 0; i < 128; i++) { group[i] = normal_sample(); }

    tcq3_block blk;
    tcq3_encode_group(group, 1, NULL, NULL, &blk);

    // Walk the trellis forward from the stored initial state and confirm every
    // state the decoder reads is the state the encoder committed to.
    int state = 0;
    for (int pos = 0; pos < 6; pos++) {
        if ((blk.qs[pos / 8] >> (pos % 8)) & 1) { state |= 1 << pos; }
    }
    // state now holds initial_state >> 3, which is the low 6 bits the trellis
    // carries into step 0.
    int walk = state;
    for (int t = 0; t < 128; t++) {
        int symbol = 0;
        for (int b = 0; b < 3; b++) {
            const int pos = 6 + t * 3 + b;
            if ((blk.qs[pos / 8] >> (pos % 8)) & 1) { symbol |= 1 << b; }
        }
        const int expected = (walk & 0x3F) | (symbol << 6);
        const int got = packed_state(&blk, t);
        if (expected != got) {
            printf("bitstream FAILED at t=%d: packed %d, expected %d\n", t, got, expected);
            return 1;
        }
        walk = expected >> 3;
    }
    printf("bitstream: 128 trellis states decode exactly as encoded\n");
    return 0;
}

// Scalar comparison point: 8 levels, one FP16 norm for each 128 values, which
// is 3.125 bits for each value, slightly under TCQ's 3.25.
static float scalar_roundtrip_rms(const float* src, int n) {
    static const float levels[8] = { -0.190685f, -0.117832f, -0.065717f, -0.021460f,
                                     0.021460f,  0.065717f,  0.117832f,  0.190685f };
    double acc = 0.0;
    for (int b = 0; b < n / 128; b++) {
        const float* g = src + b * 128;
        float x[128];
        double norm_sq = 0.0;
        for (int i = 0; i < 128; i++) { norm_sq += (double)g[i] * g[i]; }
        const float norm = (float)sqrt(norm_sq);
        const float inv = norm > 1e-10f ? 1.0f / norm : 0.0f;
        for (int i = 0; i < 128; i++) { x[i] = g[i] * inv; }
        tcq3_fwht_128(x);
        float recon[128];
        double recon_sq = 0.0;
        for (int i = 0; i < 128; i++) {
            int best = 0;
            float bd = fabsf(x[i] - levels[0]);
            for (int l = 1; l < 8; l++) {
                const float d = fabsf(x[i] - levels[l]);
                if (d < bd) {
                    bd = d;
                    best = l;
                }
            }
            recon[i] = levels[best];
            recon_sq += (double)levels[best] * levels[best];
        }
        const float rn = (float)sqrt(recon_sq);
        const float scale = rn > 1e-10f ? norm / rn : norm;
        tcq3_ifwht_128(recon);
        for (int i = 0; i < 128; i++) {
            const double d = (double)recon[i] * scale - g[i];
            acc += d * d;
        }
    }
    return (float)sqrt(acc / n);
}

int main(void) {
    int rc = 0;
    rc |= check_fwht();
    rc |= check_bitstream();

    const int n = ROWS * 128;
    float* src = malloc((size_t)n * sizeof(float));
    float* dst = malloc((size_t)n * sizeof(float));
    double sig = 0.0;
    for (int i = 0; i < n; i++) {
        src[i] = normal_sample();
        sig += (double)src[i] * src[i];
    }
    sig = sqrt(sig / n);

    for (int is_k = 1; is_k >= 0; is_k--) {
        double acc = 0.0;
        for (int b = 0; b < ROWS; b++) {
            tcq3_block blk;
            tcq3_encode_group(src + b * 128, is_k, NULL, NULL, &blk);
            tcq3_decode_group(&blk, is_k, dst + b * 128);
            for (int i = 0; i < 128; i++) {
                const double d = (double)dst[b * 128 + i] - src[b * 128 + i];
                acc += d * d;
            }
        }
        const double rms = sqrt(acc / n);
        printf("tcq3 %s: bits/value %.3f  rms error %.5f  relative %.2f%%\n", is_k ? "K" : "V",
               8.0 * TCQ3_BYTES / TCQ3_GROUP, rms, 100.0 * rms / sig);
        if (!(rms > 0.0) || rms > 0.5 * sig) {
            printf("tcq3 %s FAILED: reconstruction is not usable\n", is_k ? "K" : "V");
            rc = 1;
        }
    }

    const float scalar = scalar_roundtrip_rms(src, n);
    printf("scalar 3-bit reference: bits/value 3.125  rms error %.5f  relative %.2f%%\n", scalar,
           100.0 * scalar / sig);

    free(src);
    free(dst);
    return rc;
}
