// CPU reference for the turbo3_tcq KV codec.
//
// This mirrors k_set_rows_turbo3_tcq and dequantize_V_turbo3_tcq_cb from the
// llama.cpp TurboQuant fork, in scalar C. It exists so the NInfer port has a
// known-good target: every CUDA kernel written for the port is checked against
// this file, and this file is checked against the fork's own dump buffers.
//
// Deliberate deviations from the fork are marked DEVIATION. There should be
// none that affect output bits.

#pragma once

#include "tcq3_constants.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

#define TCQ3_GROUP 128 // values in one block
#define TCQ3_STATES 512
#define TCQ3_BYTES 52 // 2 norm + 49 bitstream + 1 padding

typedef struct {
    uint16_t norm; // FP16 bits
    uint8_t qs[49];
    uint8_t pad;
} tcq3_block;

// Half conversion, round to nearest even, matching __float2half.
static inline uint16_t tcq3_f32_to_f16(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t man = x & 0x7FFFFFu;
    if (((x >> 23) & 0xFF) == 0xFF) { return (uint16_t)(sign | 0x7C00u | (man ? 0x200u : 0u)); }
    if (exp >= 0x1F) { return (uint16_t)(sign | 0x7C00u); }
    if (exp <= 0) {
        if (exp < -10) { return (uint16_t)sign; }
        man |= 0x800000u;
        const int shift = 14 - exp;
        const uint32_t round = (man >> (shift - 1)) & 1u;
        const uint32_t sticky = (man & ((1u << (shift - 1)) - 1u)) != 0u;
        uint32_t out = man >> shift;
        if (round && (sticky || (out & 1u))) { out++; }
        return (uint16_t)(sign | out);
    }
    const uint32_t round = (man >> 12) & 1u;
    const uint32_t sticky = (man & 0xFFFu) != 0u;
    uint32_t out = (uint32_t)(exp << 10) | (man >> 13);
    if (round && (sticky || (out & 1u))) { out++; }
    return (uint16_t)(sign | out);
}

static inline float tcq3_f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1Fu;
    const uint32_t man = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {
            int e = -1;
            uint32_t m = man;
            do {
                m <<= 1;
                e++;
            } while ((m & 0x400u) == 0);
            bits = sign | ((uint32_t)(127 - 15 - e) << 23) | ((m & 0x3FFu) << 13);
        }
    } else if (exp == 0x1F) {
        bits = sign | 0x7F800000u | (man << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

// Normalized fast Walsh-Hadamard transform with the fork's two sign tables.
// The CUDA kernel splits this across warp shuffles and two shared-memory
// stages. The result is the same, because the butterfly order is identical.
static inline void tcq3_fwht_128(float* x) {
    for (int i = 0; i < 128; i++) { x[i] *= kWhtSigns1[i]; }
    for (int h = 1; h < 128; h <<= 1) {
        for (int i = 0; i < 128; i++) {
            if (i & h) { continue; }
            const float a = x[i];
            const float b = x[i | h];
            x[i] = a + b;
            x[i | h] = a - b;
        }
    }
    const float inv_sqrt_128 = 0.08838834764831845f;
    for (int i = 0; i < 128; i++) { x[i] *= inv_sqrt_128 * kWhtSigns2[i]; }
}

// Inverse of tcq3_fwht_128. The Hadamard matrix is symmetric and squares to
// 128 times the identity, so inverting means applying the sign tables in the
// opposite order.
static inline void tcq3_ifwht_128(float* x) {
    for (int i = 0; i < 128; i++) { x[i] *= kWhtSigns2[i]; }
    for (int h = 1; h < 128; h <<= 1) {
        for (int i = 0; i < 128; i++) {
            if (i & h) { continue; }
            const float a = x[i];
            const float b = x[i | h];
            x[i] = a + b;
            x[i | h] = a - b;
        }
    }
    const float inv_sqrt_128 = 0.08838834764831845f;
    for (int i = 0; i < 128; i++) { x[i] *= inv_sqrt_128 * kWhtSigns1[i]; }
}

// Trellis search over an already rotated and normalized group. Split out so
// the fork's TURBO_TCQ_DUMP_ERRORS capture, which records x after the
// transform, can be replayed against this code directly. symbols_out may be
// NULL. Returns the initial state.
static inline int tcq3_encode_rotated(const float* x, int is_k, float grp_norm,
                                      uint8_t* symbols_out, tcq3_block* out) {
    const float* cb = is_k ? kTcq3CodebookK : kTcq3CodebookV;
    const float alpha = is_k ? kTcq3NormAlphaK : kTcq3NormAlphaV;

    // Viterbi over the right-shift trellis: next = (prev >> 3) | (out << 6).
    // The best predecessor depends only on the low 6 bits of the next state.
    float cost[TCQ3_STATES];
    float next[TCQ3_STATES];
    uint8_t bt[TCQ3_GROUP][64];
    for (int s = 0; s < TCQ3_STATES; s++) { cost[s] = 0.0f; }

    for (int t = 0; t < TCQ3_GROUP; t++) {
        float pred_min[64];
        for (int low = 0; low < 64; low++) {
            const int base = low << 3;
            float best = cost[base];
            int best_p = 0;
            for (int p = 1; p < 8; p++) {
                const float c = cost[base | p];
                if (c < best) {
                    best = c;
                    best_p = p;
                }
            }
            pred_min[low] = best;
            bt[t][low] = (uint8_t)best_p;
        }
        for (int s = 0; s < TCQ3_STATES; s++) {
            const float d = x[t] - cb[s];
            next[s] = pred_min[s & 0x3F] + d * d;
        }
        memcpy(cost, next, sizeof(cost));
    }

    float best = cost[0];
    int best_state = 0;
    for (int s = 1; s < TCQ3_STATES; s++) {
        if (cost[s] < best) {
            best = cost[s];
            best_state = s;
        }
    }

    uint8_t symbols[TCQ3_GROUP];
    int state = best_state;
    for (int t = TCQ3_GROUP - 1; t >= 0; t--) {
        symbols[t] = (uint8_t)(state >> 6);
        const int p = bt[t][state & 0x3F];
        state = ((state & 0x3F) << 3) | p;
    }
    const int initial_state = state;

    // Norm correction uses the codebook values the chosen path actually walks.
    float recon_sq = 0.0f;
    int cur = initial_state;
    for (int t = 0; t < TCQ3_GROUP; t++) {
        cur = (cur >> 3) | ((int)symbols[t] << 6);
        const float c = cb[cur];
        recon_sq += c * c;
    }
    const float recon_norm = sqrtf(recon_sq);
    float corrected = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;
    corrected *= alpha;

    memset(out->qs, 0, sizeof(out->qs));
    const int init_bits = (initial_state >> 3) & 0x3F;
    for (int pos = 0; pos < 6; pos++) {
        if ((init_bits >> pos) & 1) { out->qs[pos / 8] |= (uint8_t)(1u << (pos % 8)); }
    }
    for (int t = 0; t < TCQ3_GROUP; t++) {
        for (int b = 0; b < 3; b++) {
            if ((symbols[t] >> b) & 1) {
                const int pos = 6 + t * 3 + b;
                out->qs[pos / 8] |= (uint8_t)(1u << (pos % 8));
            }
        }
    }
    out->pad = 0;
    out->norm = tcq3_f32_to_f16(corrected);
    if (symbols_out) { memcpy(symbols_out, symbols, sizeof(symbols)); }
    return initial_state;
}

// Encode one group of 128 raw values. is_k selects the K or V codebook and
// alpha. channel_scale and channel_mean are optional per-channel calibration
// taps; pass NULL for either to skip it, which matches an uncalibrated run.
static inline void tcq3_encode_group(const float* src, int is_k, const float* channel_scale,
                                     const float* channel_mean, tcq3_block* out) {
    float x[TCQ3_GROUP];
    for (int i = 0; i < TCQ3_GROUP; i++) {
        float v = src[i];
        if (channel_mean) { v -= channel_mean[i]; }
        if (channel_scale) { v *= channel_scale[i]; }
        x[i] = v;
    }

    float norm_sq = 0.0f;
    for (int i = 0; i < TCQ3_GROUP; i++) { norm_sq += x[i] * x[i]; }
    const float grp_norm = sqrtf(norm_sq);
    const float inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;
    for (int i = 0; i < TCQ3_GROUP; i++) { x[i] *= inv_norm; }

    tcq3_fwht_128(x);
    tcq3_encode_rotated(x, is_k, grp_norm, NULL, out);
}

// Decode one value, matching the random-access read the attention kernels use.
static inline float tcq3_decode_value(const tcq3_block* blk, int t, int is_k) {
    const float* cb = is_k ? kTcq3CodebookK : kTcq3CodebookV;
    const int bit_pos = t * 3;
    const uint16_t raw =
        (uint16_t)blk->qs[bit_pos / 8] | (uint16_t)((uint16_t)blk->qs[bit_pos / 8 + 1] << 8);
    const int state = (raw >> (bit_pos % 8)) & 0x1FF;
    const float alpha = is_k ? kTcq3DecodeAlphaK : kTcq3DecodeAlphaV;
    return cb[state] * (tcq3_f16_to_f32(blk->norm) * alpha);
}

// Decode a whole group back to the pre-rotation domain.
static inline void tcq3_decode_group(const tcq3_block* blk, int is_k, float* dst) {
    for (int t = 0; t < TCQ3_GROUP; t++) { dst[t] = tcq3_decode_value(blk, t, is_k); }
    // The attention kernels consume the rotated domain directly and undo the
    // rotation on the output, so the inverse here is for testing only.
    tcq3_ifwht_128(dst);
}
