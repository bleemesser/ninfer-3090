#!/usr/bin/env python3
"""Emit the NInfer TCQ codec header.

The fitted tables come from reference/tcq3_constants.h, which extract-constants.py
generates from the llama.cpp fork. Keeping the generation scripted means the
NInfer copy cannot drift from the reference the port is validated against.
"""

import pathlib
import re

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SRC = HERE / "reference" / "tcq3_constants.h"
DST = ROOT / "src" / "ops" / "kv_cache" / "tcq3_g128_codec.cuh"
TEST_DST = ROOT / "tests" / "ops" / "softmax_attention" / "tcq3_host_tables.h"

HEADER = '''#pragma once

// Trellis-coded 3-bit KV-cache codec, ported from the turbo3_tcq cache type in the
// llama.cpp TurboQuant fork. One block covers 128 values in 52 bytes, which is 3.25
// bits for each value: an FP16 norm, six initial-state bits, and 128 symbols of three
// bits. Decode reads nine bits at a symbol boundary and indexes a 512-entry codebook,
// so it has no sequential dependency. Encode needs a Viterbi search and lives in the
// append Op.
//
// A 256-wide row is two independent blocks. The codebooks and the Hadamard sign tables
// were fitted at 128 dimensions, so the group size stays 128 rather than widening to
// match the row.

#include "ops/common/memory.cuh"
#include "ops/kernel/paged_kv_address.cuh"

#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops {{

inline constexpr int kKVCacheTcq3HeadDim = 256;
inline constexpr int kKVCacheTcq3Group = 128;
inline constexpr int kKVCacheTcq3Groups = kKVCacheTcq3HeadDim / kKVCacheTcq3Group;
inline constexpr int kKVCacheTcq3States = 512;
inline constexpr int kKVCacheTcq3StreamBytes = 49;
inline constexpr int kKVCacheTcq3RowBytes = kKVCacheTcq3StreamBytes * kKVCacheTcq3Groups;

// Decode-time scale factors. The fork forces the encode-time V factor to 1.0 and applies
// 1.02 here instead; the perplexity measurement the port targets used that arrangement.
inline constexpr float kKVCacheTcq3DecodeAlphaK = {alpha_k}f;
inline constexpr float kKVCacheTcq3DecodeAlphaV = {alpha_v}f;

'''

FOOTER = '''
// Byte offset of a group's bitstream within the code plane.
template <typename Geometry>
__device__ __forceinline__ std::int64_t
kv_cache_tcq3_stream_index(int physical_page, int kv_head, int group, int page_offset) {{
    return paged_kv_element_offset<kKVCacheTcq3RowBytes, Geometry::KVHeads>(
        physical_page, kv_head, page_offset, group * kKVCacheTcq3StreamBytes);
}}

// Index of a group's FP16 norm within the scale plane.
template <typename Geometry>
__device__ __forceinline__ std::int64_t
kv_cache_tcq3_norm_index(int physical_page, int kv_head, int group, int page_offset) {{
    return paged_kv_element_offset<kKVCacheTcq3Groups, Geometry::KVHeads>(physical_page, kv_head,
                                                                          page_offset, group);
}}

template <typename Geometry>
__device__ __forceinline__ std::int64_t kv_cache_tcq3_src_index(int kv_head, int d, int token) {{
    return static_cast<std::int64_t>(d) +
           static_cast<std::int64_t>(kKVCacheTcq3HeadDim) *
               (static_cast<std::int64_t>(kv_head) +
                static_cast<std::int64_t>(Geometry::KVHeads) * token);
}}

// Trellis state for symbol t. The nine bits span at most two bytes, and the six
// initial-state bits at the head of the stream make the first states well defined.
__device__ __forceinline__ int kv_cache_tcq3_state(const std::uint8_t* stream, int t) {{
    const int bit = t * 3;
    const std::uint16_t raw =
        static_cast<std::uint16_t>(stream[bit >> 3]) |
        static_cast<std::uint16_t>(static_cast<std::uint16_t>(stream[(bit >> 3) + 1]) << 8);
    return (raw >> (bit & 7)) & 0x1FF;
}}

__device__ __forceinline__ float kv_cache_tcq3_decode(const std::uint8_t* stream, int t,
                                                      float norm, bool is_key) {{
    const int state = kv_cache_tcq3_state(stream, t);
    const float* codebook = is_key ? kKVCacheTcq3CodebookK : kKVCacheTcq3CodebookV;
    return codebook[state] * norm;
}}

// Scale to multiply a stored FP16 norm by before decoding a group.
__device__ __forceinline__ float kv_cache_tcq3_norm_scale(__half stored, bool is_key) {{
    return __half2float(stored) * (is_key ? kKVCacheTcq3DecodeAlphaK : kKVCacheTcq3DecodeAlphaV);
}}

// Trellis states for eight consecutive symbols starting at a multiple of eight.
//
// Symbol t starts at bit 3t, so a run of eight spans bits 3*t0 through 3*t0+29, which is
// 30 bits. When t0 is a multiple of eight, 3*t0 is a multiple of 24 and the run therefore
// starts on a byte boundary, so four bytes hold every state in the run. Reading those
// four bytes once replaces the sixteen byte loads that eight separate state reads cost.
__device__ __forceinline__ void kv_cache_tcq3_states8(const std::uint8_t* stream, int t0,
                                                      int (&states)[8]) {{
    const int byte0 = (t0 * 3) >> 3;
    const std::uint32_t word = static_cast<std::uint32_t>(stream[byte0]) |
                               (static_cast<std::uint32_t>(stream[byte0 + 1]) << 8) |
                               (static_cast<std::uint32_t>(stream[byte0 + 2]) << 16) |
                               (static_cast<std::uint32_t>(stream[byte0 + 3]) << 24);
#pragma unroll
    for (int i = 0; i < 8; ++i) {{ states[i] = static_cast<int>((word >> (3 * i)) & 0x1FFu); }}
}}

}} // namespace ninfer::ops
'''


def read_table(text, name, count):
    match = re.search(re.escape(name) + r"\[" + str(count) + r"\]\s*=\s*\{(.*?)\};", text, re.S)
    if not match:
        raise SystemExit(f"{name} not found in {SRC}")
    values = re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?f", match.group(1))
    if len(values) != count:
        raise SystemExit(f"{name}: found {len(values)} values, expected {count}")
    return values


def read_scalar(text, name):
    match = re.search(re.escape(name) + r"\s*=\s*(-?\d+\.?\d*)f", text)
    if not match:
        raise SystemExit(f"{name} not found in {SRC}")
    return match.group(1)


TEST_HEADER = '''#pragma once

// Host-side copy of the turbo3_tcq codec tables, for the test oracle.
//
// The Op reads the same values out of __constant__ memory in
// src/ops/kv_cache/tcq3_g128_codec.cuh. Both headers come from one run of
// gen-ninfer-codec-header.py over the reference constants, so the oracle cannot
// drift from the kernel, and neither one transcribes a fitted value by hand.
//
// The test encoder is written against these tables independently of the kernel.

namespace ninfer::test {{

inline constexpr int kTcq3HostGroup       = 128;
inline constexpr int kTcq3HostStates      = 512;
inline constexpr int kTcq3HostStreamBytes = 49;
inline constexpr float kTcq3HostDecodeAlphaK = {alpha_k}f;
inline constexpr float kTcq3HostDecodeAlphaV = {alpha_v}f;

'''

TEST_FOOTER = '''
}} // namespace ninfer::test
'''


def emit_host(name, values):
    lines = [f"inline constexpr float {name}[{len(values)}] = {{"]
    for i in range(0, len(values), 8):
        lines.append("    " + " ".join(f"{v}," for v in values[i : i + 8]))
    lines.append("};")
    lines.append("")
    return lines


def emit(name, values):
    # static, so each translation unit that includes the header gets its own copy
    # rather than colliding at link time.
    lines = [f"static __constant__ float {name}[{len(values)}] = {{"]
    for i in range(0, len(values), 8):
        lines.append("    " + " ".join(f"{v}," for v in values[i : i + 8]))
    lines.append("};")
    lines.append("")
    return lines


def main():
    text = SRC.read_text()
    out = [
        HEADER.format(
            alpha_k=read_scalar(text, "kTcq3DecodeAlphaK"),
            alpha_v=read_scalar(text, "kTcq3DecodeAlphaV"),
        )
    ]
    body = []
    for dst_name, src_name, count in (
        ("kKVCacheTcq3CodebookK", "kTcq3CodebookK", 512),
        ("kKVCacheTcq3CodebookV", "kTcq3CodebookV", 512),
        ("kKVCacheTcq3HadamardSigns1", "kWhtSigns1", 128),
        ("kKVCacheTcq3HadamardSigns2", "kWhtSigns2", 128),
    ):
        body += emit(dst_name, read_table(text, src_name, count))
    out.append("\n".join(body))
    out.append(FOOTER.format())
    DST.write_text("".join(out))
    print(f"wrote {DST}")

    test_out = [
        TEST_HEADER.format(
            alpha_k=read_scalar(text, "kTcq3DecodeAlphaK"),
            alpha_v=read_scalar(text, "kTcq3DecodeAlphaV"),
        )
    ]
    test_body = []
    for dst_name, src_name, count in (
        ("kTcq3HostCodebookK", "kTcq3CodebookK", 512),
        ("kTcq3HostCodebookV", "kTcq3CodebookV", 512),
    ):
        test_body += emit_host(dst_name, read_table(text, src_name, count))
    test_out.append("\n".join(test_body))
    test_out.append(TEST_FOOTER.format())
    TEST_DST.write_text("".join(test_out))
    print(f"wrote {TEST_DST}")


if __name__ == "__main__":
    main()
