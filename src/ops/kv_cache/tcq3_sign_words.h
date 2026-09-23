#pragma once

// Sign pattern for the TCQ3 randomized rotation, and the switch that turns it on, shared
// by the kernels and the test oracle so the two cannot drift. Plain C++ with no CUDA,
// because the oracle is host code.
//
// Randomization is on because it measures better: perplexity 6.352256 against 6.368235
// for the plain Sylvester transform, on the same 94453-token corpus. See tcq3_rotation.cuh
// for why the plain transform loses.
//
// A balanced 256-bit pattern, 128 of each sign, packed into eight words. Dimension d uses
// bit d % 32 of word d / 32. A warp lane holds dimension lane + 32r, so word r is uniform
// across the warp and only the bit select varies.

#include <cstdint>

namespace ninfer::ops {

inline constexpr bool kTcq3SignRandomized = true;

inline constexpr std::uint32_t kTcq3SignWords[8] = {
    0x3e7944bcu, 0xb6a940bbu, 0xb160249au, 0x5f34214bu,
    0xc079d1feu, 0x984be08fu, 0x60f5d05eu, 0x675247efu,
};

} // namespace ninfer::ops
