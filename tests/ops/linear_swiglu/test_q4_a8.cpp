#include "ops/linear_swiglu/linear_swiglu_test_common.h"

#include <array>
#include <exception>
#include <iostream>

int main() {
    using namespace ninfer;
    using namespace ninfer::test::linear_swiglu;

    try {
        // AllowA8 keeps A16 through T=16 and quantizes activations from T=17. The cases cover
        // both sides of that boundary, a partial 128-token tile, and several full tiles.
        constexpr std::array<std::int32_t, 7> kTokenCases{1, 16, 17, 64, 129, 200, 1024};
        const int failures = run_profile(
            "LinearSwiGLU Q4_A8",
            {QType::Q4G64_F16S, 34816, 5120, 17408, 1403U, ActivationCompute::A8}, kTokenCases);
        std::cout << (failures == 0 ? "OK" : "FAIL") << " LinearSwiGLU Q4_A8 correctness\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "LinearSwiGLU Q4_A8 test failed: " << error.what() << '\n';
        return 1;
    }
}
