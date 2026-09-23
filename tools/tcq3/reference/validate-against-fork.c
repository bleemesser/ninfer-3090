// Replays the fork's TCQ encode dump through the CPU reference.
//
// Capture a dump first:
//   TURBO_TCQ_DUMP_ERRORS=64 llama-perplexity -m <model> -f <text> -c 512 \
//       --chunks 1 -ngl 99 -fa on --cache-type-k turbo3_tcq --cache-type-v turbo3_tcq
//
// /tmp/tcq_errors.bin holds int32 group count, then count*128 floats of x
// after the Walsh-Hadamard transform, then count*128 output symbols. The dump
// does not record whether a group was K or V, so try both codebooks and report
// which one matches.

#include "tcq3_reference.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "/tmp/tcq_errors.bin";
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        return 2;
    }

    int32_t count = 0;
    if (fread(&count, sizeof(int32_t), 1, f) != 1 || count <= 0) {
        fprintf(stderr, "bad dump header\n");
        return 2;
    }

    float* x = malloc((size_t)count * 128 * sizeof(float));
    uint8_t* sym = malloc((size_t)count * 128);
    if (fread(x, sizeof(float), (size_t)count * 128, f) != (size_t)count * 128 ||
        fread(sym, 1, (size_t)count * 128, f) != (size_t)count * 128) {
        fprintf(stderr, "dump is short\n");
        return 2;
    }
    fclose(f);

    // Each group was encoded with one codebook or the other. A group counts as
    // reproduced when either choice matches every symbol.
    int matched[2] = { 0, 0 };
    int unmatched = 0;
    long worst_symbols = 0;

    for (int g = 0; g < count; g++) {
        int ok = -1;
        int fewest = 128;
        for (int is_k = 0; is_k < 2; is_k++) {
            tcq3_block blk;
            uint8_t mine[128];
            tcq3_encode_rotated(x + (size_t)g * 128, is_k, 1.0f, mine, &blk);
            int bad = 0;
            for (int t = 0; t < 128; t++) {
                if (mine[t] != sym[(size_t)g * 128 + t]) { bad++; }
            }
            if (bad == 0) { ok = is_k; }
            if (bad < fewest) { fewest = bad; }
        }
        if (ok >= 0) {
            matched[ok]++;
        } else {
            unmatched++;
            worst_symbols += fewest;
        }
    }

    printf("groups %d\n", count);
    printf("bit-exact with the V codebook: %d\n", matched[0]);
    printf("bit-exact with the K codebook: %d\n", matched[1]);
    printf("reproduced by neither: %d (%ld symbols differ at best)\n", unmatched, worst_symbols);

    const int best = count - unmatched;
    free(x);
    free(sym);
    return best == count ? 0 : 1;
}
