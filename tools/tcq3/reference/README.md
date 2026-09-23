# turbo3_tcq CPU reference

Step 1 of the NInfer port. This is a scalar C implementation of the codec, and
it reproduces the fork's CUDA encoder bit for bit. Every kernel written for the
port is checked against it.

## Files

`extract-constants.py` reads the two 512-entry codebooks and the two
Walsh-Hadamard sign tables out of the fork and writes `tcq3_constants.h`. The
tables are fitted values, so read them rather than retype them.

`tcq3_reference.h` holds the codec: the transform and its inverse, the Viterbi
encoder, the bitstream packer, and the random-access decoder.

`test-tcq3.c` checks the transform, the bitstream layout, and round-trip error.

`validate-against-fork.c` replays a capture from the fork's own encoder.

## Results

The transform is orthonormal to 5e-8 and its inverse is exact to 2.4e-7. All
128 trellis states in a packed block decode to the states the encoder chose.

Round-trip error on unit Gaussian input, relative to the input standard
deviation: TCQ 14.35 percent at 3.25 bits for each value, against 18.44 percent
for an equal-rate scalar quantizer with the same transform at 3.125 bits. The
trellis buys about a fifth less error at the same storage, which is why
turbo3_tcq beats turbo4 in the perplexity measurement despite using fewer bits.

Against the fork: 4096 of 4096 captured groups reproduce every symbol exactly.

## Reproducing the capture

Run the fork's perplexity tool with the dump variable set. The tool writes
/tmp/tcq_errors.bin at exit.

    cd /home/blee/code/bridge-bench
    TURBO_TCQ_DUMP_ERRORS=4096 ./build-tcq/bin/llama-perplexity \
        -m ~/.lmstudio/models/lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf \
        -f data/wikitext-2-raw/wiki.test.raw -c 512 --chunks 1 -ngl 99 -fa on \
        --cache-type-k turbo3_tcq --cache-type-v turbo3_tcq

Then build and run the validator:

    gcc -O2 -o validate validate-against-fork.c -lm && ./validate

The dump records the group after the transform and the 128 symbols the Viterbi
chose. It does not record whether the group was K or V, so the validator tries
both codebooks and counts a group as reproduced when either matches.

## Alpha, and why it is not what the source says

The fork compiles an encode-time V alpha of 1.04 but overrides it to 1.0 at
startup, then applies 1.02 at decode time instead. Only
TURBO_TCQ_ENCODE_ALPHA switches that around. The deployed configuration is
encode 1.0 for K and V, decode 1.0 for K and 1.02 for V. The measured
perplexity of 5.8327 used the deployed configuration, so the port has to match
it and not the compiled default.

## Not yet covered

The per-channel mean subtraction and the per-channel scale are calibration taps
that default to off. If a calibrated run is what produced 5.8327, these have to
be ported too. Confirm before the quality comparison.
