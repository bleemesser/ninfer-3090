#!/usr/bin/env bash
# Serve Qwen3.8-27B through ninfer-serve at full context, with MTP and vision.
# Reachable on the Tailscale IP and (optionally) 127.0.0.1, never the LAN.
set -euo pipefail

HERE="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
BIN="${BIN:-$HERE/build-sm86/apps/ninfer-serve}"
MODEL="${MODEL:-$HERE/models/qwen3_8_27b.ninfer}"
PORT="${PORT:-8099}"
CTX="${CTX:-262144}"
LOOPBACK="${LOOPBACK:-on}"  # also listen on 127.0.0.1 through a socat forwarder
ALIAS="${ALIAS:-Qwen3.8-27B-Q4_K_M}"  # /v1/models id; matches the id llama-turbo.sh reports

usage() {
    cat <<'EOF'
Usage: serve.sh [-p PORT] [-m MODEL] [-- extra ninfer-serve args]

Serves the full 262144 context with the tcq3 KV cache, MTP speculative
decoding, and vision. That combination needs about 22.3 GiB, which leaves
1.8 GiB free on a 24 GiB card, so close other GPU users first.

tcq3 is the trellis-coded 3.25-bit KV cache ported from the llama.cpp
TurboQuant fork, and it is what makes 262144 context fit. int8 prefills about
2 percent faster at 16384 tokens but runs out of memory well before full
context; pass -- --kv-dtype int8 with a smaller CTX to try it.

Binding: ninfer-serve takes one --host, so it binds the Tailscale IP. With
LOOPBACK=on a socat forwarder adds 127.0.0.1 on the same port. Do not use
--host 0.0.0.0: the FedoraWorkstation zone leaves 1025-65535/tcp open on the
wired interface, which would expose this unauthenticated server to the LAN.

Concurrency is 1. MTP is single-sequence only, and pooled slots share one KV
cache, so several requests that each fit can still overrun it together.

Env overrides: CTX, PORT, MODEL, BIN, ALIAS, APIKEY, LOOPBACK=on|off,
MINFREE, MAXTOKENS
EOF
    exit 0
}

EXTRA=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--port)  PORT="$2"; shift 2 ;;
        -m|--model) MODEL="$2"; shift 2 ;;
        -h|--help)  usage ;;
        --) shift; EXTRA=("$@"); break ;;
        *) echo "unknown argument: $1" >&2; exit 1 ;;
    esac
done

[[ -x "$BIN"   ]] || { echo "server binary missing: $BIN" >&2; exit 1; }
[[ -f "$MODEL" ]] || { echo "artifact not found: $MODEL" >&2; exit 1; }

TSIP="$(tailscale ip -4 2>/dev/null | head -1)"
[[ -n "$TSIP" ]] || { echo "no Tailscale IPv4 -- is tailscaled up?" >&2; exit 1; }

AUTH=()
[[ -n "${APIKEY:-}" ]] && AUTH=(--api-key "$APIKEY")

# tcq3 needs the whole KV cache resident. Warn on a crowded GPU rather than
# fail during warmup.
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
(( FREE < ${MINFREE:-23000} )) && echo "warning: only ${FREE} MiB free, ${MINFREE:-23000} MiB needed -- close other GPU users first" >&2

FWD=""
if [[ "$LOOPBACK" == "on" ]]; then
    if command -v socat >/dev/null; then
        socat "TCP-LISTEN:${PORT},bind=127.0.0.1,fork,reuseaddr" "TCP:${TSIP}:${PORT}" &
        FWD=$!
        # pkill -f would match this script's own cmdline; kill the pid directly.
        trap 'kill $FWD 2>/dev/null || true' EXIT INT TERM
    else
        echo "note: socat not installed, 127.0.0.1 unavailable (use $TSIP)" >&2
    fi
fi

echo "model : $(basename "$MODEL")"
echo "config: ctx=$CTX kv=tcq3 mtp=on vision=on preserve-thinking=on"
echo "url   : http://${TSIP}:${PORT}/v1"
[[ -n "$FWD" ]] && echo "        http://127.0.0.1:${PORT}/v1"
echo

# exec would drop the trap that cleans up socat, so run in the foreground.
# --default-max-tokens is raised from the 8192 default because this model can
# spend a whole 8192 budget on reasoning before it writes an answer.
# --preserve-thinking keeps reasoning from closed turns in later prompts, which
# matches --reasoning-preserve in llama-turbo.sh. Sampler values are left to the
# model defaults, as in llama-turbo.sh.
"$BIN" "$MODEL" \
    --host "$TSIP" \
    --port "$PORT" \
    --model-id "$ALIAS" \
    --max-context "$CTX" \
    --kv-capacity "$CTX" \
    --kv-dtype tcq3 \
    --prefill-chunk 1024 \
    --max-concurrency 1 \
    --spec mtp --draft-tokens 3 --lm-head-draft \
    --vision \
    --preserve-thinking \
    --default-max-tokens "${MAXTOKENS:-32768}" \
    "${AUTH[@]}" "${EXTRA[@]}"
