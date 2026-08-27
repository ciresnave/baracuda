#!/usr/bin/env bash
# PTX-lever / occupancy survey over ship-exact kernels-sys objects.
# Reports PRESENT and ABSENT per kernel; counts, never "the kernels don't use it".
#
# NOTE the read-only-cache pattern is `.CONSTANT|.CI`, NOT just `.CI`. On this
# toolchain's SASS the read-only/non-coherent global load (PTX `ld.global.nc`) is
# spelled `LDG.E.CONSTANT`; a `.CI`-only grep returns a FALSE 0 (see README — the
# instrument error this survey survived). Verify the load notation on real data
# (`cuobjdump -sass | grep -oE 'LDG[.A-Z0-9]*' | sort | uniq -c`) before trusting any
# lever count.
set -uo pipefail

# Pick the most-populated ship-exact object dir (built with the shipping flags).
DIR=$(for d in target/debug/build/baracuda-kernels-sys-*/out; do
        [ -d "$d" ] && echo "$(find "$d" -maxdepth 1 -name '*.o' | wc -l) $d"
      done | sort -rn | head -1 | awk '{print $2}')
if [ -z "$DIR" ]; then
  echo "no baracuda-kernels-sys build found under target/ — build it first:" >&2
  echo "  cargo build -p baracuda-kernels-sys" >&2
  exit 1
fi
echo "object dir: $DIR"
echo

for key_glob in \
  "elementwise:binary_add:binary_add_fp-*.o" \
  "reduction:softmax:gumbel_softmax_fp-*.o" \
  "reduction:rms_norm:rms_norm_fp-*.o" \
  "gemm:int8_tiled:gemm_s8_rrr_sm80-*.o" \
  "gemv:memory_bound:gemv_dense-*.o" \
  "attention:flash_bwd:flash_bwd_hdim128_bf16_causal_sm80-*.o" \
  "attention:arbmask:attn_arbmask_fp-*.o" \
  "quant:dequantize:dequantize-*.o" \
  "sort:topk:topk-*.o"; do
  key="${key_glob%:*}"; glob="${key_glob##*:}"
  obj=$(find "$DIR" -maxdepth 1 -name "$glob" 2>/dev/null | head -1)
  echo "======================================================================"
  if [ -z "$obj" ]; then echo "$key -> (no object matching $glob)"; continue; fi
  echo "$key -> $(basename "$obj")"
  sass=$(cuobjdump -sass "$obj" 2>/dev/null)
  res=$(cuobjdump -res-usage "$obj" 2>/dev/null)
  c(){ printf '%s' "$sass" | grep -cE "$1"; }
  m(){ printf '%s' "$res" | grep -oE "$1:[0-9]+" | grep -oE '[0-9]+' | sort -rn | head -1; }
  echo "  OCCUPANCY worst: REG=$(m REG) STACK=$(m STACK) SHARED=$(m SHARED) LOCAL=$(m LOCAL)"
  echo "  read-only cache (.CONSTANT|.CI): $(c '\bLDG\.[A-Z0-9.]*(CONSTANT|CI)\b')   of LDG total $(c '\bLDG')"
  echo "  vector: LDG.128=$(c 'LDG\.E\.128')  LDG.64=$(c 'LDG\.E\.64')  STG.128=$(c 'STG\.E\.128')  LDGSTS.128=$(c 'LDGSTS[.A-Z0-9]*\.128')"
  echo "  cp.async LDGSTS=$(c '\bLDGSTS')   tensor HMMA=$(c '\bHMMA') IMMA=$(c '\bIMMA')"
  echo "  local traffic LDL=$(c '\bLDL') STL=$(c '\bSTL')   predication @P=$(c '@!?P[0-9]') BRA=$(c '\bBRA\b')"
done
echo "======================================================================"
echo "DONE — see README.md for the analysis and the .CI/.CONSTANT correction."
