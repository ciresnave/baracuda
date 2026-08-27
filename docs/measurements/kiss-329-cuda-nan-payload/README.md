# KISS #329 — does CUDA preserve a NaN payload through a MOVED minmax? (device measurement)

This directory is the **citable, reproducible substantiation** for a single device
measurement answering the non-x86 half of KISS **§6.8-0010(a)**: on an sm_89 CUDA
device, does a `min`/`max` result that is a **MOVED float operand** (a §6.13
select-decomposition, not an arithmetic combination) preserve the NaN payload — sign,
quiet bit, and trailing payload bits — or canonicalize it?

It exists so the kiss-ref conformance corpus can point its provenance NOTE at a named
ref instead of an assertion. Everything needed to re-run the measurement and re-check
the compiled instruction is committed here.

## Result

Measured on a physical device (see toolchain below). Bits are raw `__float_as_uint`.

| input `a` (`b` = `1.0`)      | `select-MOVED max_prop(a,b)` | `arithmetic a+b` (contrast) |
|------------------------------|------------------------------|-----------------------------|
| sNaN `0x7F801234`            | `0x7F801234` **PRESERVED**   | `0x7FFFFFFF` canonicalized   |
| qNaN `0x7FC01234`            | `0x7FC01234` **PRESERVED**   | `0x7FFFFFFF` canonicalized   |

The MOVED result is **byte-identical** to the input NaN — including the **signaling
bit** (`0x7F80…`, quiet bit clear) and the trailing payload `0x1234`. The arithmetic
`a + b` alongside is the already-settled §6.8-0010 *main* rule (a COMPUTED NaN
canonicalizes to `0x7FFFFFFF`); it is the contrast control, not the thing under test.

**Reading:** on this arch/toolchain, a MOVED minmax preserves the payload exactly,
which is consistent with §6.8-0010(a) — the clause needs no amendment for CUDA.

## Scope — what this does and does NOT establish

- **Does:** one arch (`sm_89`), one toolchain (CUDA 13.3, `-O3` default), one op shape
  (`max_prop` as a §6.13 select over float operands). SASS-verified: the instruction
  that executed is the select-move, not a folded hardware `max`.
- **Does NOT:** generalize across SM generations, compiler versions, optimization
  levels, or op shapes. It is a point measurement, deliberately narrow. A different
  `-arch`, a newer nvcc, or a differently-written kernel could fold differently and
  must be re-measured, not assumed.

## The fold hazard, and how it is ruled out

The trap: nvcc can fold a hand-written select-decomposition into a hardware
`max.f32` / `FMNMX` — a *different op* with its own NaN behavior — so a "preserved"
result could be measuring the wrong instruction. The control is to prove the emitted
instruction is a genuine float **select-move**, at both ISA levels. Grep the committed
dumps yourself:

```
# PTX (virtual ISA): a NaN-test + a float select, and NO hardware max/min
grep -nE 'setp\.nan|selp\.f32|max\.f32|min\.f32' minmax_nan_move.ptx
#   setp.nan.f32 %p1, %f2, %f2;   <- NaN test
#   setp.nan.f32 %p2, %f1, %f1;   <- NaN test
#   selp.f32     %f6, %f2, %f1, %p3;  <- float select-move   (no max.f32/min.f32)

# SASS (real machine ISA on sm_89): FSEL present, hardware min/max ABSENT
grep -cE 'FSEL'      minmax_nan_move.sass.txt   # -> 1   (the select-move executed)
grep -cE 'FSETP\.NAN' minmax_nan_move.sass.txt  # -> 2   (the NaN predicates)
grep -cE 'FMNMX'     minmax_nan_move.sass.txt   # -> 0   (no hardware min/max fold)
```

`FMNMX = 0` is the load-bearing line: the measured op is the select-move, not a
hardware minmax that would answer a different question.

## Toolchain (as measured)

- **Device:** NVIDIA GeForce RTX 4070, compute capability `sm_89`.
- **CUDA:** compilation tools release 13.3, V13.3.33 (Build `cuda_13.3.r13.3`,
  NVVM 7.0.1, CL-37862127).
- **Compile arch:** `-arch=sm_89`, default optimization (`-O3`).
- **Host compiler:** MSVC (invoked via `-ccbin <VS MSVC Hostx64/x64 dir>`).

## Reproduce

```
# 1. Compile + run the measurement (prints the bit results above).
nvcc -ccbin "<VS MSVC Hostx64/x64 dir>" -arch=sm_89 minmax_nan_move.cu -o minmax_nan_move
./minmax_nan_move

# 2. Regenerate the PTX dump (virtual ISA).
nvcc -ccbin "<...>" -arch=sm_89 -ptx minmax_nan_move.cu -o minmax_nan_move.ptx

# 3. Regenerate the SASS dump (real machine ISA) from the built exe.
cuobjdump -sass minmax_nan_move > minmax_nan_move.sass.txt

# 4. Re-run the fold-check greps above against the fresh dumps.
```

On a non-Windows host drop `-ccbin` (nvcc finds the system host compiler). A device of
a different compute capability requires the matching `-arch` and re-verification of the
SASS — a different arch is a different measurement.

## Files

- `minmax_nan_move.cu` — the measurement kernel + host driver (self-documenting).
- `minmax_nan_move.ptx` — committed PTX dump (virtual ISA; the select-decomposition).
- `minmax_nan_move.sass.txt` — committed SASS dump (real sm_89 machine ISA; `FMNMX=0`).
