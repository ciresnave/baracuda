# KISS #329 — does CUDA preserve a NaN payload through a MOVED min/max? (device measurement)

This directory is the **citable, reproducible substantiation** for a device measurement
answering the non-x86 half of KISS **§6.8-0010(a)**: on an sm_89 CUDA device, does a
`min`/`max` result that is a **MOVED float operand** (a §6.13 select-decomposition, not
an arithmetic combination) preserve the NaN payload — sign, quiet bit, and trailing
payload bits — or canonicalize it?

It exists so the kiss-ref conformance corpus can point its provenance NOTE at a named
ref instead of an assertion. Everything needed to re-run the measurement and re-check
the compiled instruction is committed here.

**Both arms are measured.** `min` and `max` decompose to *different* selects
(`a<=b` vs `a>=b`) and could lower differently — one could fold to a hardware `FMNMX`
while the other stays a select — so each is measured and fold-checked independently. A
`minmax` claim over a `max`-only kernel would be a label wider than its evidence.

## Result

Measured on a physical device (see toolchain below). Bits are raw `__float_as_uint`.

| input `a` (`b` = `1.0`) | `max_prop` (MOVED) | `min_prop` (MOVED) | `a+b` (arith. contrast) |
|-------------------------|--------------------|--------------------|-------------------------|
| sNaN `0x7F801234`       | `0x7F801234` **PRESERVED** | `0x7F801234` **PRESERVED** | `0x7FFFFFFF` canonicalized |
| qNaN `0x7FC01234`       | `0x7FC01234` **PRESERVED** | `0x7FC01234` **PRESERVED** | `0x7FFFFFFF` canonicalized |

Both MOVED results are **byte-identical** to the input NaN — including the **signaling
bit** (`0x7F80…`, quiet bit clear) and the trailing payload `0x1234` — for both the
`max` and the `min` arm. The arithmetic `a + b` alongside is the already-settled
§6.8-0010 *main* rule (a COMPUTED NaN canonicalizes to `0x7FFFFFFF`); it is the contrast
control, not a thing under test.

**Reading:** on this arch/toolchain, a MOVED min/max preserves the payload exactly on
both arms, which is consistent with §6.8-0010(a) — the clause needs no amendment for
CUDA.

## Scope — what this does and does NOT establish

- **Does:** one arch (`sm_89`), one toolchain (CUDA 13.3, `-O3` default), one op shape
  (`max_prop`/`min_prop` as §6.13 selects over float operands), both min and max arms.
  SASS-verified: the instructions that executed are the two select-moves, not folded
  hardware `FMNMX`.
- **Does NOT:** generalize across SM generations, compiler versions, optimization
  levels, or op shapes. It is a point measurement, deliberately narrow. A different
  `-arch`, a newer nvcc, or a differently-written kernel could fold differently and
  must be re-measured, not assumed.

## The fold hazard, and how it is ruled out — for BOTH arms

The trap: nvcc can fold a hand-written select-decomposition into a hardware
`max.f32`/`min.f32` (SASS `FMNMX`) — a *different op* with its own NaN behavior (the
IEEE hardware minmax is NaN-*suppressing*, the opposite of these NaN-*propagating*
selects) — so a "preserved" result could be measuring the wrong instruction. The
control is to prove the emitted instructions are genuine float **select-moves**, at
both ISA levels. Grep the committed dumps yourself:

```
# PTX (virtual ISA): TWO float selects, and NO hardware max/min
grep -nE 'setp\.nan|selp\.f32|max\.f32|min\.f32' minmax_nan_move.ptx
#   setp.nan.f32 %p1, %f2, %f2;   <- NaN test (a)
#   selp.f32     %f5, %f2, %f1, %p4;  <- select-move, arm 1   (no max.f32)
#   setp.nan.f32 %p5, %f1, %f1;   <- NaN test (b)
#   selp.f32     %f7, %f2, %f1, %p6;  <- select-move, arm 2   (no min.f32)

# SASS (real machine ISA on sm_89): two selects, hardware min/max ABSENT
grep -cE 'FSEL'      minmax_nan_move.sass.txt   # -> 2   (one select-move per arm)
grep -cE 'FSETP\.NAN' minmax_nan_move.sass.txt  # -> 2   (see note)
grep -cE 'FMNMX'     minmax_nan_move.sass.txt   # -> 0   (no hardware min/max fold)
```

`FMNMX = 0` is the load-bearing line: neither arm's measured op is a hardware minmax
that would answer a different question. `FSEL = 2` confirms two independent
select-moves, one per arm.

**Note on `FSETP.NAN = 2` (not 4):** both arms perform the *same* two NaN tests
(`a != a`, `b != b`); nvcc computes those two predicates once and reuses them across
both selects (common-subexpression elimination). The NaN test is shared; the
**select is per-arm** (`FSEL = 2`). This is a correct optimization and does not merge
the two measured results — `out[0]` (max) and `out[1]` (min) are produced by distinct
`FSEL` instructions.

## Toolchain (as measured)

- **Device:** `NVIDIA GeForce RTX 4070 Laptop GPU`, compute capability `sm_89` (the
  exact string the runtime reported; cite it, not a rounded model name).
- **CUDA:** compilation tools release 13.3, V13.3.33 (Build `cuda_13.3.r13.3`,
  NVVM 7.0.1, CL-37862127).
- **Compile arch:** `-arch=sm_89`, default optimization (`-O3`).
- **Host compiler:** MSVC 14.51.36231 (Visual Studio 2026), invoked via `-ccbin`.
- **Host OS:** Windows 11.

## Reproduce

Measured on Windows with the MSVC host compiler; the commands below are the ones that
produced these artifacts. `<VS-MSVC>` is the `Hostx64\x64` directory of a Visual Studio
MSVC toolset, e.g.
`C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.51.36231\bin\Hostx64\x64`.

```powershell
# 1. Compile + run the measurement (prints the bit results above).
nvcc -ccbin "<VS-MSVC>" -arch=sm_89 minmax_nan_move.cu -o minmax_nan_move.exe
.\minmax_nan_move.exe

# 2. Regenerate the PTX dump (virtual ISA).
nvcc -ccbin "<VS-MSVC>" -arch=sm_89 -ptx minmax_nan_move.cu -o minmax_nan_move.ptx

# 3. Regenerate the SASS dump (real machine ISA) from the built exe.
cuobjdump -sass minmax_nan_move.exe > minmax_nan_move.sass.txt

# 4. Re-run the fold-check greps above against the fresh dumps.
```

On a non-Windows host, drop `-ccbin` (nvcc finds the system host compiler) and use
`./minmax_nan_move`. A device of a different compute capability requires the matching
`-arch` and re-verification of the SASS — a different arch is a different measurement.

## Files

- `minmax_nan_move.cu` — the measurement kernel + host driver (self-documenting;
  CUDA API return values checked so a failed copy cannot print stale bytes as a result).
- `minmax_nan_move.ptx` — committed PTX dump (virtual ISA; two select-decompositions).
- `minmax_nan_move.sass.txt` — committed SASS dump (real sm_89 machine ISA; `FMNMX=0`,
  `FSEL=2`).
