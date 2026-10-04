//! The NVRTC on-demand JIT compiler — the production source → PTX
//! [`Compiler`](unpopped::Compiler). Carved out of
//! `baracuda-kernelgen`'s `jit.rs` (carve step 2). The whole module is behind
//! `--features nvrtc` (wired in `lib.rs`), so the per-item `#[cfg]` the source
//! carried is dropped here.

use std::fmt;

use unpopped::{ArtifactKind, Compiler};
use unpopped_vocab::TargetId;

/// Why [`NvrtcCompiler::new`] declined a [`TargetId`] — board #106 option C:
/// `TargetId` is the open architecture identity; `NvrtcCompiler` is the one
/// baracuda type that turns a target into a real `--gpu-architecture=` flag,
/// so it is the one place that must validate the `cuda:` namespace's own
/// grammar (cuda.md §2) on top of [`TargetId::parse`]'s already-enforced
/// generic KISS-Classify §6.8-0001/§6.8-0005 grammar. A decline here is a
/// typed value, never a panic — an unreachable architecture is a configuration
/// fact, not a bug in this process.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum NvrtcTargetError {
    /// The token's namespace isn't `cuda:` — e.g. `vulkan:sg64...` reaching a
    /// CUDA-only compiler. Carries the namespace actually found.
    WrongNamespace(String),
    /// The namespace is `cuda:`, but the capability-set doesn't match cuda.md
    /// §2's `sm<digits>[<letter>]` grammar (baracuda's own namespace rule —
    /// `TargetId::parse` only enforces the KISS-generic token grammar, never a
    /// namespace's own vocabulary, by design). Carries the capability-set.
    MalformedCapabilitySet(String),
}

impl fmt::Display for NvrtcTargetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongNamespace(ns) => {
                write!(
                    f,
                    "NvrtcCompiler only targets the `cuda:` namespace, got `{ns}:`"
                )
            }
            Self::MalformedCapabilitySet(cap) => write!(
                f,
                "`cuda:{cap}` does not match the cuda.md §2 grammar `sm<digits>[<letter>]`"
            ),
        }
    }
}

impl std::error::Error for NvrtcTargetError {}

/// The production on-demand compiler: nvrtc source → PTX. Feature-gated
/// (`--features nvrtc`) because it needs the nvrtc runtime; constructed per
/// target (the `--gpu-architecture` flag the schedule was keyed for).
///
/// Takes a [`TargetId`], not an `ArchSku` — board #106 option C. `ArchSku`
/// stays closed and alive only as `baracuda-cutlass`'s dispatch-SKU key; it is
/// not the architecture identity any more, and this compiler never reads it.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct NvrtcCompiler {
    target: TargetId,
}

impl NvrtcCompiler {
    /// A compiler targeting `target`. Declines (does not panic) a target
    /// outside the `cuda:` namespace, or a `cuda:` token whose capability-set
    /// doesn't match the `sm<digits>[<letter>]` grammar — both are
    /// configuration facts about an unreachable architecture, not failures of
    /// this call. `TargetId` itself already guarantees the generic
    /// KISS-Classify §6.8-0001 (exactly one `:`, non-empty parts) and
    /// §6.8-0005 (ASCII, no `structure_key` separators/whitespace/control)
    /// grammar by construction — a `TargetId` cannot exist without already
    /// satisfying those, so this constructor only adds the `cuda:`-specific
    /// check neither of those clauses can know about. The §6.4-0004 4096-byte
    /// cap bounds a *full* `structure_key` token, not a bare `TargetId` in
    /// isolation (a `cuda:sm<N>[<letter>]` token is a handful of bytes), so it
    /// is not re-checked here.
    ///
    /// # Errors
    ///
    /// [`NvrtcTargetError`] naming which cuda.md §2 rule the target fails.
    pub fn new(target: TargetId) -> Result<Self, NvrtcTargetError> {
        arch_flag(target)?;
        Ok(Self { target })
    }
}

impl Compiler for NvrtcCompiler {
    fn compile(&self, source: &str, entry: &str, _max_compile_ms: u32) -> Result<Vec<u8>, String> {
        // nvrtc has no compile-deadline API; `max_compile_ms` gates optimization
        // depth / the inward e-graph's iteration count at a coarser grain (future).
        // Use the low-level path so a compilation error surfaces the nvrtc log.
        use baracuda_nvrtc::Program;
        let name = format!("{entry}.cu");
        let prog =
            Program::new(source, &name).map_err(|e| format!("nvrtc({entry}) create: {e}"))?;
        // `self.target` was already validated in `new` (`arch_flag` cannot
        // fail here), so unwrap is a restatement of that invariant, not a new
        // panic risk.
        let arch = format!(
            "--gpu-architecture={}",
            arch_flag(self.target).expect("validated in NvrtcCompiler::new")
        );
        let mut opts = vec![arch];
        // fp16/bf16 kernels `#include <cuda_fp16.h>`/`<cuda_bf16.h>`; headerless
        // nvrtc has no default search path, so point it at the CUDA include dir
        // (env-detected) — without this, f16/bf16 JIT fails to find the header even
        // though the AOT (nvcc) path compiles. Harmless for header-light f32 source.
        if let Some(inc) = cuda_include_dir() {
            opts.push(format!("-I{inc}"));
        }
        let opt_refs: Vec<&str> = opts.iter().map(String::as_str).collect();
        match prog.compile_raw(&opt_refs) {
            Ok(()) => prog
                .ptx()
                .map(String::into_bytes)
                .map_err(|e| format!("nvrtc({entry}) ptx: {e}")),
            Err(e) => {
                let log = prog.log().unwrap_or_default();
                Err(format!(
                    "nvrtc({entry}): {e}\n--- nvrtc log ---\n{}",
                    log.trim()
                ))
            }
        }
    }
    fn artifact_kind(&self) -> ArtifactKind {
        ArtifactKind::Ptx
    }
}

/// The CUDA toolkit `include/` directory (for nvrtc's `-I`), detected from the
/// usual environment vars. `None` if unset/missing — header-light (f32/f64/int)
/// kernels still compile; only the fp16/bf16 headers need it.
fn cuda_include_dir() -> Option<String> {
    for var in ["CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"] {
        if let Ok(root) = std::env::var(var) {
            let inc = std::path::Path::new(&root).join("include");
            if inc.is_dir() {
                return Some(inc.to_string_lossy().into_owned());
            }
        }
    }
    None
}

/// `--gpu-architecture` flag for a `cuda:sm<digits>[<letter>]` [`TargetId`] —
/// generic over the whole open token space (board #106 Step B), not a closed
/// per-`ArchSku` match. `cuda:sm80` → `"sm_80"`; `cuda:sm90a` → `"sm_90a"`;
/// `cuda:sm61`/`cuda:sm121`/anything matching the grammar works the same way,
/// with no per-architecture arm to add as the vocabulary grows — that growth
/// is exactly what board #106 moved out of this crate's closed enum.
///
/// # Errors
///
/// [`NvrtcTargetError::WrongNamespace`] if `target`'s namespace isn't `cuda`.
/// [`NvrtcTargetError::MalformedCapabilitySet`] if the capability-set isn't
/// `sm` followed by one or more ASCII digits and an optional single ASCII
/// lowercase letter (cuda.md §2) — the only part of this grammar
/// `TargetId::parse` leaves to the namespace owner (module docs on
/// [`TargetId`]: "this crate ... validates the grammar and never the
/// vocabulary").
fn arch_flag(target: TargetId) -> Result<String, NvrtcTargetError> {
    let ns = target.namespace();
    if ns != "cuda" {
        return Err(NvrtcTargetError::WrongNamespace(ns));
    }
    let cap = target.capability_set();
    let digits_and_letter = cap
        .strip_prefix("sm")
        .ok_or_else(|| NvrtcTargetError::MalformedCapabilitySet(cap.clone()))?;
    let digits_end = digits_and_letter
        .bytes()
        .take_while(u8::is_ascii_digit)
        .count();
    let (digits, letter) = digits_and_letter.split_at(digits_end);
    let letter_ok = letter.is_empty()
        || (letter.len() == 1
            && letter
                .bytes()
                .next()
                .is_some_and(|b| b.is_ascii_lowercase()));
    if digits.is_empty() || !letter_ok {
        return Err(NvrtcTargetError::MalformedCapabilitySet(cap));
    }
    Ok(format!("sm_{digits_and_letter}"))
}

#[cfg(test)]
mod tests {
    use super::{NvrtcCompiler, NvrtcTargetError, arch_flag};
    use unpopped_vocab::{ArchSku, TargetId};

    /// Board #106 Step B boundary-condition test 1: every `ArchSku` variant
    /// round-trips through `From<ArchSku> for TargetId` (already verified
    /// elsewhere against unpopped-vocab 0.14.3) to the exact
    /// `--gpu-architecture=` string `NvrtcCompiler::new` would configure —
    /// the generic `arch_flag` derivation produces byte-identical output to
    /// the old closed `ArchSku` match it replaces, for all 4 variants the
    /// closed enum ever had. Also exercises `NvrtcCompiler::new` itself (not
    /// just the free function), confirming it accepts all 4 and the
    /// constructed compiler carries the expected target.
    #[test]
    fn target_id_from_arch_sku_matches_the_cuda_token_rule() {
        for (sku, token, flag) in [
            (ArchSku::Sm80, "cuda:sm80", "sm_80"),
            (ArchSku::Sm89, "cuda:sm89", "sm_89"),
            (ArchSku::Sm90, "cuda:sm90", "sm_90"),
            (ArchSku::Sm90a, "cuda:sm90a", "sm_90a"),
        ] {
            let got: TargetId = sku.into();
            assert_eq!(got.as_str(), token, "{sku:?} -> TargetId token");
            assert_eq!(
                arch_flag(got),
                Ok(flag.to_string()),
                "{sku:?} -> --gpu-architecture= flag"
            );
            assert!(
                NvrtcCompiler::new(got).is_ok(),
                "{sku:?}: NvrtcCompiler::new must accept every ArchSku-derived target"
            );
        }
    }

    /// Board #106 Step B boundary-condition test 2: `NvrtcCompiler::new`
    /// accepts NVRTC-only targets board #106 opened up — architectures with
    /// no `ArchSku` variant at all (sm61 Pascal is CireSnave's stated
    /// priority; sm121 is Unpopped's newest-sourced capability row). Proves
    /// the open-token path is real, not just type-accepted: these previously
    /// had NO way to reach `NvrtcCompiler` because `ArchSku` had no variant
    /// for them.
    #[test]
    fn nvrtc_only_targets_with_no_archsku_variant_are_accepted() {
        for (token, flag) in [
            ("cuda:sm61", "sm_61"),
            ("cuda:sm70", "sm_70"),
            ("cuda:sm75", "sm_75"),
            ("cuda:sm86", "sm_86"),
            ("cuda:sm100", "sm_100"),
            ("cuda:sm100a", "sm_100a"),
            ("cuda:sm120", "sm_120"),
            ("cuda:sm120a", "sm_120a"),
            ("cuda:sm121", "sm_121"),
        ] {
            let target = TargetId::parse(token).expect("a valid cuda: token must intern");
            assert_eq!(
                arch_flag(target),
                Ok(flag.to_string()),
                "{token} -> --gpu-architecture= flag"
            );
            assert!(
                NvrtcCompiler::new(target).is_ok(),
                "{token}: NvrtcCompiler::new must accept a well-formed NVRTC-only target"
            );
        }
    }

    /// Board #106 Step B boundary-condition test 3: an invalid/unknown
    /// `TargetId` DECLINES from `NvrtcCompiler::new` — never a panic. Covers
    /// both error arms: a non-`cuda:` namespace (a real, well-formed
    /// `TargetId` under KISS-Classify §6.8-0001/§6.8-0005, just not ours),
    /// and a `cuda:` token whose capability-set doesn't match cuda.md §2's
    /// `sm<digits>[<letter>]` grammar (a malformed vocabulary word, which
    /// `TargetId::parse` itself cannot catch — module docs: "this crate ...
    /// validates the grammar and never the vocabulary").
    #[test]
    fn invalid_target_declines_never_panics() {
        let wrong_namespace =
            TargetId::parse("vulkan:sg64.ops-abr.arith-f16-i8.cm-none").expect("valid token");
        assert_eq!(
            NvrtcCompiler::new(wrong_namespace),
            Err(NvrtcTargetError::WrongNamespace("vulkan".to_string()))
        );

        for bad_cap_token in [
            "cuda:not-an-sm-token",
            "cuda:sm",
            "cuda:smAB",
            "cuda:sm90A",  // uppercase letter: cuda.md §2 is lowercase-only
            "cuda:sm90ab", // more than one letter after the digits
        ] {
            let target = TargetId::parse(bad_cap_token).expect("KISS-generic grammar is valid");
            let cap = target.capability_set();
            assert_eq!(
                NvrtcCompiler::new(target),
                Err(NvrtcTargetError::MalformedCapabilitySet(cap)),
                "{bad_cap_token} must decline as a malformed capability-set, not panic"
            );
        }
    }
}
