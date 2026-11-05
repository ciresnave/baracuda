//! The seam crates are pinned with a caret requirement, never `=`.
//!
//! Fuel's own workspace carries newer 0.26 patch releases of `fuel-kernel-seam` and
//! `fuel-kernel-seam-types`. An exact pin (`=0.26.12`) cannot unify with them: cargo
//! then holds two copies and `BaracudaSynthesizer` stops implementing Fuel's
//! `Synthesizer` (E0308 at the conversions in `lib.rs`). The compile proof against a
//! newer patch release lives in the PR that introduced this test (a scratch crate
//! patched to 0.26.13); this test keeps the manifest from regressing.

const MANIFEST: &str = include_str!("../Cargo.toml");

fn requirement(dep: &str) -> String {
    MANIFEST
        .lines()
        .find_map(|line| {
            let line = line.trim();
            let rest = line.strip_prefix(dep)?.trim_start().strip_prefix('=')?;
            Some(rest.trim().trim_matches('"').to_owned())
        })
        .unwrap_or_else(|| panic!("no `{dep} = \"...\"` line in Cargo.toml"))
}

/// A caret requirement: an explicit `^` or a bare version, never `=`, `~`, `<`, `>` or `*`.
fn is_caret(req: &str) -> bool {
    let rest = req.strip_prefix('^').unwrap_or(req);
    rest.chars().next().is_some_and(|c| c.is_ascii_digit())
}

#[test]
fn seam_crates_use_caret_requirements() {
    for dep in ["fuel-kernel-seam-types", "fuel-kernel-seam"] {
        let req = requirement(dep);
        assert!(
            is_caret(&req),
            "{dep} must use a caret requirement, found `{req}`"
        );
        assert_eq!(
            req, "^0.43",
            "{dep} must follow Fuel's 0.43 seam line (the PM's pin for alpha.94)"
        );
    }
}

#[test]
fn the_requirement_parser_sees_an_exact_pin() {
    // Control: the helper would catch `=0.43.0` (so the test above is not vacuous).
    let manifest = "fuel-kernel-seam = \"=0.43.0\"";
    let line = manifest.trim();
    let rest = line
        .strip_prefix("fuel-kernel-seam")
        .and_then(|r| r.trim_start().strip_prefix('='))
        .map(|r| r.trim().trim_matches('"').to_owned())
        .unwrap();
    assert_eq!(rest, "=0.43.0");
    assert!(!is_caret(&rest));
    assert!(is_caret("^0.43") && is_caret("0.43.0"));
    assert!(!is_caret("~0.43") && !is_caret("*"));
}

/// `BaracudaSynthesizer` implements the `Synthesizer` trait of the `fuel-kernel-seam` this
/// graph resolves. With two copies of the seam crates (the exact-pin failure) the bound
/// below would not hold, so this also fails to compile in that situation.
#[test]
fn baracuda_synthesizer_is_fuels_synthesizer() {
    fn takes_fuels_synthesizer<S: fuel_kernel_seam::Synthesizer>(_: &S) {}
    let synthesizer = baracuda_cuda_seam::BaracudaSynthesizer::new(1000);
    takes_fuels_synthesizer(&synthesizer);
}
