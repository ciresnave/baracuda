//! Proc-macros for `baracuda-types`: `#[derive(DeviceRepr)]`.
//!
//! `#[derive(KernelArg)]` is deliberately *not* provided: `KernelArg` is
//! already implemented for `&T where T: DeviceRepr` via a blanket impl, so
//! deriving `DeviceRepr` is sufficient for a type to be usable as a
//! kernel argument via `&my_value`.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{parse_macro_input, Data, DeriveInput, Fields, Meta};

/// `#[derive(DeviceRepr)]` — implement `baracuda_types::DeviceRepr` for a
/// `#[repr(C)]` or `#[repr(transparent)]` struct whose fields are all
/// `DeviceRepr`. Enums and unions are rejected (use a `#[repr(C)]` struct).
#[proc_macro_derive(DeviceRepr)]
pub fn derive_device_repr(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match expand_device_repr(input) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

fn expand_device_repr(input: DeriveInput) -> syn::Result<TokenStream2> {
    let name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    ensure_repr_c_or_transparent(&input)?;
    let field_types = collect_field_types(&input)?;

    // Augment the where-clause so every field type must also be DeviceRepr.
    // This means if a user forgets to implement the trait on an inner field,
    // the compile error points here rather than at a remote launch site.
    let mut where_clause = where_clause.cloned().unwrap_or_else(|| syn::WhereClause {
        where_token: Default::default(),
        predicates: syn::punctuated::Punctuated::new(),
    });
    for ty in &field_types {
        where_clause
            .predicates
            .push(syn::parse_quote!(#ty: ::baracuda_types::DeviceRepr));
    }

    Ok(quote! {
        // SAFETY: the `#[repr(C)]` / `#[repr(transparent)]` attribute is
        // enforced by the derive, and every field is required by the
        // where-clause to implement DeviceRepr.
        unsafe impl #impl_generics ::baracuda_types::DeviceRepr for #name #ty_generics #where_clause {}
    })
}

fn ensure_repr_c_or_transparent(input: &DeriveInput) -> syn::Result<()> {
    let mut has_required_repr = false;
    for attr in &input.attrs {
        if !attr.path().is_ident("repr") {
            continue;
        }
        if let Meta::List(list) = &attr.meta {
            for tok in list.tokens.clone() {
                if let proc_macro2::TokenTree::Ident(id) = tok {
                    let s = id.to_string();
                    if s == "C" || s == "transparent" {
                        has_required_repr = true;
                    }
                }
            }
        }
    }
    if has_required_repr {
        Ok(())
    } else {
        Err(syn::Error::new_spanned(
            &input.ident,
            "#[derive(DeviceRepr)] requires #[repr(C)] or #[repr(transparent)] on the type",
        ))
    }
}

fn collect_field_types(input: &DeriveInput) -> syn::Result<Vec<syn::Type>> {
    match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(named) => Ok(named.named.iter().map(|f| f.ty.clone()).collect()),
            Fields::Unnamed(unnamed) => Ok(unnamed.unnamed.iter().map(|f| f.ty.clone()).collect()),
            Fields::Unit => Ok(Vec::new()),
        },
        Data::Enum(_) => Err(syn::Error::new_spanned(
            &input.ident,
            "#[derive(DeviceRepr)] on enums is not supported; use a #[repr(C)] struct instead",
        )),
        Data::Union(_) => Err(syn::Error::new_spanned(
            &input.ident,
            "#[derive(DeviceRepr)] on unions is not supported",
        )),
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------
//
// Proc-macro crates (`proc-macro = true`) cannot exercise their macro via
// doc-tests because the crate isn't usable as a normal `extern crate` in
// the doc-test build environment. The integration test file
// `tests/derive_device_repr.rs` drives the *positive* path end-to-end via
// the parent `baracuda-types` crate (dev-dep). Here we cover the
// *rejection* paths by exercising the helper functions directly against
// parsed `syn::DeriveInput` trees — no compiler driver round-trip needed.
#[cfg(test)]
mod tests {
    use super::*;
    use syn::parse_quote;

    #[test]
    fn accepts_repr_c_struct() {
        let input: DeriveInput = parse_quote! {
            #[repr(C)]
            struct S { a: u32, b: f32 }
        };
        ensure_repr_c_or_transparent(&input).expect("repr(C) must be accepted");
        let fields = collect_field_types(&input).unwrap();
        assert_eq!(fields.len(), 2);
    }

    #[test]
    fn accepts_repr_transparent_newtype() {
        let input: DeriveInput = parse_quote! {
            #[repr(transparent)]
            struct N(u64);
        };
        ensure_repr_c_or_transparent(&input).expect("repr(transparent) must be accepted");
        let fields = collect_field_types(&input).unwrap();
        assert_eq!(fields.len(), 1);
    }

    #[test]
    fn accepts_repr_c_with_align() {
        let input: DeriveInput = parse_quote! {
            #[repr(C, align(16))]
            struct A { x: f32 }
        };
        ensure_repr_c_or_transparent(&input).expect("repr(C, align(N)) must still pass");
    }

    #[test]
    fn rejects_missing_repr() {
        let input: DeriveInput = parse_quote! {
            struct S { a: u32 }
        };
        let err = ensure_repr_c_or_transparent(&input).expect_err("missing repr must error");
        let msg = err.to_string();
        assert!(
            msg.contains("repr(C)") || msg.contains("repr(transparent)"),
            "error should mention required reprs: {msg}"
        );
    }

    #[test]
    fn rejects_repr_rust() {
        // `#[repr(Rust)]` is not legal syntax to spell explicitly, but
        // `#[repr(packed)]` alone (without C) is — and must be rejected.
        let input: DeriveInput = parse_quote! {
            #[repr(packed)]
            struct S { a: u32 }
        };
        ensure_repr_c_or_transparent(&input)
            .expect_err("repr(packed) alone (no C) must error");
    }

    #[test]
    fn rejects_repr_int_only() {
        // A bare `#[repr(u32)]` is fine on enums but not what DeviceRepr wants.
        let input: DeriveInput = parse_quote! {
            #[repr(u32)]
            struct S { a: u32 }
        };
        ensure_repr_c_or_transparent(&input).expect_err("repr(u32) alone must error");
    }

    #[test]
    fn rejects_enum_even_with_repr_c() {
        // Even with `#[repr(C)]`, an enum is not a valid DeviceRepr shape
        // (we want it to live in a `#[repr(C)]` struct field). The
        // `collect_field_types` helper enforces this.
        let input: DeriveInput = parse_quote! {
            #[repr(C)]
            enum E { A, B }
        };
        // ensure_repr_c_or_transparent permits the attribute…
        ensure_repr_c_or_transparent(&input).expect("repr(C) attr alone passes that check");
        // …but the data-shape check on the enum body rejects.
        // `.err().expect(..)` rather than `.expect_err(..)`: the latter requires
        // the Ok type (`Vec<syn::Type>`) to be `Debug`, which needs syn's
        // `extra-traits` feature (only present here by cross-crate feature
        // unification in a full-workspace build, so `-p` checks failed).
        let err = collect_field_types(&input).err().expect("enum body must be rejected");
        assert!(err.to_string().contains("enums"), "msg: {}", err);
    }

    #[test]
    fn rejects_union() {
        let input: DeriveInput = parse_quote! {
            #[repr(C)]
            union U { a: u32, b: f32 }
        };
        let err = collect_field_types(&input).err().expect("union body must be rejected");
        assert!(err.to_string().contains("unions"), "msg: {}", err);
    }

    #[test]
    fn unit_struct_collects_zero_fields() {
        let input: DeriveInput = parse_quote! {
            #[repr(C)]
            struct Empty;
        };
        let fields = collect_field_types(&input).unwrap();
        assert!(fields.is_empty());
    }

    #[test]
    fn tuple_struct_collects_positional_fields() {
        let input: DeriveInput = parse_quote! {
            #[repr(C)]
            struct T(f32, u32, i16);
        };
        let fields = collect_field_types(&input).unwrap();
        assert_eq!(fields.len(), 3);
    }

    #[test]
    fn end_to_end_expand_emits_unsafe_impl() {
        // Smoke-check the top-level expander: produces a TokenStream
        // that mentions both `unsafe impl` and `DeviceRepr`, and
        // includes a where-clause predicate per field type.
        let input: DeriveInput = parse_quote! {
            #[repr(C)]
            struct S { a: u32, b: f32 }
        };
        let ts = expand_device_repr(input).expect("valid input must expand cleanly");
        let s = ts.to_string();
        assert!(s.contains("unsafe impl"), "missing unsafe impl: {s}");
        assert!(s.contains("DeviceRepr"), "missing trait name: {s}");
        // Per-field where-clause predicates land verbatim:
        assert!(s.contains("u32"), "missing field type in where-clause: {s}");
        assert!(s.contains("f32"), "missing field type in where-clause: {s}");
    }

    #[test]
    fn end_to_end_expand_rejects_enum() {
        let input: DeriveInput = parse_quote! {
            enum E { A, B }
        };
        // Both checks fire here (no repr AND enum body); either is a
        // legitimate rejection — we only care that the expander errors.
        expand_device_repr(input).expect_err("enum without repr must not expand");
    }
}
