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
