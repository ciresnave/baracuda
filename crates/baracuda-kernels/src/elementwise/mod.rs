//! Elementwise op family — unified plan-based API.
//!
//! Phase 3 of the baracuda-kernels comprehensive plan: every PyTorch
//! `torch.<op>` and JAX `jax.numpy.<op>` / `jax.lax.<op>` for elementwise
//! / shape / layout primitives lives here.
//!
//! Today this module hosts the trailblazer dispatcher [`BinaryPlan`] —
//! one Plan type per op category (unary / binary / ternary / gated /
//! shape-layout), each parameterized on the scalar element type `T` and
//! the tensor rank `N`. The op identity within a category is a runtime
//! enum field on the descriptor (e.g. [`BinaryKind`]), dispatching to
//! per-(op, dtype) kernel SKUs in `baracuda-kernels-sys` — exactly the
//! shape used by the GEMM family's [`crate::EpilogueKind`].
//!
//! Future modules: `unary` (categories B + B'), `ternary` (category D),
//! `gated` (category C'), `shape_layout` (category N).

pub mod affine;
pub mod binary;
pub mod binary_backward;
pub mod binary_cmp;
pub mod binary_param;
pub mod binary_param_backward;
pub mod cast;
pub mod cast_subbyte;
pub mod gated_activation;
pub mod gated_activation_backward;
pub mod prelu;
pub mod prelu_backward;
pub mod ternary;
pub mod ternary_backward;
pub mod unary;
pub mod unary_backward;
pub mod unary_param;
pub mod unary_param_backward;
pub mod where_backward;
pub mod where_op;

pub use affine::{AffineArgs, AffineDescriptor, AffinePlan};
pub use binary::{BinaryArgs, BinaryDescriptor, BinaryPlan};
pub use cast::{CastArgs, CastDescriptor, CastPlan};
pub use cast_subbyte::{CastSubByteArgs, CastSubByteDescriptor, CastSubBytePlan};
pub use binary_backward::{BinaryBackwardArgs, BinaryBackwardDescriptor, BinaryBackwardPlan};
pub use binary_cmp::{BinaryCmpArgs, BinaryCmpDescriptor, BinaryCmpPlan};
pub use binary_param::{BinaryParamArgs, BinaryParamDescriptor, BinaryParamPlan};
pub use binary_param_backward::{
    BinaryParamBackwardArgs, BinaryParamBackwardDescriptor, BinaryParamBackwardPlan,
};
pub use gated_activation::{
    GatedActivationArgs, GatedActivationDescriptor, GatedActivationPlan,
};
pub use gated_activation_backward::{
    GatedActivationBackwardArgs, GatedActivationBackwardDescriptor, GatedActivationBackwardPlan,
};
pub use prelu::{PReluArgs, PReluDescriptor, PReluPlan};
pub use prelu_backward::{PReluBackwardArgs, PReluBackwardDescriptor, PReluBackwardPlan};
pub use ternary::{TernaryArgs, TernaryDescriptor, TernaryPlan};
pub use ternary_backward::{
    TernaryBackwardArgs, TernaryBackwardDescriptor, TernaryBackwardPlan,
};
pub use unary::{UnaryArgs, UnaryDescriptor, UnaryPlan};
pub use unary_backward::{UnaryBackwardArgs, UnaryBackwardDescriptor, UnaryBackwardPlan};
pub use unary_param::{UnaryParamArgs, UnaryParamDescriptor, UnaryParamPlan};
pub use unary_param_backward::{
    UnaryParamBackwardArgs, UnaryParamBackwardDescriptor, UnaryParamBackwardPlan,
};
pub use where_backward::{WhereBackwardArgs, WhereBackwardDescriptor, WhereBackwardPlan};
pub use where_op::{WhereArgs, WhereDescriptor, WherePlan};
