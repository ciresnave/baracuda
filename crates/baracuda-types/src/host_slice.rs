//! [`HostSlice`] — generic abstraction over "things that are a slice of `T`".

/// Trait for host-memory values that can be viewed as `&[T]`.
///
/// Safe-API wrappers in baracuda (and downstream crates) use this to accept
/// any of the common host-memory containers (`&[T]`, `Vec<T>`, `Box<[T]>`,
/// fixed-size arrays, and so on) without forcing the caller to write
/// `.as_slice()` explicitly.
///
/// The blanket impl over `AsRef<[T]>` covers every standard collection out
/// of the box. User types can implement it directly if they need custom
/// behavior (e.g. a memory-mapped file that decodes on read).
///
/// # Example
///
/// ```
/// use baracuda_types::HostSlice;
///
/// fn sum<T: core::ops::AddAssign + Default + Copy, S: HostSlice<T>>(slice: S) -> T {
///     let mut acc = T::default();
///     for &v in slice.as_host_slice() {
///         acc += v;
///     }
///     acc
/// }
///
/// assert_eq!(sum(vec![1u32, 2, 3]), 6);
/// assert_eq!(sum([1.0f32, 2.0, 3.0]), 6.0);
/// assert_eq!(sum(&[1i8, 2, 3][..]), 6);
/// ```
pub trait HostSlice<T> {
    /// Return a slice view of the host memory.
    fn as_host_slice(&self) -> &[T];

    /// Number of elements in the slice. Default impl reads the slice length.
    fn len(&self) -> usize {
        self.as_host_slice().len()
    }

    /// `true` if the slice has zero elements.
    fn is_empty(&self) -> bool {
        self.as_host_slice().is_empty()
    }
}

impl<T, S: AsRef<[T]> + ?Sized> HostSlice<T> for S {
    #[inline]
    fn as_host_slice(&self) -> &[T] {
        self.as_ref()
    }
}
