//! Thin dynamic-loader wrapper around `libloading`.
//!
//! This crate does not know about any particular NVIDIA library; each `-sys`
//! crate instantiates a [`Library`] with its own candidate filenames and
//! symbol-resolution strategy. The Driver API in particular layers
//! `cuGetProcAddress`-based symbol resolution on top of [`Library::symbol`];
//! everything else (cudart, cublas, ...) can call `symbol` directly.

use std::ffi::CStr;
use std::path::{Path, PathBuf};

use crate::error::LoaderError;
use crate::platform;

/// Dynamically-loaded NVIDIA library (wraps [`libloading::Library`]).
pub struct Library {
    name: &'static str,
    lib: libloading::Library,
    /// Records the path the library actually resolved from, for diagnostics.
    resolved_from: Option<PathBuf>,
}

impl std::fmt::Debug for Library {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Library")
            .field("name", &self.name)
            .field("resolved_from", &self.resolved_from)
            .finish_non_exhaustive()
    }
}

impl Library {
    /// Open `candidates[0]`, `candidates[1]`, ... in order, falling back to
    /// each path returned by [`platform::library_search_paths`]. Returns
    /// the first success or [`LoaderError::LibraryNotFound`] / platform
    /// error.
    pub fn open(name: &'static str, candidates: &[&'static str]) -> Result<Self, LoaderError> {
        if matches!(platform::os_family(), platform::OsFamily::Unsupported) {
            return Err(LoaderError::UnsupportedPlatform {
                platform: std::env::consts::OS,
            });
        }
        if candidates.is_empty() {
            return Err(LoaderError::library_not_found(name, candidates));
        }

        // Phase 1: try each candidate name bare (OS handles the search).
        for candidate in candidates {
            if let Ok(lib) = unsafe { libloading::Library::new(candidate) } {
                return Ok(Self {
                    name,
                    lib,
                    resolved_from: Some(PathBuf::from(candidate)),
                });
            }
        }

        // Phase 2: try each candidate inside each explicit search directory.
        let search_paths = platform::library_search_paths();
        for dir in &search_paths {
            for candidate in candidates {
                let full = dir.join(candidate);
                if let Ok(lib) = unsafe { libloading::Library::new(&full) } {
                    return Ok(Self {
                        name,
                        lib,
                        resolved_from: Some(full),
                    });
                }
            }
        }

        Err(LoaderError::library_not_found_with_search(
            name,
            candidates,
            search_paths.len(),
        ))
    }

    /// Open a library at the specific path `path` (no search). Mostly used
    /// in tests to inject a known library location.
    pub fn open_at(name: &'static str, path: &Path) -> Result<Self, LoaderError> {
        let lib = unsafe { libloading::Library::new(path) }?;
        Ok(Self {
            name,
            lib,
            resolved_from: Some(path.to_path_buf()),
        })
    }

    /// The logical library name baracuda knows it by (e.g. `"cuda-driver"`,
    /// `"cublas"`).
    #[inline]
    pub fn name(&self) -> &'static str {
        self.name
    }

    /// The absolute path the library actually resolved from, if known.
    #[inline]
    pub fn resolved_from(&self) -> Option<&Path> {
        self.resolved_from.as_deref()
    }

    /// Resolve `symbol`. The caller is responsible for the type `T` matching
    /// the C signature of the symbol; consequently, this function is `unsafe`.
    ///
    /// # Errors
    ///
    /// [`LoaderError::SymbolNotFound`] if `dlsym`/`GetProcAddress` returns
    /// a null pointer; [`LoaderError::Libloading`] for other `libloading`
    /// failures.
    ///
    /// # Safety
    ///
    /// `T` must be a function-pointer type (`unsafe extern "C" fn(...) -> ...`)
    /// matching the C signature of `symbol`. Calling the returned symbol
    /// with the wrong signature is undefined behavior.
    pub unsafe fn symbol<T>(
        &self,
        symbol: &'static str,
    ) -> Result<libloading::Symbol<'_, T>, LoaderError> {
        let bytes_with_nul: Vec<u8> = symbol.bytes().chain(std::iter::once(0)).collect();
        let cstr = CStr::from_bytes_with_nul(&bytes_with_nul).map_err(|_| {
            LoaderError::SymbolNotFound {
                library: self.name,
                symbol,
            }
        })?;
        match self.lib.get::<T>(cstr.to_bytes_with_nul()) {
            Ok(s) => Ok(s),
            Err(_) => Err(LoaderError::SymbolNotFound {
                library: self.name,
                symbol,
            }),
        }
    }

    /// Return a raw pointer to the symbol without wrapping in `libloading::Symbol`.
    /// Useful for stashing function pointers in `OnceLock`s that outlive the
    /// borrow checker's view of the library.
    ///
    /// # Safety
    ///
    /// Same as [`Self::symbol`]. Additionally, the caller must ensure the
    /// [`Library`] outlives any use of the returned pointer — in practice this
    /// means storing the [`Library`] in a `static OnceLock<Library>` or
    /// equivalent.
    pub unsafe fn raw_symbol(&self, symbol: &'static str) -> Result<*mut (), LoaderError> {
        let sym: libloading::Symbol<'_, *mut ()> = self.symbol(symbol)?;
        Ok(*sym)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_library_reports_candidates() {
        let err = Library::open(
            "unobtanium",
            &["libunobtanium.so.42", "unobtanium64_42.dll"],
        );
        match err {
            Err(LoaderError::LibraryNotFound {
                library,
                candidates,
                ..
            }) => {
                assert_eq!(library, "unobtanium");
                assert_eq!(candidates.len(), 2);
            }
            Err(LoaderError::UnsupportedPlatform { .. }) => {
                // Acceptable on non-Linux/Windows CI runners.
            }
            other => panic!("expected LibraryNotFound, got {other:?}"),
        }
    }

    #[test]
    fn empty_candidates_returns_library_not_found() {
        let err = Library::open("nothing", &[]);
        match err {
            Err(LoaderError::LibraryNotFound { library, .. }) => {
                assert_eq!(library, "nothing");
            }
            Err(LoaderError::UnsupportedPlatform { .. }) => {}
            other => panic!("expected LibraryNotFound, got {other:?}"),
        }
    }
}
