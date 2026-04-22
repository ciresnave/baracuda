//! Error types shared across the baracuda crates.

use std::path::PathBuf;

use baracuda_types::{CudaStatus, CudaVersion};
use thiserror::Error;

/// An error raised by the dynamic loader.
///
/// These surface whenever an NVIDIA shared library or one of its symbols
/// cannot be resolved at runtime — typically because CUDA is not installed,
/// the installed driver is older than what baracuda was built against, or
/// the user is on a platform NVIDIA doesn't support.
#[derive(Debug, Error)]
pub enum LoaderError {
    /// None of the candidate library filenames resolved anywhere on the
    /// library search path.
    #[error("could not load {library}: tried {candidates:?} across {search_paths} path(s)")]
    LibraryNotFound {
        library: &'static str,
        candidates: Vec<&'static str>,
        search_paths: usize,
    },

    /// The library was loaded but did not export the requested symbol.
    #[error("library '{library}' is missing symbol '{symbol}'")]
    SymbolNotFound {
        library: &'static str,
        symbol: &'static str,
    },

    /// `cuGetProcAddress` returned `CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT`
    /// for `symbol`: the installed driver does not provide it at the version
    /// baracuda asked for.
    #[error("symbol '{symbol}' requires {required} but baracuda's driver loader sees {installed}")]
    VersionTooOld {
        symbol: &'static str,
        required: CudaVersion,
        installed: CudaVersion,
    },

    /// Raw `libloading` error — kept for platform-specific diagnostics that
    /// the other variants can't express (e.g. a missing dependency on a
    /// chained `.so`).
    #[error("{0}")]
    Libloading(#[from] libloading::Error),

    /// baracuda does not target this platform (e.g. macOS).
    #[error("baracuda does not support {platform}; NVIDIA driver is only available on Linux and Windows")]
    UnsupportedPlatform { platform: &'static str },
}

impl LoaderError {
    /// Convenience constructor for the common case of "tried these names,
    /// none worked".
    pub fn library_not_found(library: &'static str, candidates: &[&'static str]) -> Self {
        Self::LibraryNotFound {
            library,
            candidates: candidates.to_vec(),
            search_paths: 0,
        }
    }

    /// As above, but records how many directories were searched.
    pub fn library_not_found_with_search(
        library: &'static str,
        candidates: &[&'static str],
        search_path_count: usize,
    ) -> Self {
        Self::LibraryNotFound {
            library,
            candidates: candidates.to_vec(),
            search_paths: search_path_count,
        }
    }
}

/// A generic error enum for any safe wrapper crate over a single NVIDIA
/// library. Safe crates may use this directly or compose their own richer
/// `Error` enum out of its variants.
#[derive(Debug, Error)]
pub enum Error<S>
where
    S: CudaStatus + Send + Sync + 'static,
{
    /// The library returned a non-success status code.
    #[error("{} returned {} ({}): {}", .status.library(), .status.name(), .status.code(), .status.description())]
    Status { status: S },

    /// The dynamic loader failed.
    #[error(transparent)]
    Loader(#[from] LoaderError),

    /// The requested API is newer than the installed driver supports.
    #[error("{api} requires {since}; install a newer driver to use it")]
    FeatureNotSupported {
        api: &'static str,
        since: CudaVersion,
    },
}

impl<S> Error<S>
where
    S: CudaStatus + Send + Sync + 'static,
{
    /// Treat a raw status code as a `Result`. Success codes yield `Ok(())`,
    /// all others yield `Err(Error::Status { .. })`.
    pub fn check(status: S) -> Result<(), Self> {
        if status.is_success() {
            Ok(())
        } else {
            Err(Self::Status { status })
        }
    }
}

/// A library-erased error, useful at process boundaries where the caller
/// doesn't want to parameterize over every NVIDIA library's status enum.
#[derive(Debug, Error)]
pub enum BaracudaError {
    /// A status code from any NVIDIA library.
    #[error("{library} returned {name} ({code}): {description}")]
    Status {
        library: &'static str,
        name: &'static str,
        description: &'static str,
        code: i32,
    },

    /// The dynamic loader failed.
    #[error(transparent)]
    Loader(#[from] LoaderError),

    /// The requested API is newer than the installed driver supports.
    #[error("{api} requires {since}; install a newer driver to use it")]
    FeatureNotSupported {
        api: &'static str,
        since: CudaVersion,
    },

    /// For sources that want to attach a path or other context (e.g. a
    /// missing PTX file).
    #[error("{context}")]
    Context { context: &'static str },
}

impl<S> From<Error<S>> for BaracudaError
where
    S: CudaStatus + Send + Sync + 'static,
{
    fn from(err: Error<S>) -> Self {
        match err {
            Error::Status { status } => BaracudaError::Status {
                library: status.library(),
                name: status.name(),
                description: status.description(),
                code: status.code(),
            },
            Error::Loader(l) => BaracudaError::Loader(l),
            Error::FeatureNotSupported { api, since } => {
                BaracudaError::FeatureNotSupported { api, since }
            }
        }
    }
}

/// Path-returning variant used by `find_library` probes. (Kept out of public
/// API surface for now — re-exported here so doc-links work.)
#[allow(dead_code)]
pub(crate) type PathList = Vec<PathBuf>;
