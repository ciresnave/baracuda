// `baracuda_kernels_abi_version()`: returns the C-ABI version compiled into this library.
// The value lives in `include/baracuda_abi.h`; the Rust constant `ABI_VERSION` in
// `src/lib.rs` must equal it (`scripts/check-abi-manifest.py` asserts that).
#include "baracuda_abi.h"

extern "C" unsigned int baracuda_kernels_abi_version(void) {
    return BARACUDA_KERNELS_ABI_VERSION;
}
