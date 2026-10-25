# Third-party notice: `baracuda-tensorrt` and `baracuda-tensorrt-sys`

Nothing from NVIDIA is vendored or redistributed in this repository. The two inputs below are fetched by whoever builds or tests the `shim` feature.

## Headers (build time, `shim` feature)

`crates/baracuda-tensorrt-sys/shim/trt_shim.cpp` includes the TensorRT public headers (`NvInferRuntime.h` and friends). It is built and tested against the open-source copy in [NVIDIA/TensorRT](https://github.com/NVIDIA/TensorRT) at tag `v10.16`, commit `52399f555c2f80cb690a4a558b604e1a5f227e7c`, directory `include/`. Those files carry `SPDX-License-Identifier: Apache-2.0`.

That tree ships the version macros as placeholders (`NV_TENSORRT_MAJOR` is `TRT_MAJOR_ENTERPRISE`, and so on), so the build supplies them with `-D` (for MSVC, `CL=/DTRT_MAJOR_ENTERPRISE=10 /DTRT_MINOR_ENTERPRISE=16 /DTRT_PATCH_ENTERPRISE=1 /DTRT_BUILD_ENTERPRISE=11`). Point `TENSORRT_INCLUDE_DIR` at that `include/` directory.

## Runtime (run time)

The device tests load `nvinfer_10.dll` (Windows) or `libnvinfer.so.10` (Linux) through the dynamic loader. The version used for the measurements in `CHANGELOG.md` is the `tensorrt-cu13-libs` 10.16.1.11 wheel from NVIDIA's package index. Its metadata says `License: Proprietary` (`LICENSE.txt` in its `dist-info`). It is linked at run time only, never vendored, and never redistributed by this repository.
