// On-device acceptance for the form-B `_doff` WriteSlice variant
// (device-resident dynamic-axis range-start).
//
// The `_doff` launchers read the range-start of ONE axis (`dyn_axis`) from a
// DEVICE pointer (`dyn_start_dev[0]`, a single i64) at kernel entry, instead
// of baking it host-side into the by-value `range_start`. WHY: Fuel's
// KV-cache decode wants CUDA-graph replay (cuGraphLaunch once/token vs ~150
// launches). The by-value `range_start` is marshaled into the captured graph
// node's param space, so the host-baked seq position FREEZES at the captured
// token. The `_doff` variant reads the position from device memory, which the
// host updates per token via a fixed-address H2D memcpy that is capture-
// tolerant. This harness proves form B survives graph replay where form A
// (the baked `_run`) would freeze.
//
// This TU is self-contained: it `#include`s `baracuda_write_slice.cuh` by
// absolute path and instantiates BOTH the `_run` (`INSTANTIATE`) and `_doff`
// (`INSTANTIATE_DOFF`) macros locally, so it holds both symbol families and
// links standalone (NOT against the crate's static lib). It is NOT wired into
// `cargo test` or `build.rs` — mirrors the kernelgen `ondevice/` convention.
//
// Cells (§6 of the brief):
//   A  : stream-capture + replay. Capture {H2D memcpy -> dyn_start_dev, the
//        `_doff` launch}; instantiate; replay over a seq sweep, mutating the
//        pinned host source of the memcpy each token. Assert each write lands
//        at the UPDATED slab AND no other row (incl. prior KV writes) is
//        disturbed. The load-bearing form-B-survives-replay proof.
//   A' : same, but the per-token position update goes through
//        cudaGraphExecMemcpyNodeSetParams (the node-update path Fuel uses).
//   B  : `_doff`-at-fixed-start vs baked `_run`-at-matched-start whole-buffer
//        memcmp == 0, across b1/b2/b4/b8 + NaN/Inf/subnormal/±0 probes. The
//        additive byte-identity witness (the `_run` PTX is untouched).
//   C  : compute-sanitizer edge — p = max_seq-1 (slab touches the buffer end)
//        under memcheck/racecheck. Run via the `san` arg (small shapes).
//
// GPU/toolchain: RTX 4070 Laptop (sm_89), CUDA 13.3 / nvcc.
//
// Build (from a VS dev shell so nvcc finds cl.exe):
//   nvcc -O3 -arch=sm_89 -std=c++17 -Xcompiler "/Zc:preprocessor /std:c++17" \
//        crates/baracuda-kernels-sys/ondevice/write_slice_doff_validate.cu \
//        -o write_slice_doff_validate
//   ./write_slice_doff_validate            (full)
//   ./write_slice_doff_validate san        (small shapes, for the sanitizers)
//   compute-sanitizer --tool memcheck   ./write_slice_doff_validate san
//   compute-sanitizer --tool racecheck  ./write_slice_doff_validate san

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

// The bespoke WriteSlice kernel + launchers + INSTANTIATE macros.
#include "../kernels/include/baracuda_write_slice.cuh"

// Instantiate BOTH families locally so this TU holds `_run` and `_doff_run`.
BARACUDA_KERNELS_WRITE_SLICE_INSTANTIATE(b1, baracuda::write_slice::Blob1)
BARACUDA_KERNELS_WRITE_SLICE_INSTANTIATE(b2, baracuda::write_slice::Blob2)
BARACUDA_KERNELS_WRITE_SLICE_INSTANTIATE(b4, baracuda::write_slice::Blob4)
BARACUDA_KERNELS_WRITE_SLICE_INSTANTIATE(b8, baracuda::write_slice::Blob8)
BARACUDA_KERNELS_WRITE_SLICE_INSTANTIATE_DOFF(b1, baracuda::write_slice::Blob1)
BARACUDA_KERNELS_WRITE_SLICE_INSTANTIATE_DOFF(b2, baracuda::write_slice::Blob2)
BARACUDA_KERNELS_WRITE_SLICE_INSTANTIATE_DOFF(b4, baracuda::write_slice::Blob4)
BARACUDA_KERNELS_WRITE_SLICE_INSTANTIATE_DOFF(b8, baracuda::write_slice::Blob8)

static int fails = 0;
#define CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); fails++; } } while (0)

static void pass(const char* label) { printf("PASS %-58s\n", label); }
static void fail(const char* label, const char* why) { printf("FAIL %-58s %s\n", label, why); fails++; }

// ---- `_run` / `_doff_run` FFI signatures (as emitted by the macros) -------
// Byte-width -> the pair of extern "C" launchers. We call them by width.
typedef int32_t (*run_fn)(void*, const void*, int64_t, int32_t,
                          const int32_t*, const int32_t*, const int32_t*,
                          void*, size_t, void*);
typedef int32_t (*doff_fn)(void*, const void*, int64_t, int32_t,
                           const int32_t*, const int32_t*, const int32_t*,
                           int32_t, const long long*, void*, size_t, void*);

struct WidthLaunchers { int esz; const char* tag; run_fn run; doff_fn doff; };

static WidthLaunchers kWidths[] = {
    {1, "b1", &baracuda_kernels_write_slice_b1_run, &baracuda_kernels_write_slice_b1_doff_run},
    {2, "b2", &baracuda_kernels_write_slice_b2_run, &baracuda_kernels_write_slice_b2_doff_run},
    {4, "b4", &baracuda_kernels_write_slice_b4_run, &baracuda_kernels_write_slice_b4_doff_run},
    {8, "b8", &baracuda_kernels_write_slice_b8_run, &baracuda_kernels_write_slice_b8_doff_run},
};

// A deterministic byte pattern for the source slab. `_run` / `_doff` are a
// byte-level memcpy so any pattern round-trips; the goal is a per-(h,d)
// distinguishable value so a mis-landed slab is caught.
static void fill_pattern(std::vector<uint8_t>& buf, uint64_t seed) {
    uint64_t s = seed ? seed : 0x9e3779b97f4a7c15ull;
    for (size_t i = 0; i < buf.size(); ++i) {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        buf[i] = (uint8_t)(s >> 24);
    }
}

// The house f32 probe classes (Cell B bit-exactness): NaN payloads / ±Inf /
// subnormal / ±0 / extremes. Written as 4-byte encodings; for b1/b2/b8 we
// reinterpret the same byte stream (the copy is byte-level).
static const uint32_t kProbeU32[] = {
    0x00000000u, 0x80000000u,              // +0, -0
    0x3f800000u, 0xbf800000u,              // +1, -1
    0x7f800000u, 0xff800000u,              // +inf, -inf
    0x7fc00000u, 0xffc00000u,              // qNaN, -qNaN
    0x7fc00001u, 0x7fdead01u, 0x7ff00000u, // qNaN payloads
    0x7f800001u, 0x7fbfffffu,              // sNaN payloads
    0xffc0beefu, 0xffbfffffu,              // negative NaN payloads
    0x00000001u, 0x80000001u,              // +/- smallest subnormal
    0x007fffffu, 0x807fffffu,              // +/- largest subnormal
    0x00800000u, 0x80800000u,              // +/- smallest normal
    0x7f7fffffu, 0xff7fffffu,              // +/- largest finite
};
static void probe_head(std::vector<uint8_t>& buf) {
    size_t nb = sizeof kProbeU32;
    if (nb > buf.size()) nb = buf.size();
    memcpy(buf.data(), kProbeU32, nb);
}

// A KV fixture. rank-4 KV decode: dest=[1,H,S,D], source=[1,H,1,D],
// dyn_axis=2. rank-3 layout: dest=[S,H,D], source=[1,H,D], dyn_axis=0.
struct Fixture {
    int rank;
    int dyn_axis;
    int H, S, D;             // heads, max_seq, head_dim
    std::vector<int32_t> dest_shape, source_shape, range_start;
    int64_t source_numel;    // elements in the source slab
    // Flat element offset of head `h`, seq `p`, feature `d` in dest.
    int64_t slab_off(int h, int p, int d) const {
        if (rank == 4) return (int64_t)h * S * D + (int64_t)p * D + d; // [1,H,S,D]
        return (int64_t)p * H * D + (int64_t)h * D + d;               // [S,H,D]
    }
    int64_t dest_numel() const { return (int64_t)H * S * D; }
};

static Fixture make_fixture(int rank, int H, int S, int D) {
    Fixture f; f.rank = rank; f.H = H; f.S = S; f.D = D;
    if (rank == 4) {
        f.dyn_axis = 2;
        f.dest_shape   = {1, H, S, D};
        f.source_shape = {1, H, 1, D};
        f.range_start  = {0, 0, /*dyn placeholder*/ 0, 0};
        f.source_numel = (int64_t)1 * H * 1 * D;
    } else { // rank 3
        f.dyn_axis = 0;
        f.dest_shape   = {S, H, D};
        f.source_shape = {1, H, D};
        f.range_start  = {/*dyn placeholder*/ 0, 0, 0};
        f.source_numel = (int64_t)1 * H * D;
    }
    return f;
}

// -------------------------------------------------------------------------
// Cell A / A' : capture the {H2D memcpy -> dyn_start_dev, `_doff` launch}
// pair, instantiate, and replay over a seq sweep. Assert each token's write
// lands at the UPDATED slab and NO other row (incl. prior KV writes) moves.
//   mechanism 0 = mutate the pinned host source of the memcpy (Cell A)
//   mechanism 1 = cudaGraphExecMemcpyNodeSetParams (Cell A')
// -------------------------------------------------------------------------
static void case_capture_replay(const WidthLaunchers& w, const Fixture& f,
                                int mechanism, const std::vector<int>& sweep) {
    const int esz = w.esz;
    const size_t dest_bytes = (size_t)f.dest_numel() * esz;
    const size_t src_bytes  = (size_t)f.source_numel * esz;

    std::vector<uint8_t> h_src(src_bytes);
    fill_pattern(h_src, 0xABCDEF01ull ^ (uint64_t)esz);

    void* d_dest = nullptr; CHECK(cudaMalloc(&d_dest, dest_bytes));
    void* d_src  = nullptr; CHECK(cudaMalloc(&d_src, src_bytes));
    CHECK(cudaMemcpy(d_src, h_src.data(), src_bytes, cudaMemcpyHostToDevice));
    long long* d_pos = nullptr; CHECK(cudaMalloc(&d_pos, sizeof(long long)));

    // Device shape arrays are host pointers per the ABI — keep host copies.
    const int32_t* dshape = f.dest_shape.data();
    const int32_t* sshape = f.source_shape.data();
    const int32_t* rstart = f.range_start.data();

    // Pinned host slot holding the position the captured memcpy reads.
    long long* h_pos = nullptr; CHECK(cudaMallocHost(&h_pos, sizeof(long long)));
    // A separate pinned slot for the node-param-set mechanism (A').
    long long* h_pos_node = nullptr; CHECK(cudaMallocHost(&h_pos_node, sizeof(long long)));

    cudaStream_t stream; CHECK(cudaStreamCreate(&stream));

    // ---- Capture ----
    CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    CHECK(cudaMemcpyAsync(d_pos, h_pos, sizeof(long long),
                          cudaMemcpyHostToDevice, stream));
    int32_t st = w.doff((void*)d_dest, (const void*)d_src, f.source_numel, f.rank,
                        dshape, sshape, rstart, f.dyn_axis, d_pos,
                        /*workspace*/ nullptr, 0, (void*)stream);
    cudaGraph_t graph;
    CHECK(cudaStreamEndCapture(stream, &graph));
    if (st != 0) { char why[64]; snprintf(why, sizeof why, "doff launch status %d during capture", st); fail("capture status", why); }

    cudaGraphExec_t exec;
    CHECK(cudaGraphInstantiate(&exec, graph, 0));

    // For A', find the memcpy node.
    cudaGraphNode_t memcpy_node = nullptr;
    if (mechanism == 1) {
        size_t n = 0; CHECK(cudaGraphGetNodes(graph, nullptr, &n));
        std::vector<cudaGraphNode_t> nodes(n);
        CHECK(cudaGraphGetNodes(graph, nodes.data(), &n));
        for (size_t i = 0; i < n; ++i) {
            cudaGraphNodeType t; CHECK(cudaGraphNodeGetType(nodes[i], &t));
            if (t == cudaGraphNodeTypeMemcpy) { memcpy_node = nodes[i]; break; }
        }
        if (!memcpy_node) fail("A' find memcpy node", "no memcpy node in captured graph");
    }

    // Host mirror of the expected dest state (KV cache built up token by token).
    std::vector<uint8_t> expected(dest_bytes, 0x00);
    CHECK(cudaMemset(d_dest, 0, dest_bytes)); // sentinel: fresh KV cache

    long long bad_p = -2;
    for (int p : sweep) {
        // Update the position via the selected mechanism, then replay.
        if (mechanism == 0) {
            *h_pos = p; // the captured memcpy reads this fixed address each replay
        } else {
            *h_pos_node = p;
            cudaMemcpy3DParms params; memset(&params, 0, sizeof params);
            params.srcPtr = make_cudaPitchedPtr((void*)h_pos_node, sizeof(long long), 1, 1);
            params.dstPtr = make_cudaPitchedPtr((void*)d_pos, sizeof(long long), 1, 1);
            params.extent = make_cudaExtent(sizeof(long long), 1, 1);
            params.kind   = cudaMemcpyHostToDevice;
            CHECK(cudaGraphExecMemcpyNodeSetParams(exec, memcpy_node, &params));
        }
        CHECK(cudaGraphLaunch(exec, stream));
        CHECK(cudaStreamSynchronize(stream));

        // Update the host mirror: slab p (every head) now holds the source.
        for (int h = 0; h < f.H; ++h) {
            int64_t base = f.slab_off(h, p, 0) * esz;
            int64_t sbase = (int64_t)h * f.D * esz; // source slab for head h
            memcpy(&expected[(size_t)base], &h_src[(size_t)sbase], (size_t)f.D * esz);
        }

        std::vector<uint8_t> got(dest_bytes);
        CHECK(cudaMemcpy(got.data(), d_dest, dest_bytes, cudaMemcpyDeviceToHost));
        if (memcmp(got.data(), expected.data(), dest_bytes) != 0) { bad_p = p; break; }
    }

    char lbl[128];
    snprintf(lbl, sizeof lbl, "%s rank%d ax%d %s replay sweep",
             w.tag, f.rank, f.dyn_axis, mechanism ? "A'(node-set)" : "A(mem-mutate)");
    if (bad_p == -2) pass(lbl);
    else { char why[64]; snprintf(why, sizeof why, "mismatch at seq p=%lld", bad_p); fail(lbl, why); }

    CHECK(cudaGraphExecDestroy(exec));
    CHECK(cudaGraphDestroy(graph));
    CHECK(cudaStreamDestroy(stream));
    cudaFreeHost(h_pos); cudaFreeHost(h_pos_node);
    cudaFree(d_dest); cudaFree(d_src); cudaFree(d_pos);
}

// -------------------------------------------------------------------------
// Cell B : `_doff` at a fixed device start == baked `_run` at the matched
// host start, whole-buffer memcmp == 0. Sweeps b1/b2/b4/b8, dyn edges, and
// seeds the source with the NaN/Inf/subnormal/±0 probe classes. This is the
// additive byte-identity witness: the `_run` kernel PTX is untouched, so the
// two paths must produce bit-identical buffers at the same start.
// -------------------------------------------------------------------------
static void case_run_identity(const WidthLaunchers& w, const Fixture& f, int p, bool probe) {
    const int esz = w.esz;
    const size_t dest_bytes = (size_t)f.dest_numel() * esz;
    const size_t src_bytes  = (size_t)f.source_numel * esz;

    std::vector<uint8_t> h_src(src_bytes);
    fill_pattern(h_src, 0x1234567 ^ (uint64_t)(p * 131 + esz));
    if (probe) probe_head(h_src);

    void* d_src = nullptr; CHECK(cudaMalloc(&d_src, src_bytes));
    CHECK(cudaMemcpy(d_src, h_src.data(), src_bytes, cudaMemcpyHostToDevice));
    void* d_g = nullptr; CHECK(cudaMalloc(&d_g, dest_bytes)); // _doff result
    void* d_r = nullptr; CHECK(cudaMalloc(&d_r, dest_bytes)); // _run  result
    CHECK(cudaMemset(d_g, 0xEE, dest_bytes));
    CHECK(cudaMemset(d_r, 0xEE, dest_bytes)); // identical sentinel fill

    long long* d_pos = nullptr; CHECK(cudaMalloc(&d_pos, sizeof(long long)));
    long long hp = p; CHECK(cudaMemcpy(d_pos, &hp, sizeof(long long), cudaMemcpyHostToDevice));

    // _doff: dyn slot from device; host range_start[dyn_axis] is a placeholder.
    int32_t st_g = w.doff((void*)d_g, (const void*)d_src, f.source_numel, f.rank,
                          f.dest_shape.data(), f.source_shape.data(), f.range_start.data(),
                          f.dyn_axis, d_pos, nullptr, 0, nullptr);
    // _run: bake p into range_start[dyn_axis] host-side (form A).
    std::vector<int32_t> rstart_baked = f.range_start;
    rstart_baked[f.dyn_axis] = p;
    int32_t st_r = w.run((void*)d_r, (const void*)d_src, f.source_numel, f.rank,
                         f.dest_shape.data(), f.source_shape.data(), rstart_baked.data(),
                         nullptr, 0, nullptr);
    CHECK(cudaDeviceSynchronize());

    char lbl[128];
    snprintf(lbl, sizeof lbl, "%s rank%d ax%d p=%d%s doff==run memcmp",
             w.tag, f.rank, f.dyn_axis, p, probe ? " [probe]" : "");
    if (st_g != 0 || st_r != 0) {
        char why[64]; snprintf(why, sizeof why, "status doff=%d run=%d", st_g, st_r);
        fail(lbl, why);
    } else {
        std::vector<uint8_t> g(dest_bytes), r(dest_bytes);
        CHECK(cudaMemcpy(g.data(), d_g, dest_bytes, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(r.data(), d_r, dest_bytes, cudaMemcpyDeviceToHost));
        if (memcmp(g.data(), r.data(), dest_bytes) == 0) pass(lbl);
        else {
            size_t i = 0; while (i < dest_bytes && g[i] == r[i]) ++i;
            char why[80]; snprintf(why, sizeof why, "first diff byte %zu: 0x%02x vs 0x%02x", i, g[i], r[i]);
            fail(lbl, why);
        }
    }
    cudaFree(d_src); cudaFree(d_g); cudaFree(d_r); cudaFree(d_pos);
}

// -------------------------------------------------------------------------
// Guard rail: `_doff_can_implement` host-side validation (dyn_axis range +
// null dyn_start_dev), no device deref. A tiny sanity net.
// -------------------------------------------------------------------------
static void case_can_implement() {
    Fixture f = make_fixture(4, 8, 64, 128);
    long long dummy = 0; long long* d_pos = nullptr; CHECK(cudaMalloc(&d_pos, sizeof(long long)));
    CHECK(cudaMemcpy(d_pos, &dummy, sizeof(long long), cudaMemcpyHostToDevice));
    int32_t ok  = baracuda_kernels_write_slice_b2_doff_can_implement(
        nullptr, nullptr, f.source_numel, f.rank, f.dest_shape.data(),
        f.source_shape.data(), f.range_start.data(), f.dyn_axis, d_pos);
    int32_t bad_axis = baracuda_kernels_write_slice_b2_doff_can_implement(
        nullptr, nullptr, f.source_numel, f.rank, f.dest_shape.data(),
        f.source_shape.data(), f.range_start.data(), /*dyn_axis*/ 9, d_pos);
    int32_t bad_null = baracuda_kernels_write_slice_b2_doff_can_implement(
        nullptr, nullptr, f.source_numel, f.rank, f.dest_shape.data(),
        f.source_shape.data(), f.range_start.data(), f.dyn_axis, /*dyn_start_dev*/ nullptr);
    if (ok == 0 && bad_axis == 2 && bad_null == 2) pass("can_implement: ok=0, oob-axis=2, null-dev=2");
    else { char why[80]; snprintf(why, sizeof why, "ok=%d oob=%d null=%d", ok, bad_axis, bad_null); fail("can_implement", why); }
    cudaFree(d_pos);
}

int main(int argc, char** argv) {
    bool san = (argc > 1 && strcmp(argv[1], "san") == 0);
    printf("== write_slice_doff_validate (form-B device-resident dyn start) ==\n");

    int nd = 0; cudaGetDeviceCount(&nd);
    if (nd == 0) { printf("no CUDA device — SKIP\nRESULT: SKIP\n"); return 0; }

    // Fixtures. rank-4 KV decode (dyn_axis=2, the CapturedRun consumer) and
    // rank-3 layout (dyn_axis=0). Small shapes for the sanitizer run.
    const int H = san ? 4 : 8;
    const int S = san ? 8 : 64;
    const int D = san ? 16 : 128;
    Fixture f4 = make_fixture(4, H, S, D);
    Fixture f3 = make_fixture(3, H, S, D);

    // Seq sweep {0,1,7,S/2,S-1} clamped to distinct in-range positions.
    std::vector<int> sweep;
    for (int p : {0, 1, 7, S / 2, S - 1}) if (p >= 0 && p < S) {
        bool dup = false; for (int q : sweep) if (q == p) dup = true;
        if (!dup) sweep.push_back(p);
    }

    case_can_implement();

    if (san) {
        // Sanitizer/edge (Cell C): b2 primary, both layouts, incl. p=S-1
        // (slab touches the buffer end). Small shapes keep memcheck fast.
        case_capture_replay(kWidths[1], f4, /*A */ 0, sweep);
        case_capture_replay(kWidths[1], f4, /*A'*/ 1, sweep);
        case_capture_replay(kWidths[1], f3, /*A */ 0, sweep);
        case_run_identity(kWidths[1], f4, S - 1, /*probe*/ true);
        case_run_identity(kWidths[1], f3, S - 1, /*probe*/ true);
        printf(fails ? "\n%d case(s) FAILED\nRESULT: FAIL\n" : "\nRESULT: ALL PASSED\n", fails);
        return fails ? 1 : 0;
    }

    // Cell A + A': capture/replay across every width, rank-4 (dyn_axis=2).
    for (auto& w : kWidths) {
        case_capture_replay(w, f4, /*A  mem-mutate*/ 0, sweep);
        case_capture_replay(w, f4, /*A' node-set  */ 1, sweep);
    }
    // Rank-3 layout (dyn_axis=0) — b2 + b4 (the two live KV dtypes), both
    // replay mechanisms each (A mem-mutate + A' node-set), a symmetric 4 cells.
    case_capture_replay(kWidths[1], f3, 0, sweep);
    case_capture_replay(kWidths[1], f3, 1, sweep);
    case_capture_replay(kWidths[2], f3, 0, sweep);
    case_capture_replay(kWidths[2], f3, 1, sweep);

    // Cell B: `_doff` == baked `_run`, all widths, dyn edges {0, S-1}, +probe.
    for (auto& w : kWidths) {
        for (int p : {0, S / 2, S - 1}) {
            case_run_identity(w, f4, p, /*probe*/ false);
        }
        case_run_identity(w, f4, 0,     /*probe*/ true);
        case_run_identity(w, f4, S - 1, /*probe*/ true);
        case_run_identity(w, f3, S - 1, /*probe*/ true);
    }

    printf(fails ? "\n%d case(s) FAILED\nRESULT: FAIL\n" : "\nRESULT: ALL PASSED\n", fails);
    return fails ? 1 : 0;
}
