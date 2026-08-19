#include "dxait/dxait.hpp"
#include "dxait/dxla.hpp"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <random>

using namespace dxait;

static void sync(Queue* q, Fence* f, uint64_t& fv) {
    q->signal(*f, ++fv);
    f->wait(fv);
}

// Staging upload -> Default -> GPU copy (mirrors dx_capi.cpp do_copy).
static void gpu_copy(Device* device, Queue* q, Buffer* src, uint64_t soff, Buffer* dst, uint64_t doff, uint64_t bytes) {
    static ComPtr<ID3D12CommandAllocator> alloc;
    static ComPtr<ID3D12GraphicsCommandList> list;
    if (!alloc) {
        device->get()->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&alloc));
        device->get()->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, alloc.Get(), nullptr, IID_PPV_ARGS(&list));
        list->Close();
    }
    alloc->Reset();
    list->Reset(alloc.Get(), nullptr);
    list->CopyBufferRegion(dst->get(), doff, src->get(), soff, bytes);
    list->Close();
    ID3D12CommandList* lists[] = { list.Get() };
    q->execute(lists, 1);
}

static std::unique_ptr<Buffer> to_default(Device* device, Queue* q, Fence* f, uint64_t& fv, const void* data, uint64_t bytes) {
    auto staging = device->create_buffer(bytes, MemLocation::Upload);
    std::memcpy(staging->map(), data, bytes);
    staging->unmap();
    auto def = device->create_buffer(bytes, MemLocation::Default);
    gpu_copy(device, q, staging.get(), 0, def.get(), 0, bytes);
    sync(q, f, fv);
    return def;
}

static std::vector<uint8_t> from_default(Device* device, Queue* q, Fence* f, uint64_t& fv, Buffer* def, uint64_t bytes) {
    auto rb = device->create_buffer(bytes, MemLocation::Readback);
    gpu_copy(device, q, def, 0, rb.get(), 0, bytes);
    sync(q, f, fv);
    std::vector<uint8_t> out(bytes);
    std::memcpy(out.data(), rb->map(), bytes);
    rb->unmap();
    return out;
}

static bool check_f32(const char* name, const float* got, const float* ref, uint32_t n, float tol) {
    uint32_t bad = 0;
    float max_err = 0.0f;
    for (uint32_t i = 0; i < n; ++i) {
        float err = std::fabs(got[i] - ref[i]);
        if (err > max_err) max_err = err;
        if (err > tol) ++bad;
    }
    printf("  %-14s got[0]=%.6f ref[0]=%.6f max_err=%.2e bad=%u/%u -> %s\n",
           name, got[0], ref[0], max_err, bad, n, bad == 0 ? "PASS" : "FAIL");
    return bad == 0;
}

static uint16_t f32_to_f16(float x) {
    uint32_t u; std::memcpy(&u, &x, 4);
    uint32_t sign = (u >> 16) & 0x8000u;
    int32_t exp = (int32_t)((u >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = u & 0x7FFFFFu;
    if (exp >= 31) return (uint16_t)(sign | 0x7C00u);
    if (exp <= 0) {
        mant |= 0x800000u;
        int32_t shift = 1 - exp;
        if (shift >= 24) return (uint16_t)sign;
        return (uint16_t)(sign | (mant >> shift));
    }
    uint16_t h = (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
    if ((mant & 0x1000u) && ((mant & 0x2000u) || (mant & 0xFFFu))) h++;
    return h;
}

static float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t u;
    if (exp == 0) {
        if (mant == 0) u = sign;
        else { uint32_t e = 127 - 15 + 1; while (!(mant & 0x400u)) { mant <<= 1; e--; } mant &= 0x3FFu; u = sign | (e << 23) | (mant << 13); }
    } else if (exp == 31) {
        u = sign | 0x7F800000u | (mant << 13);
    } else {
        u = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f; std::memcpy(&f, &u, 4);
    return f;
}

static float gelu(float x) { return x * 0.5f * (1.0f + std::erf(x / std::sqrt(2.0f))); }

int main() {
    printf("DXAiT LA op-layer test\n");
    printf("======================\n\n");

    auto adapters = Adapter::enumerate();
    if (adapters.empty()) { printf("No GPU found, skipping test.\n"); return 0; }
    auto device = Adapter::create_device(0);
    const AdapterCaps& caps = device->caps();
    printf("GPU: %s (vendor 0x%04X dev 0x%04X, WMMA=%d, arch=%d)\n",
           caps.name.c_str(), caps.vendor_id, caps.device_id,
           caps.wmma_supported ? 1 : 0, (int)caps.arch_family);

    LA la(device.get());
    auto q = device->create_queue(QueueType::Compute);
    auto fence = device->create_fence(0);
    uint64_t fv = 0;

    std::mt19937 rng(0x1A2B3C4Du);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> pos(0.01f, 1.0f);

    bool all_ok = true;

    // ---- elementwise Add/Sub/Mul/Div ----
    {
        constexpr uint32_t N = 10000;
        std::vector<float> a(N), b(N), ref(N);
        for (uint32_t i = 0; i < N; ++i) { a[i] = dist(rng); b[i] = dist(rng); }
        const uint64_t bytes = (uint64_t)N * 4;
        auto da = to_default(device.get(), q.get(), fence.get(), fv, a.data(), bytes);
        auto db = to_default(device.get(), q.get(), fence.get(), fv, b.data(), bytes);
        auto dout = device->create_buffer(bytes, MemLocation::Default);

        const char* names[] = { "Add", "Sub", "Mul", "Div" };
        for (int op = 0; op < 4; ++op) {
            for (uint32_t i = 0; i < N; ++i) {
                switch (op) {
                    case 0: ref[i] = a[i] + b[i]; break;
                    case 1: ref[i] = a[i] - b[i]; break;
                    case 2: ref[i] = a[i] * b[i]; break;
                    default: ref[i] = a[i] / b[i]; break;
                }
            }
            la.elementwise(q.get(), dout.get(), da.get(), db.get(), N, (LAOp)op, 1.0f, 0.0f);
            sync(q.get(), fence.get(), fv);
            auto res = from_default(device.get(), q.get(), fence.get(), fv, dout.get(), bytes);
            all_ok &= check_f32(names[op], (const float*)res.data(), ref.data(), N, 1e-3f);
        }
        printf("\n");
    }

    // ---- activations ----
    {
        constexpr uint32_t N = 10000;
        std::vector<float> in(N), ref(N);
        for (uint32_t i = 0; i < N; ++i) in[i] = dist(rng) * 3.0f;
        const uint64_t bytes = (uint64_t)N * 4;
        auto din = to_default(device.get(), q.get(), fence.get(), fv, in.data(), bytes);
        auto dout = device->create_buffer(bytes, MemLocation::Default);

        const char* names[] = { "relu", "gelu", "silu", "tanh", "sigmoid", "leaky" };
        const float alphas[] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.01f };
        for (int act = 0; act < 6; ++act) {
            for (uint32_t i = 0; i < N; ++i) {
                float x = in[i];
                switch (act) {
                    case 0: ref[i] = x > 0.0f ? x : 0.0f; break;
                    case 1: ref[i] = gelu(x); break;
                    case 2: ref[i] = x / (1.0f + std::exp(-x)); break;
                    case 3: ref[i] = std::tanh(x); break;
                    case 4: ref[i] = 1.0f / (1.0f + std::exp(-x)); break;
                    default: ref[i] = x >= 0.0f ? x : alphas[act] * x; break;
                }
            }
            la.activation(q.get(), dout.get(), din.get(), N, (LAActivation)act, alphas[act]);
            sync(q.get(), fence.get(), fv);
            auto res = from_default(device.get(), q.get(), fence.get(), fv, dout.get(), bytes);
            all_ok &= check_f32(names[act], (const float*)res.data(), ref.data(), N, 1e-3f);
        }
        printf("\n");
    }

    // ---- rmsnorm ----
    {
        constexpr uint32_t rows = 4, dim = 256;
        std::vector<float> in(rows * dim), gamma(dim), ref(rows * dim);
        for (uint32_t i = 0; i < rows * dim; ++i) in[i] = dist(rng);
        for (uint32_t j = 0; j < dim; ++j) gamma[j] = pos(rng);
        const float eps = 1e-5f;
        for (uint32_t r = 0; r < rows; ++r) {
            float s = 0.0f;
            for (uint32_t j = 0; j < dim; ++j) s += in[r * dim + j] * in[r * dim + j];
            float rstd = 1.0f / std::sqrt(s / (float)dim + eps);
            for (uint32_t j = 0; j < dim; ++j) ref[r * dim + j] = gamma[j] * in[r * dim + j] * rstd;
        }
        const uint64_t bytes = (uint64_t)rows * dim * 4;
        auto din = to_default(device.get(), q.get(), fence.get(), fv, in.data(), bytes);
        auto dg = to_default(device.get(), q.get(), fence.get(), fv, gamma.data(), (uint64_t)dim * 4);
        auto dout = device->create_buffer(bytes, MemLocation::Default);
        la.rmsnorm(q.get(), dout.get(), din.get(), dg.get(), rows, dim, eps);
        sync(q.get(), fence.get(), fv);
        auto res = from_default(device.get(), q.get(), fence.get(), fv, dout.get(), bytes);
        all_ok &= check_f32("rmsnorm", (const float*)res.data(), ref.data(), rows * dim, 1e-3f);
        printf("\n");
    }

    // ---- softmax ----
    {
        constexpr uint32_t rows = 8, dim = 64;
        std::vector<float> in(rows * dim), ref(rows * dim);
        for (uint32_t i = 0; i < rows * dim; ++i) in[i] = dist(rng) * 4.0f;
        for (uint32_t r = 0; r < rows; ++r) {
            float m = *std::max_element(in.begin() + r * dim, in.begin() + r * dim + dim);
            float s = 0.0f;
            for (uint32_t j = 0; j < dim; ++j) s += std::exp(in[r * dim + j] - m);
            for (uint32_t j = 0; j < dim; ++j) ref[r * dim + j] = std::exp(in[r * dim + j] - m) / s;
        }
        const uint64_t bytes = (uint64_t)rows * dim * 4;
        auto din = to_default(device.get(), q.get(), fence.get(), fv, in.data(), bytes);
        auto dout = device->create_buffer(bytes, MemLocation::Default);
        la.softmax(q.get(), dout.get(), din.get(), rows, dim);
        sync(q.get(), fence.get(), fv);
        auto res = from_default(device.get(), q.get(), fence.get(), fv, dout.get(), bytes);
        all_ok &= check_f32("softmax", (const float*)res.data(), ref.data(), rows * dim, 1e-3f);
        printf("\n");
    }

    // ---- reduce sum/max/min/mean ----
    {
        constexpr uint32_t rows = 8, dim = 64;
        std::vector<float> in(rows * dim), ref(rows);
        for (uint32_t i = 0; i < rows * dim; ++i) in[i] = dist(rng);
        const uint64_t ibytes = (uint64_t)rows * dim * 4;
        const uint64_t obytes = (uint64_t)rows * 4;
        auto din = to_default(device.get(), q.get(), fence.get(), fv, in.data(), ibytes);
        auto dout = device->create_buffer(obytes, MemLocation::Default);

        const char* names[] = { "sum", "max", "min", "mean" };
        for (int op = 0; op < 4; ++op) {
            for (uint32_t r = 0; r < rows; ++r) {
                float v = (op == 0 || op == 3) ? 0.0f : in[r * dim];
                for (uint32_t j = 0; j < dim; ++j) {
                    float x = in[r * dim + j];
                    switch (op) {
                        case 0: v += x; break;
                        case 1: v = (std::max)(v, x); break;
                        case 2: v = (std::min)(v, x); break;
                        default: v += x; break;
                    }
                }
                ref[r] = (op == 3) ? v / (float)dim : v;
            }
            la.reduce(q.get(), dout.get(), din.get(), rows, dim, (LAReduce)op);
            sync(q.get(), fence.get(), fv);
            auto res = from_default(device.get(), q.get(), fence.get(), fv, dout.get(), obytes);
            all_ok &= check_f32(names[op], (const float*)res.data(), ref.data(), rows, 1e-3f);
        }
        printf("\n");
    }

    // ---- gemm_f16_dot2 (M=N=K=16, already 16-aligned) ----
    {
        constexpr uint32_t M = 16, N = 16, K = 16;
        std::vector<uint16_t> af(M * K), bf(K * N), aref(M * K), bref(K * N);
        for (uint32_t r = 0; r < M; ++r)
            for (uint32_t k = 0; k < K; ++k) { float v = 0.25f * (float)r + 0.1f * (float)k; af[r * K + k] = f32_to_f16(v); aref[r * K + k] = (uint16_t)af[r * K + k]; }
        for (uint32_t k = 0; k < K; ++k)
            for (uint32_t c = 0; c < N; ++c) { float v = 0.5f * (float)(k + 1); bf[k * N + c] = f32_to_f16(v); bref[k * N + c] = (uint16_t)bf[k * N + c]; }

        // pack halfs per uint: A pairs along K (row-major flat), B pairs along K per column
        std::vector<uint32_t> apack(M * K / 2), bpack(K * N / 2);
        for (uint32_t i = 0; i < M * K / 2; ++i) apack[i] = (uint32_t)af[2 * i] | ((uint32_t)af[2 * i + 1] << 16);
        for (uint32_t c = 0; c < N; ++c)
            for (uint32_t k2 = 0; k2 < K / 2; ++k2)
                bpack[c * (K / 2) + k2] = (uint32_t)bf[(2 * k2) * N + c] | ((uint32_t)bf[(2 * k2 + 1) * N + c] << 16);

        std::vector<float> ref(M * N, 0.0f);
        for (uint32_t r = 0; r < M; ++r)
            for (uint32_t c = 0; c < N; ++c) {
                float s = 0.0f;
                for (uint32_t k = 0; k < K; ++k) s += f16_to_f32(af[r * K + k]) * f16_to_f32(bf[k * N + c]);
                ref[r * N + c] = s;
            }

        auto da = to_default(device.get(), q.get(), fence.get(), fv, apack.data(), apack.size() * 4);
        auto db = to_default(device.get(), q.get(), fence.get(), fv, bpack.data(), bpack.size() * 4);
        auto dout = device->create_buffer((uint64_t)M * N * 4, MemLocation::Default);

        la.gemm_f16_dot2(q.get(), dout.get(), da.get(), db.get(), M, N, K);
        sync(q.get(), fence.get(), fv);
        auto res = from_default(device.get(), q.get(), fence.get(), fv, dout.get(), (uint64_t)M * N * 4);
        all_ok &= check_f32("gemm_f16_dot2", (const float*)res.data(), ref.data(), M * N, 1e-2f);
        printf("\n");
    }

    // ---- gemm_f16_wmma (WMMA→dot2: WaveMatrix HLSL was shelved, so this entry
    //      dispatches the dot2 kernel - same result on all hardware) ----
    {
        constexpr uint32_t M = 16, N = 16, K = 16;
        std::vector<uint16_t> af(M * K), bf(K * N);
        for (uint32_t r = 0; r < M; ++r)
            for (uint32_t k = 0; k < K; ++k) af[r * K + k] = f32_to_f16(0.25f * (float)r + 0.1f * (float)k);
        for (uint32_t k = 0; k < K; ++k)
            for (uint32_t c = 0; c < N; ++c) bf[k * N + c] = f32_to_f16(0.5f * (float)(k + 1));
        std::vector<uint32_t> apack(M * K / 2), bpack(K * N / 2);
        for (uint32_t i = 0; i < M * K / 2; ++i) apack[i] = (uint32_t)af[2 * i] | ((uint32_t)af[2 * i + 1] << 16);
        for (uint32_t c = 0; c < N; ++c)
            for (uint32_t k2 = 0; k2 < K / 2; ++k2)
                bpack[c * (K / 2) + k2] = (uint32_t)bf[(2 * k2) * N + c] | ((uint32_t)bf[(2 * k2 + 1) * N + c] << 16);
        std::vector<float> ref(M * N, 0.0f);
        for (uint32_t r = 0; r < M; ++r)
            for (uint32_t c = 0; c < N; ++c) {
                float s = 0.0f;
                for (uint32_t k = 0; k < K; ++k) s += f16_to_f32(af[r * K + k]) * f16_to_f32(bf[k * N + c]);
                ref[r * N + c] = s;
            }
        auto da = to_default(device.get(), q.get(), fence.get(), fv, apack.data(), apack.size() * 4);
        auto db = to_default(device.get(), q.get(), fence.get(), fv, bpack.data(), bpack.size() * 4);
        auto dout = device->create_buffer((uint64_t)M * N * 4, MemLocation::Default);
        la.gemm_f16_wmma(q.get(), dout.get(), da.get(), db.get(), M, N, K);
        sync(q.get(), fence.get(), fv);
        auto res = from_default(device.get(), q.get(), fence.get(), fv, dout.get(), (uint64_t)M * N * 4);
        all_ok &= check_f32("gemm_f16_wmma", (const float*)res.data(), ref.data(), M * N, 1e-2f);
    }

    printf("\nOverall: %s\n", all_ok ? "ALL TESTS PASSED" : "FAILED");
    return all_ok ? 0 : 1;
}
