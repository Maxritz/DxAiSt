#include "dxait/dx_c_api.h"

#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <vector>

// Pure-C-API consumption demo: only dx_c_api.h, no C++ dxait classes.
// Links against the shipped dxait.dll.

static void fail(const char* step, HRESULT hr) {
    printf("FAILED: %s (HRESULT 0x%08lX) err: %s\n", step, (unsigned long)hr, dx_last_error());
    exit(1);
}

static uint16_t f32_to_f16(float x) {
    uint32_t u; memcpy(&u, &x, 4);
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

int main() {
    printf("dxla_playground (C API demo against dxait.dll)\n");
    printf("===============================================\n\n");

    dx_device* dev = nullptr;
    HRESULT hr = dx_create_device(0, &dev);
    if (FAILED(hr)) fail("dx_create_device", hr);

    char name[256] = {};
    hr = dx_device_desc(dev, name, sizeof(name));
    if (FAILED(hr)) fail("dx_device_desc", hr);
    printf("Device: %s\n", name);

    dx_queue* q = nullptr;
    hr = dx_device_queue(dev, &q);
    if (FAILED(hr)) fail("dx_device_queue", hr);

    // ---- elementwise Mul on two float arrays ----
    constexpr uint32_t N = 8;
    std::vector<float> a(N), b(N);
    for (uint32_t i = 0; i < N; ++i) { a[i] = (float)i + 1.0f; b[i] = 2.0f; }

    dx_buffer* ba = nullptr;
    dx_buffer* bb = nullptr;
    dx_buffer* bout = nullptr;
    hr = dx_create_buffer(dev, (uint64_t)N * 4, 2, &ba); // Upload
    if (FAILED(hr)) fail("create buffer a", hr);
    hr = dx_create_buffer(dev, (uint64_t)N * 4, 2, &bb);
    if (FAILED(hr)) fail("create buffer b", hr);
    hr = dx_create_buffer(dev, (uint64_t)N * 4, 3, &bout); // Readback
    if (FAILED(hr)) fail("create buffer out", hr);

    hr = dx_upload(dev, ba, 0, a.data(), (uint64_t)N * 4);
    if (FAILED(hr)) fail("dx_upload a", hr);
    hr = dx_upload(dev, bb, 0, b.data(), (uint64_t)N * 4);
    if (FAILED(hr)) fail("dx_upload b", hr);

    hr = dx_la_elementwise(dev, q, bout, ba, bb, N, 2, 1.0f, 0.0f); // op 2 = Mul
    if (FAILED(hr)) fail("dx_la_elementwise", hr);

    std::vector<float> out(N);
    hr = dx_download(dev, bout, 0, out.data(), (uint64_t)N * 4);
    if (FAILED(hr)) fail("dx_download", hr);

    printf("\nelementwise Mul (a*2):\n");
    for (uint32_t i = 0; i < N; ++i)
        printf("  a[%u]=%.1f * 2.0 = %.1f\n", i, a[i], out[i]);

    // ---- gemm_f16_dot2, 16x16x16 ----
    constexpr uint32_t M = 16, Nn = 16, K = 16;
    std::vector<uint16_t> af(M * K), bf(K * Nn);
    for (uint32_t r = 0; r < M; ++r)
        for (uint32_t k = 0; k < K; ++k) af[r * K + k] = f32_to_f16(0.25f * (float)r + 0.1f * (float)k);
    for (uint32_t k = 0; k < K; ++k)
        for (uint32_t c = 0; c < Nn; ++c) bf[k * Nn + c] = f32_to_f16(0.5f * (float)(k + 1));

    std::vector<uint32_t> apack(M * K / 2), bpack(K * Nn / 2);
    for (uint32_t i = 0; i < M * K / 2; ++i) apack[i] = (uint32_t)af[2 * i] | ((uint32_t)af[2 * i + 1] << 16);
    for (uint32_t c = 0; c < Nn; ++c)
        for (uint32_t k2 = 0; k2 < K / 2; ++k2)
            bpack[c * (K / 2) + k2] = (uint32_t)bf[(2 * k2) * Nn + c] | ((uint32_t)bf[(2 * k2 + 1) * Nn + c] << 16);

    dx_buffer* ga = nullptr;
    dx_buffer* gb = nullptr;
    dx_buffer* gout = nullptr;
    hr = dx_create_buffer(dev, (uint64_t)apack.size() * 4, 2, &ga);
    if (FAILED(hr)) fail("create gemm a", hr);
    hr = dx_create_buffer(dev, (uint64_t)bpack.size() * 4, 2, &gb);
    if (FAILED(hr)) fail("create gemm b", hr);
    hr = dx_create_buffer(dev, (uint64_t)M * Nn * 4, 3, &gout);
    if (FAILED(hr)) fail("create gemm out", hr);

    hr = dx_upload(dev, ga, 0, apack.data(), (uint64_t)apack.size() * 4);
    if (FAILED(hr)) fail("dx_upload ga", hr);
    hr = dx_upload(dev, gb, 0, bpack.data(), (uint64_t)bpack.size() * 4);
    if (FAILED(hr)) fail("dx_upload gb", hr);

    hr = dx_la_gemm_f16_dot2(dev, q, gout, ga, gb, M, Nn, K);
    if (FAILED(hr)) fail("dx_la_gemm_f16_dot2", hr);

    std::vector<float> gres(M * Nn);
    hr = dx_download(dev, gout, 0, gres.data(), (uint64_t)M * Nn * 4);
    if (FAILED(hr)) fail("download gemm", hr);
    printf("\ngemm_f16_dot2 C[0][0] = %.4f\n", gres[0]);

    dx_destroy_buffer(ba);
    dx_destroy_buffer(bb);
    dx_destroy_buffer(bout);
    dx_destroy_buffer(ga);
    dx_destroy_buffer(gb);
    dx_destroy_buffer(gout);
    dx_destroy_queue(q);
    dx_destroy_device(dev);

    printf("\nplayground OK\n");
    return 0;
}
