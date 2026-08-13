#include "dxait/dxait.hpp"
#include "dxait/dxblas.hpp"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

int main() {
    printf("DXAiT CUTLASS-style Tiled GEMM Test\n");
    printf("===================================\n\n");

    auto adapters = dxait::Adapter::enumerate();
    if (adapters.empty()) { printf("No GPU found, skipping test.\n"); return 0; }
    auto device = dxait::Adapter::create_device(0);

    constexpr uint32_t M = 32, N = 32, K = 32;
    constexpr uint64_t a_bytes = (uint64_t)M * K * sizeof(float);
    constexpr uint64_t b_bytes = (uint64_t)K * N * sizeof(float);
    constexpr uint64_t c_bytes = (uint64_t)M * N * sizeof(float);

    // A[row][k] = row + 0.1*k,  B[k][col] = 0.5*(k+1)
    std::vector<float> a(M * K), b(K * N);
    for (uint32_t r = 0; r < M; ++r)
        for (uint32_t k = 0; k < K; ++k) a[r * K + k] = (float)r + 0.1f * (float)k;
    for (uint32_t k = 0; k < K; ++k)
        for (uint32_t c = 0; c < N; ++c) b[k * N + c] = 0.5f * (float)(k + 1);

    // CPU reference
    std::vector<float> ref(M * N, 0.0f);
    for (uint32_t r = 0; r < M; ++r)
        for (uint32_t c = 0; c < N; ++c) {
            float s = 0.0f;
            for (uint32_t k = 0; k < K; ++k) s += a[r * K + k] * b[k * N + c];
            ref[r * N + c] = s;
        }

    auto a_up = device->create_buffer(a_bytes, dxait::MemLocation::Upload);
    auto b_up = device->create_buffer(b_bytes, dxait::MemLocation::Upload);
    auto c_rb = device->create_buffer(c_bytes, dxait::MemLocation::Readback);

    std::memcpy(a_up->map(), a.data(), a_bytes);
    a_up->unmap();
    std::memcpy(b_up->map(), b.data(), b_bytes);
    b_up->unmap();

    dxait::BLAS blas(device.get());
    auto q = device->create_queue(dxait::QueueType::Compute);
    auto fence = device->create_fence(0);

    printf("Dispatching tiled GEMM %ux%ux%u (tile 16x16)...\n", M, N, K);
    blas.gemm(q.get(), c_rb.get(), a_up.get(), b_up.get(), M, N, K);
    q->signal(*fence, 1);
    fence->wait(1);

    float* out = (float*)c_rb->map();
    float max_err = 0.0f;
    bool ok = true;
    for (uint32_t r = 0; r < M; ++r) {
        for (uint32_t c = 0; c < N; ++c) {
            float err = std::fabs(out[r * N + c] - ref[r * N + c]);
            if (err > max_err) max_err = err;
            if (err > 1e-2f) {
                printf("  mismatch[%u][%u] got %.4f expected %.4f\n", r, c, out[r * N + c], ref[r * N + c]);
                ok = false;
            }
        }
    }
    c_rb->unmap();

    printf("  C[0][0] = %.4f (ref %.4f)\n", ((float*)out)[0], ref[0]);
    printf("  max abs error = %.6f\n", max_err);
    printf("\nResult: %s\n", ok ? "Tiled GEMM PASSED" : "FAILED");

    // ---- F16 dot2add GEMM ----
    auto f32_to_f16 = [](float x) -> uint16_t {
        uint32_t u; std::memcpy(&u, &x, 4);
        uint32_t sign = (u >> 16) & 0x8000u;
        int32_t exp = (int32_t)((u >> 23) & 0xFFu) - 127 + 15;
        uint32_t mant = u & 0x7FFFFFu;
        if (exp >= 31) return (uint16_t)(sign | 0x7C00u); // inf
        if (exp <= 0) { // subnormal
            mant |= 0x800000u;
            int32_t shift = 1 - exp;
            if (shift >= 24) return (uint16_t)sign;
            return (uint16_t)(sign | (mant >> shift));
        }
        uint16_t h = (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
        // round to nearest even
        if ((mant & 0x1000u) && ((mant & 0x2000u) || (mant & 0xFFFu))) h++;
        return h;
    };

    constexpr uint32_t M2 = 32, N2 = 32, K2 = 32;
    std::vector<uint16_t> af(K2 * K2), bf(K2 * K2);
    for (uint32_t r = 0; r < M2; ++r)
        for (uint32_t k = 0; k < K2; ++k) af[r * K2 + k] = f32_to_f16((float)r + 0.1f * (float)k);
    for (uint32_t k = 0; k < K2; ++k)
        for (uint32_t c = 0; c < N2; ++c) bf[k * N2 + c] = f32_to_f16(0.5f * (float)(k + 1));

    // pack halves per uint (little-endian half pairs)
    std::vector<uint32_t> apack(K2 * K2 / 2), bpack(K2 * N2 / 2);
    for (uint32_t i = 0; i < K2 * K2 / 2; ++i)
        apack[i] = (uint32_t)af[2 * i] | ((uint32_t)af[2 * i + 1] << 16);
    for (uint32_t i = 0; i < K2 * N2 / 2; ++i)
        bpack[i] = (uint32_t)bf[2 * i] | ((uint32_t)bf[2 * i + 1] << 16);

    auto f16_to_f32 = [](uint16_t h) -> float {
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
    };

    // CPU ref in fp32 from fp16 values
    std::vector<float> ref2(M2 * N2, 0.0f);
    for (uint32_t r = 0; r < M2; ++r)
        for (uint32_t c = 0; c < N2; ++c) {
            float s = 0.0f;
            for (uint32_t k = 0; k < K2; ++k) s += f16_to_f32(af[r * K2 + k]) * f16_to_f32(bf[k * N2 + c]);
            ref2[r * N2 + c] = s;
        }

    auto a_up2 = device->create_buffer(apack.size() * 4, dxait::MemLocation::Upload);
    auto b_up2 = device->create_buffer(bpack.size() * 4, dxait::MemLocation::Upload);
    auto c_rb2 = device->create_buffer(c_bytes, dxait::MemLocation::Readback);
    std::memcpy(a_up2->map(), apack.data(), apack.size() * 4); a_up2->unmap();
    std::memcpy(b_up2->map(), bpack.data(), bpack.size() * 4); b_up2->unmap();

    printf("\nDispatching F16 dot2add GEMM %ux%ux%u...\n", M2, N2, K2);
    blas.gemm_f16_dot2(q.get(), c_rb2.get(), a_up2.get(), b_up2.get(), M2, N2, K2);
    q->signal(*fence, 2);
    fence->wait(2);

    float* o2 = (float*)c_rb2->map();
    float max_err2 = 0.0f;
    bool ok2 = true;
    for (uint32_t r = 0; r < M2; ++r)
        for (uint32_t c = 0; c < N2; ++c) {
            float err = std::fabs(o2[r * N2 + c] - ref2[r * N2 + c]);
            if (err > max_err2) max_err2 = err;
            // fp16 input quantization + fp16 product rounding in dot2add:
            // allow ~5% relative (llama accepts same precision path).
            float tol = 0.05f * (std::max)(std::fabs(ref2[r * N2 + c]), 1.0f) + 1.0f;
            if (err > tol) { printf("  f16 mismatch[%u][%u] got %.2f ref %.2f\n", r, c, o2[r * N2 + c], ref2[r * N2 + c]); ok2 = false; }
        }
    c_rb2->unmap();
    printf("  F16 max abs error = %.6f\n", max_err2);
    printf("Result: %s\n", ok2 ? "F16 dot2add GEMM PASSED" : "FAILED");

    return (ok && ok2) ? 0 : 1;
}
