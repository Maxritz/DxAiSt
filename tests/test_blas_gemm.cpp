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
    return ok ? 0 : 1;
}
