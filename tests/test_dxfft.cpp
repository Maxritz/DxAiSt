#include "dxait/dxait.hpp"
#include "dxait/dxfft.hpp"
#include <cstdio>
#include <cstring>
#include <cmath>

int main() {
    printf("DXAiT Radix2 FFT Verification Suite\n");
    printf("===================================\n\n");

    auto adapters = dxait::Adapter::enumerate();
    if (adapters.empty()) { printf("No GPU found, skipping test.\n"); return 0; }
    auto device = dxait::Adapter::create_device(0);

    constexpr uint32_t n = 8;
    constexpr uint64_t bytes = n * sizeof(float);

    auto in_r = device->create_buffer(bytes, dxait::MemLocation::Upload);
    auto in_i = device->create_buffer(bytes, dxait::MemLocation::Upload);
    auto out_r = device->create_buffer(bytes, dxait::MemLocation::Readback);
    auto out_i = device->create_buffer(bytes, dxait::MemLocation::Readback);

    // Impulse x[0]=1, rest 0. FFT of impulse = all ones.
    float* pr = (float*)in_r->map();
    float* pi = (float*)in_i->map();
    for (uint32_t k = 0; k < n; ++k) { pr[k] = (k == 0) ? 1.0f : 0.0f; pi[k] = 0.0f; }
    in_r->unmap(); in_i->unmap();

    dxait::FFTOps fft(device.get());
    auto queue = device->create_queue(dxait::QueueType::Compute);
    auto fence = device->create_fence(0);

    printf("Running FFT on impulse signal (n=%u)...\n", n);
    fft.fft_1d_radix2(queue.get(), out_r.get(), out_i.get(), in_r.get(), in_i.get(), n);
    queue->signal(*fence, 1);
    fence->wait(1);

    float* or_ = (float*)out_r->map();
    float* oi = (float*)out_i->map();

    bool ok = true;
    printf("  Output (real, imag):\n");
    for (uint32_t k = 0; k < n; ++k) {
        printf("    [%u] (%.4f, %.4f)\n", k, or_[k], oi[k]);
        // impulse FFT: real all 1.0, imag all 0.0
        if (std::fabs(or_[k] - 1.0f) > 1e-3f || std::fabs(oi[k]) > 1e-3f) ok = false;
    }
    out_r->unmap(); out_i->unmap();

    printf("\nResult: %s\n", ok ? "FFT impulse test PASSED (all ones, imag zero)" : "FAILED");
    return ok ? 0 : 1;
}
