#include "dxait/dxait.hpp"
#include "dxait/dxmodel.hpp"
#include "dxait/dxchunk.hpp"
#include "dxait/dxquant.hpp"
#include "dxait/dxmath.hpp"
#include "dxait/dxattention.hpp"
#include "dxait/dxblas.hpp"
#include <iostream>
#include <chrono>
#include <vector>
#include <cassert>
#include <iomanip>

int main() {
    std::cout << "========================================================\n";
    std::cout << " DXAiT Microsecond GPU Compute & Loader Profiler Benchmark\n";
    std::cout << "========================================================\n\n";

    auto adapters = dxait::Adapter::enumerate();
    if (adapters.empty()) {
        std::cout << "No D3D12 hardware GPU adapter found, exiting.\n";
        return 0;
    }

    auto device = dxait::Adapter::create_device(0);
    const auto& caps = device->caps();
    std::cout << "Target GPU: " << caps.name << " (" << (caps.arch_family == dxait::ArchitectureFamily::AMD_RDNA4 ? "RDNA4" : "RDNA2") << ")\n";
    std::cout << "Preferred Wave Size: " << caps.preferred_wave_size << " lanes\n\n";

    auto queue = device->create_queue(dxait::QueueType::Compute);
    auto fence = device->create_fence(0);

    // 1. Benchmark GPU Vector Add & Memory Allocation
    constexpr uint32_t N = 1048576; // 1M floats = 4MB
    constexpr uint64_t bytes = N * sizeof(float);

    auto in1 = device->create_buffer(bytes, dxait::MemLocation::Upload);
    auto in2 = device->create_buffer(bytes, dxait::MemLocation::Upload);
    auto out = device->create_buffer(bytes, dxait::MemLocation::Readback);

    dxait::BLAS blas(device.get());

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < 100; ++iter) {
        blas.vec_add(queue.get(), out.get(), in1.get(), in2.get(), N);
    }
    queue->signal(*fence, 1);
    fence->wait(1);
    auto t1 = std::chrono::high_resolution_clock::now();

    double avg_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / 100.0;
    double bandwidth_gbs = (3.0 * bytes) / (avg_us * 1e-6) / 1e9;

    std::cout << "1. Vector Add (1M floats):          " << std::fixed << std::setprecision(2) << avg_us << " us | " << bandwidth_gbs << " GB/s\n";

    // 2. Benchmark RMSNorm Compute Kernel
    dxait::MathOps math(device.get());
    auto weight = device->create_buffer(4096 * sizeof(float), dxait::MemLocation::Upload);

    t0 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < 100; ++iter) {
        math.rms_norm(queue.get(), out.get(), in1.get(), weight.get(), 256, 4096);
    }
    queue->signal(*fence, 2);
    fence->wait(2);
    t1 = std::chrono::high_resolution_clock::now();

    avg_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / 100.0;
    std::cout << "2. RMSNorm Compute (256x4096):       " << avg_us << " us\n";

    // 3. Benchmark FlashAttention-2 SDPA Compute Kernel
    dxait::AttentionOps attn(device.get());
    t0 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < 50; ++iter) {
        attn.scaled_dot_product_attention(queue.get(), out.get(), in1.get(), in2.get(), weight.get(), 256, 128, 0.125f);
    }
    queue->signal(*fence, 3);
    fence->wait(3);
    t1 = std::chrono::high_resolution_clock::now();

    avg_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / 50.0;
    std::cout << "3. SDPA FlashAttention-2 (256x128):  " << avg_us << " us\n";

    // 4. Benchmark Q4_0 GPU Dequantization Kernel
    dxait::QuantOps quant(device.get());
    uint32_t num_q4_blocks = 32768; // 1M elements quantized
    auto q4_in = device->create_buffer(num_q4_blocks * sizeof(dxait::BlockQ4_0), dxait::MemLocation::Upload);

    t0 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < 100; ++iter) {
        quant.dequantize_q4_0_gpu(queue.get(), out.get(), q4_in.get(), num_q4_blocks);
    }
    queue->signal(*fence, 4);
    fence->wait(4);
    t1 = std::chrono::high_resolution_clock::now();

    avg_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / 100.0;
    double dequant_gbs = (num_q4_blocks * sizeof(dxait::BlockQ4_0) + bytes) / (avg_us * 1e-6) / 1e9;

    std::cout << "4. Q4_0 GPU Dequantization (1M elems):" << avg_us << " us | " << dequant_gbs << " GB/s\n";
    std::cout << "========================================================\n";
    std::cout << " All GPU Compute Kernel Perf Traces Executed Successfully!\n";

    return 0;
}
