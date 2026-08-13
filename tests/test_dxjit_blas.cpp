#include "dxait/dxait.hpp"
#include "dxait/dxblas.hpp"
#include <iostream>
#include <vector>
#include <cassert>

int main() {
    auto adapters = dxait::Adapter::enumerate();
    if (adapters.empty()) {
        std::cout << "No D3D12 hardware GPU found, skipping BLAS test.\n";
        return 0;
    }

    auto device = dxait::Adapter::create_device(0);
    assert(device != nullptr);

    constexpr uint32_t N = 1024;
    constexpr uint64_t buf_size = N * sizeof(float);

    // Create Buffers: 2 Uploads, 1 Readback
    auto in1_upload = device->create_buffer(buf_size, dxait::MemLocation::Upload);
    auto in2_upload = device->create_buffer(buf_size, dxait::MemLocation::Upload);
    auto out_readback = device->create_buffer(buf_size, dxait::MemLocation::Readback);

    std::vector<float> h_in1(N), h_in2(N);
    for (uint32_t i = 0; i < N; ++i) {
        h_in1[i] = static_cast<float>(i) * 1.5f;
        h_in2[i] = static_cast<float>(i) * 2.5f;
    }

    void* p1 = in1_upload->map();
    std::memcpy(p1, h_in1.data(), buf_size);
    in1_upload->unmap();

    void* p2 = in2_upload->map();
    std::memcpy(p2, h_in2.data(), buf_size);
    in2_upload->unmap();

    auto compute_queue = device->create_queue(dxait::QueueType::Compute);
    auto fence = device->create_fence(0);

    dxait::BLAS blas(device.get());

    // Execute Vector Add compute shader: out = 2.0 * in1 + 1.0 * in2
    blas.vec_add(compute_queue.get(), out_readback.get(), in1_upload.get(), in2_upload.get(), N, 2.0f, 1.0f);

    compute_queue->signal(*fence, 1);
    fence->wait(1);

    // Verify GPU results on CPU
    float* p_out = static_cast<float*>(out_readback->map());
    assert(p_out != nullptr);

    bool passed = true;
    for (uint32_t i = 0; i < N; ++i) {
        float expected = 2.0f * h_in1[i] + 1.0f * h_in2[i];
        if (std::abs(p_out[i] - expected) > 1e-3f) {
            std::cerr << "Mismatch at index " << i << ": got " << p_out[i] << ", expected " << expected << "\n";
            passed = false;
            break;
        }
    }
    out_readback->unmap();

    assert(passed && "GPU Compute Vector Add Verification Failed");
    std::cout << "DXAiT JIT + BLAS Vector Add Compute Dispatch Passed Perfectly!\n";
    return 0;
}
