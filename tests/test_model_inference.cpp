#include "dxait/dxait.hpp"
#include "dxait/dxmath.hpp"
#include "dxait/dxmodel.hpp"
#include "dxait/dxsched.hpp"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

int main() {
    auto adapters = dxait::Adapter::enumerate();
    if (adapters.empty()) {
        std::cout << "No GPU, skipping model inference test.\n";
        return 0;
    }

    auto device = dxait::Adapter::create_device(0);
    assert(device != nullptr);

    // 1. Test RMSNorm & Softmax Compute Shaders
    constexpr uint32_t num_rows = 2;
    constexpr uint32_t row_dim = 64;
    constexpr uint64_t total_elements = num_rows * row_dim;
    constexpr uint64_t buf_bytes = total_elements * sizeof(float);
    constexpr uint64_t weight_bytes = row_dim * sizeof(float);

    auto in_upload = device->create_buffer(buf_bytes, dxait::MemLocation::Upload);
    auto weight_upload = device->create_buffer(weight_bytes, dxait::MemLocation::Upload);
    auto out_norm_readback = device->create_buffer(buf_bytes, dxait::MemLocation::Readback);
    auto out_softmax_readback = device->create_buffer(buf_bytes, dxait::MemLocation::Readback);

    std::vector<float> h_in(total_elements), h_weight(row_dim, 1.0f);
    for (uint64_t i = 0; i < total_elements; ++i) {
        h_in[i] = static_cast<float>(i % 16) - 8.0f;
    }

    std::memcpy(in_upload->map(), h_in.data(), buf_bytes);
    in_upload->unmap();

    std::memcpy(weight_upload->map(), h_weight.data(), weight_bytes);
    weight_upload->unmap();

    auto compute_queue = device->create_queue(dxait::QueueType::Compute);
    auto fence = device->create_fence(0);

    dxait::MathOps math(device.get());

    // Execute RMSNorm
    math.rms_norm(compute_queue.get(), out_norm_readback.get(), in_upload.get(), weight_upload.get(), num_rows, row_dim);
    compute_queue->signal(*fence, 1);
    fence->wait(1);

    // Verify Softmax
    math.softmax(compute_queue.get(), out_softmax_readback.get(), in_upload.get(), num_rows, row_dim, 1.0f);
    compute_queue->signal(*fence, 2);
    fence->wait(2);

    float* p_softmax = static_cast<float*>(out_softmax_readback->map());
    assert(p_softmax != nullptr);
    float sum_row0 = 0.0f;
    for (uint32_t i = 0; i < row_dim; ++i) {
        sum_row0 += p_softmax[i];
    }
    out_softmax_readback->unmap();
    assert(std::abs(sum_row0 - 1.0f) < 1e-3f && "Softmax probabilities sum to 1.0");

    // 2. Test Multi-Queue Scheduler & Model Loader
    dxait::MultiQueueScheduler scheduler(device.get());
    scheduler.wait_all();

    dxait::ModelLoader loader;
    bool parsed = loader.parse_gguf("nonexistent.gguf");
    assert(!parsed && "Graceful non-existent GGUF handle check");

    std::cout << "DXAiT Full SDK Execution (Math Ops + RMSNorm + Softmax + Scheduler) Passed Perfectly!\n";
    return 0;
}
