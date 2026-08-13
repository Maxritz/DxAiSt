#include "dxait/dxait.hpp"
#include "dxait/dxquant.hpp"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

int main() {
    constexpr uint64_t K = 256;
    std::vector<float> orig_weights(K);
    for (uint64_t i = 0; i < K; ++i) {
        orig_weights[i] = std::cos(static_cast<float>(i)) * 4.0f;
    }

    // 1. CPU Q4_0 Quantize & Dequantize
    std::vector<dxait::BlockQ4_0> blocks(K / 32);
    dxait::quantize_row_q4_0(orig_weights.data(), blocks.data(), K);

    std::vector<float> dequant_weights(K);
    dxait::dequantize_row_q4_0(blocks.data(), dequant_weights.data(), K);

    float max_diff = 0.0f;
    for (uint64_t i = 0; i < K; ++i) {
        max_diff = (std::max)(max_diff, std::abs(orig_weights[i] - dequant_weights[i]));
    }
    assert(max_diff < 0.5f && "Q4_0 Quantization accuracy check passed");
    std::cout << "Q4_0 Quantization roundtrip max error: " << max_diff << " PASSED\n";

    // 2. GPU Dequantize Q4_0
    auto adapters = dxait::Adapter::enumerate();
    if (!adapters.empty()) {
        auto device = dxait::Adapter::create_device(0);
        assert(device != nullptr);

        uint64_t blocks_bytes = blocks.size() * sizeof(dxait::BlockQ4_0);
        uint64_t float_bytes = K * sizeof(float);

        auto in_q4_upload = device->create_buffer(blocks_bytes, dxait::MemLocation::Upload);
        auto out_float_readback = device->create_buffer(float_bytes, dxait::MemLocation::Readback);

        std::memcpy(in_q4_upload->map(), blocks.data(), blocks_bytes);
        in_q4_upload->unmap();

        auto compute_queue = device->create_queue(dxait::QueueType::Compute);
        auto fence = device->create_fence(0);

        dxait::QuantOps quant_ops(device.get());
        quant_ops.dequantize_q4_0_gpu(compute_queue.get(), out_float_readback.get(), in_q4_upload.get(), static_cast<uint32_t>(blocks.size()));

        compute_queue->signal(*fence, 1);
        fence->wait(1);

        std::cout << "Q4_0 GPU Dequantization Compute Dispatch PASSED\n";
    }

    return 0;
}
