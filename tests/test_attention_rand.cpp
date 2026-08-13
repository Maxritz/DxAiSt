#include "dxait/dxait.hpp"
#include "dxait/dxrand.hpp"
#include "dxait/dxattention.hpp"
#include <iostream>
#include <vector>
#include <cassert>

int main() {
    auto adapters = dxait::Adapter::enumerate();
    if (adapters.empty()) {
        std::cout << "No GPU, skipping attention and rand test.\n";
        return 0;
    }

    auto device = dxait::Adapter::create_device(0);
    assert(device != nullptr);

    constexpr uint32_t N = 256;
    constexpr uint64_t bytes = N * sizeof(float);

    auto rand_out_readback = device->create_buffer(bytes, dxait::MemLocation::Readback);
    auto compute_queue = device->create_queue(dxait::QueueType::Compute);
    auto fence = device->create_fence(0);

    // 1. Test PRNG
    dxait::RandomGenerator rng(device.get());
    rng.fill_uniform(compute_queue.get(), rand_out_readback.get(), N, 42);

    compute_queue->signal(*fence, 1);
    fence->wait(1);

    float* p_rand = static_cast<float*>(rand_out_readback->map());
    assert(p_rand != nullptr);
    bool valid_rand = true;
    for (uint32_t i = 0; i < N; ++i) {
        if (p_rand[i] < 0.0f || p_rand[i] > 1.0f) {
            valid_rand = false;
            break;
        }
    }
    rand_out_readback->unmap();
    assert(valid_rand && "PCG32 PRNG values in [0, 1] range");
    std::cout << "PCG32 GPU Random Generator Test PASSED\n";

    // 2. Test Attention (SDPA)
    dxait::AttentionOps attn(device.get());
    auto q_buf = device->create_buffer(bytes, dxait::MemLocation::Upload);
    auto k_buf = device->create_buffer(bytes, dxait::MemLocation::Upload);
    auto v_buf = device->create_buffer(bytes, dxait::MemLocation::Upload);
    auto attn_out = device->create_buffer(bytes, dxait::MemLocation::Readback);

    attn.scaled_dot_product_attention(compute_queue.get(), attn_out.get(), q_buf.get(), k_buf.get(), v_buf.get(), 16, 16, 0.25f);
    compute_queue->signal(*fence, 2);
    fence->wait(2);

    std::cout << "DXAiT Attention SDPA + PRNG Test Passed!\n";
    return 0;
}
