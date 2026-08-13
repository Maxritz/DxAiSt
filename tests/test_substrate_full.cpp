#include "dxait/dxait.hpp"
#include "dxait/dxbarrier.hpp"
#include "dxait/dxgraph.hpp"
#include "dxait/dxquant.hpp"
#include "dxait/dxkv.hpp"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

int main() {
    auto adapters = dxait::Adapter::enumerate();
    if (adapters.empty()) {
        std::cout << "No GPU, skipping substrate full test.\n";
        return 0;
    }

    auto device = dxait::Adapter::create_device(0);
    assert(device != nullptr);

    // 1. Test Quantization (Q8_0 roundtrip)
    constexpr uint64_t K = 256;
    std::vector<float> orig_weights(K);
    for (uint64_t i = 0; i < K; ++i) {
        orig_weights[i] = std::sin(static_cast<float>(i)) * 5.0f;
    }

    std::vector<dxait::BlockQ8_0> blocks(K / 32);
    dxait::quantize_row_q8_0(orig_weights.data(), blocks.data(), K);

    std::vector<float> dequant_weights(K);
    dxait::dequantize_row_q8_0(blocks.data(), dequant_weights.data(), K);

    float max_diff = 0.0f;
    for (uint64_t i = 0; i < K; ++i) {
        max_diff = (std::max)(max_diff, std::abs(orig_weights[i] - dequant_weights[i]));
    }
    assert(max_diff < 0.1f && "Q8_0 Quantization accuracy check passed");
    std::cout << "Q8_0 Quantization roundtrip max error: " << max_diff << " PASSED\n";

    // 2. Test KV Cache Manager
    dxait::KVCacheConfig kv_cfg{8, 16, 64, 512, 16};
    dxait::KVCacheManager kv_mgr(device.get(), kv_cfg);
    uint32_t seq1 = kv_mgr.allocate_sequence();
    assert(seq1 != 0xFFFFFFFF && "KV Cache allocation passed");
    kv_mgr.free_sequence(seq1);

    // 3. Test Command Graph & Barrier Batching
    dxait::CommandGraph graph(device.get());
    auto buf1 = device->create_buffer(1024, dxait::MemLocation::Default);
    
    graph.add_node("barrier_node", [buf1 = buf1->get()](ID3D12GraphicsCommandList* cmd_list) {
        dxait::BarrierBatch barriers;
        barriers.add_uav(buf1);
        barriers.flush(cmd_list);
    });

    graph.compile();
    auto queue = device->create_queue(dxait::QueueType::Direct);
    auto fence = device->create_fence(0);
    graph.execute(queue.get());

    queue->signal(*fence, 1);
    fence->wait(1);
    assert(fence->is_completed(1));

    std::cout << "DXAiT Full Substrate (Graph + Quant + KV + Barriers) Test Passed!\n";
    return 0;
}
