#include "dxait/dxait.hpp"
#include "dxait/dxmodel.hpp"
#include "dxait/dxchunk.hpp"
#include "dxait/dxshard.hpp"
#include "dxait/dxmath.hpp"
#include <iostream>
#include <vector>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iomanip>

int main() {
    std::cout << "========================================================\n";
    std::cout << " DXAiT Split-Load Multi-Layer Inference Harness Test\n";
    std::cout << "========================================================\n\n";

    auto adapters = dxait::Adapter::enumerate();
    if (adapters.empty()) {
        std::cout << "No GPU found, skipping test.\n";
        return 0;
    }

    auto device = dxait::Adapter::create_device(0);
    const auto& caps = device->caps();

    std::cout << "Target GPU:              " << caps.name << "\n";
    std::cout << "Dedicated VRAM Capacity:  " << caps.dedicated_video_memory / (1024 * 1024) << " MB\n";
    std::cout << "Shared System RAM:       " << caps.shared_system_memory / (1024 * 1024) << " MB\n\n";

    constexpr uint32_t num_layers = 16;
    constexpr uint32_t num_rows = 4;
    constexpr uint32_t row_dim = 2048;
    constexpr uint64_t tensor_bytes = num_rows * row_dim * sizeof(float); // 32 KB per layer tensor

    dxait::OffloadConfig config{0.5f, 0.5f, tensor_bytes * num_layers};
    dxait::OffloadPartitionEngine offloader(device.get(), config);

    std::cout << "1. Sharding " << num_layers << " Transformer Layers (50% VRAM / 50% System RAM)...\n";
    std::vector<std::unique_ptr<dxait::Buffer>> layer_tensors;
    std::vector<dxait::MemLocation> layer_locations;

    // Allocate 1 VRAM staging slot for streaming System RAM layers into VRAM
    auto vram_staging = device->create_buffer(tensor_bytes, dxait::MemLocation::Default);

    for (uint32_t l = 0; l < num_layers; ++l) {
        dxait::MemLocation loc = (l < num_layers / 2) ? dxait::MemLocation::Default : dxait::MemLocation::Upload;
        layer_locations.push_back(loc);

        auto upload_buf = device->create_buffer(tensor_bytes, dxait::MemLocation::Upload);
        std::vector<float> h_data(num_rows * row_dim, static_cast<float>(l + 1) * 0.1f);
        std::memcpy(upload_buf->map(), h_data.data(), tensor_bytes);
        upload_buf->unmap();

        if (loc == dxait::MemLocation::Default) {
            auto vram_buf = device->create_buffer(tensor_bytes, dxait::MemLocation::Default);
            auto queue = device->create_queue(dxait::QueueType::Copy);
            offloader.page_swap(queue.get(), upload_buf.get(), vram_buf.get(), tensor_bytes);
            layer_tensors.push_back(std::move(vram_buf));
        } else {
            layer_tensors.push_back(std::move(upload_buf));
        }
    }

    std::cout << "   Successfully allocated and sharded 16 transformer layer tensors!\n\n";

    // 2. Prepare Activations & Work Queues
    auto act_input = device->create_buffer(tensor_bytes, dxait::MemLocation::Upload);
    auto act_output = device->create_buffer(tensor_bytes, dxait::MemLocation::Default);
    auto act_readback = device->create_buffer(tensor_bytes, dxait::MemLocation::Readback);

    std::vector<float> h_in(num_rows * row_dim, 1.0f);
    std::memcpy(act_input->map(), h_in.data(), tensor_bytes);
    act_input->unmap();

    dxait::MathOps math(device.get());
    auto compute_queue = device->create_queue(dxait::QueueType::Compute);
    auto copy_queue = device->create_queue(dxait::QueueType::Copy);

    std::cout << "2. Executing 16-Layer Split-Load Transformer Forward Pass...\n";
    auto t0 = std::chrono::high_resolution_clock::now();

    for (uint32_t l = 0; l < num_layers; ++l) {
        auto layer_t0 = std::chrono::high_resolution_clock::now();

        dxait::Buffer* active_tensor = nullptr;
        if (layer_locations[l] == dxait::MemLocation::Default) {
            active_tensor = layer_tensors[l].get();
        } else {
            // System RAM layer -> PCIe DMA page-swap into VRAM staging slot
            offloader.page_swap(copy_queue.get(), layer_tensors[l].get(), vram_staging.get(), tensor_bytes);
            active_tensor = vram_staging.get();
        }

        // Dispatch RMSNorm activation normalization
        math.rms_norm(compute_queue.get(), act_output.get(), act_input.get(), active_tensor, num_rows, row_dim);

        // Dispatch Softmax activation
            math.softmax(compute_queue.get(), act_output.get(), act_output.get(), num_rows, row_dim, 1.0f);

        auto layer_t1 = std::chrono::high_resolution_clock::now();
        double layer_ms = std::chrono::duration<double, std::milli>(layer_t1 - layer_t0).count();

        std::cout << "   [Layer " << std::setw(2) << l + 1 << "/16] Location: "
                  << ((layer_locations[l] == dxait::MemLocation::Default) ? "Dedicated VRAM" : "Shared System RAM")
                  << " | Compute Latency: " << std::fixed << std::setprecision(3) << layer_ms << " ms\n";
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Copy final output activation to readback buffer
    offloader.page_swap(copy_queue.get(), act_output.get(), act_readback.get(), tensor_bytes);

    float* out_ptr = static_cast<float*>(act_readback->map());
    assert(out_ptr != nullptr);
    assert(!std::isnan(out_ptr[0]));
    assert(!std::isinf(out_ptr[0]));
    std::cout << "\n   Sample Output Activation Token Vector [0..3]: "
              << out_ptr[0] << ", " << out_ptr[1] << ", " << out_ptr[2] << ", " << out_ptr[3] << "\n";
    act_readback->unmap();

    std::cout << "\n========================================================\n";
    std::cout << " Inference Harness Performance Summary:\n";
    std::cout << "   Total Layers Evaluated: 16 (8 VRAM + 8 System RAM)\n";
    std::cout << "   Total Forward Pass:     " << std::fixed << std::setprecision(2) << total_ms << " ms\n";
    std::cout << "   Avg Latency per Layer:  " << std::fixed << std::setprecision(3) << (total_ms / num_layers) << " ms\n";
    std::cout << "========================================================\n";
    std::cout << " Split-Load Model Inference Harness Test PASSED!\n";

    return 0;
}
