#include "dxait/dxait.hpp"
#include "dxait/dxmodel.hpp"
#include "dxait/dxchunk.hpp"
#include "dxait/dxblas.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include <cstring>

int main() {
    std::cout << "========================================================\n";
    std::cout << " DXAiT Real End-to-End Model Loading & Streaming Test\n";
    std::cout << "========================================================\n\n";

    // 1. Create a binary model weight file payload
    const std::string model_path = "real_test_weights.bin";
    constexpr uint32_t N = 512;
    constexpr uint64_t tensor_bytes = N * sizeof(float);

    std::vector<float> orig_weights(N);
    for (uint32_t i = 0; i < N; ++i) {
        orig_weights[i] = static_cast<float>(i) * 0.5f + 1.0f;
    }

    {
        std::ofstream out(model_path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(orig_weights.data()), tensor_bytes);
    }

    // 2. Initialize DXAiT D3D12 Device & Streamer
    auto adapters = dxait::Adapter::enumerate();
    if (adapters.empty()) {
        std::cout << "No D3D12 GPU found, skipping GPU execution test.\n";
        return 0;
    }

    auto device = dxait::Adapter::create_device(0);
    assert(device != nullptr && "Device creation succeeded");

    // 3. Register and Stream Weight Chunk directly into GPU VRAM Upload Buffer
    dxait::ChunkStreamer streamer(device.get());
    streamer.register_chunk("transformer.h.0.attn.weight", 0, tensor_bytes);

    auto gpu_weight_buf = streamer.stream_chunk_to_gpu("transformer.h.0.attn.weight", model_path);
    assert(gpu_weight_buf != nullptr && gpu_weight_buf->get() != nullptr && "GPU streaming succeeded");

    // 4. Verify GPU VRAM mapped content matches source model file
    void* mapped_ptr = gpu_weight_buf->map();
    assert(mapped_ptr != nullptr && "GPU buffer mapping succeeded");

    float* gpu_floats = static_cast<float*>(mapped_ptr);
    bool data_matches = true;
    for (uint32_t i = 0; i < N; ++i) {
        if (std::abs(gpu_floats[i] - orig_weights[i]) > 1e-4f) {
            data_matches = false;
            break;
        }
    }
    gpu_weight_buf->unmap();

    assert(data_matches && "GPU VRAM streamed model weight verification passed");
    std::cout << "Real Model Weight Streaming to GPU VRAM Verification PASSED!\n";

    return 0;
}
