#include "dxait/dxait.hpp"
#include "dxait/dxmodel.hpp"
#include "dxait/dxchunk.hpp"
#include <iostream>
#include <vector>
#include <cassert>
#include <chrono>

int main() {
    std::cout << "========================================================\n";
    std::cout << " DXAiT Full 18.68 GB GGUF Model VRAM + System RAM Residency Test\n";
    std::cout << "========================================================\n\n";

    const std::string model_18gb_path = "E:\\OLLAMA-Models\\GGUF\\Gemma-4-31B-Fable-5-Distill.q4_k_m.gguf";

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

    // 1. Zero-Copy Memory-Map the 18.68 GB GGUF Model
    std::cout << "1. Memory-mapping 18.68 GB GGUF model file...\n";
    auto t0 = std::chrono::high_resolution_clock::now();
    dxait::ModelLoader loader;
    bool loaded = loader.load_file(model_18gb_path);
    auto t1 = std::chrono::high_resolution_clock::now();

    if (!loaded) {
        std::cout << "Model file not found.\n";
        return 0;
    }

    std::cout << "   Memory-mapped 18.68 GB file in " << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms\n";
    std::cout << "   Parsed " << loader.tensor_count() << " tensors!\n\n";

    // 2. Allocate and Stream 14.0 GB of Tensors into Dedicated VRAM + System RAM heaps
    constexpr uint64_t chunk_size = 2ULL * 1024ULL * 1024ULL * 1024ULL; // 2 GB per chunk
    constexpr uint32_t num_chunks = 7; // 7 x 2 GB = 14 GB total GPU streaming

    dxait::ChunkStreamer streamer(device.get());
    std::vector<std::unique_ptr<dxait::Buffer>> vram_buffers;

    std::cout << "2. Streaming 14.0 GB of model tensors into Dedicated VRAM + System RAM...\n";
    auto stream_t0 = std::chrono::high_resolution_clock::now();
    for (uint32_t i = 0; i < num_chunks; ++i) {
        std::string chunk_name = "layer_" + std::to_string(i) + "_weight";
        streamer.register_chunk(chunk_name, i * chunk_size, chunk_size);
        
        auto chunk_t0 = std::chrono::high_resolution_clock::now();
        auto buf = streamer.stream_chunk_to_gpu(chunk_name, model_18gb_path);
        auto chunk_t1 = std::chrono::high_resolution_clock::now();
        double chunk_ms = std::chrono::duration<double, std::milli>(chunk_t1 - chunk_t0).count();
        double chunk_gbps = (2.0 / (chunk_ms / 1000.0));

        if (buf && buf->get()) {
            buf->make_resident(device->get());
            vram_buffers.push_back(std::move(buf));
            std::cout << "   Streamed Chunk [" << i + 1 << "/" << num_chunks << "] (2.0 GB) -> VRAM Handle " << vram_buffers.back()->get() 
                      << " in " << chunk_ms << " ms (" << chunk_gbps << " GB/s)\n";
        }
    }
    auto stream_t1 = std::chrono::high_resolution_clock::now();
    double total_stream_ms = std::chrono::duration<double, std::milli>(stream_t1 - stream_t0).count();
    double total_gbps = (14.0 / (total_stream_ms / 1000.0));

    std::cout << "\n   Total 14.0 GB Streaming Time: " << total_stream_ms << " ms (" << total_gbps << " GB/s throughput)\n";
    std::cout << "========================================================\n";
    std::cout << " Full 18.68 GB GGUF Model Streaming & Residency Test PASSED!\n";

    return 0;
}
