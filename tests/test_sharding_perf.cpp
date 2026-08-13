#include "dxait/dxait.hpp"
#include "dxait/dxmodel.hpp"
#include "dxait/dxchunk.hpp"
#include <iostream>
#include <vector>
#include <cassert>
#include <chrono>
#include <iomanip>

int main() {
    std::cout << "========================================================\n";
    std::cout << " DXAiT Optimal Chunk Size Sweep & Throughput Benchmark\n";
    std::cout << "========================================================\n\n";

    const std::string model_path = "E:\\OLLAMA-Models\\GGUF\\Gemma-4-31B-Fable-5-Distill.q4_k_m.gguf";

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

    // Sweep Chunk Sizes: 32 MB, 64 MB, 128 MB, 256 MB, 512 MB, 1024 MB
    const std::vector<uint64_t> test_sizes_mb = { 32, 64, 128, 256, 512, 1024 };

    std::cout << "| Chunk Size (MB) | Num Chunks | Total MB | Time (ms) | Speed (GB/s) |\n";
    std::cout << "|-----------------|------------|----------|-----------|--------------|\n";

    uint64_t optimal_size_mb = 32;
    double max_gbps = 0.0;

    for (uint64_t size_mb : test_sizes_mb) {
        uint64_t chunk_size = size_mb * 1024ULL * 1024ULL;
        uint32_t num_chunks = static_cast<uint32_t>(std::min<uint64_t>(8, (2048ULL / size_mb)));
        if (num_chunks < 2) num_chunks = 2;

        dxait::ChunkStreamer streamer(device.get());
        std::vector<std::unique_ptr<dxait::Buffer>> buffers;

        for (uint32_t i = 0; i < num_chunks; ++i) {
            streamer.register_chunk("chunk_" + std::to_string(i), i * chunk_size, chunk_size, dxait::MemLocation::Default);
        }

        auto t0 = std::chrono::high_resolution_clock::now();
        for (uint32_t i = 0; i < num_chunks; ++i) {
            auto buf = streamer.stream_chunk_to_gpu("chunk_" + std::to_string(i), model_path);
            if (buf && buf->get()) {
                buffers.push_back(std::move(buf));
            }
        }
        auto t1 = std::chrono::high_resolution_clock::now();

        double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double total_gb = (num_chunks * size_mb) / 1024.0;
        double gbps = total_gb / (total_ms / 1000.0);

        std::cout << "| " << std::setw(15) << size_mb
                  << " | " << std::setw(10) << num_chunks
                  << " | " << std::setw(8) << (num_chunks * size_mb)
                  << " | " << std::setw(9) << std::fixed << std::setprecision(1) << total_ms
                  << " | " << std::setw(12) << std::fixed << std::setprecision(3) << gbps << " |\n";

        if (gbps > max_gbps) {
            max_gbps = gbps;
            optimal_size_mb = size_mb;
        }
    }

    std::cout << "\n========================================================\n";
    std::cout << " Optimal Chunk Size Result:\n";
    std::cout << "   Optimal Chunk Granularity: " << optimal_size_mb << " MB\n";
    std::cout << "   Peak DMA Streaming Speed:  " << std::fixed << std::setprecision(3) << max_gbps << " GB/s\n";
    std::cout << "========================================================\n";

    return 0;
}
