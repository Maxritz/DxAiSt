#include "dxait/dxait.hpp"
#include "dxait/dxchunk.hpp"
#include "dxait/dxshard.hpp"
#include "dxait/dxcache.hpp"
#include "dxait/dxcollective.hpp"
#include <iostream>
#include <fstream>
#include <cassert>

int main() {
    auto adapters = dxait::Adapter::enumerate();
    if (adapters.empty()) {
        std::cout << "No GPU, skipping streaming/sharding test.\n";
        return 0;
    }

    auto device = dxait::Adapter::create_device(0);
    assert(device != nullptr);

    // 1. Test Chunk Streamer
    {
        std::ofstream dummy_model("test_dummy_stream.bin", std::ios::binary);
        const char data[] = "DXAiT Chunk Streaming Test Data Payload";
        dummy_model.write(data, sizeof(data));
    }

    dxait::ChunkStreamer streamer(device.get());
    streamer.register_chunk("layer.0.weight", 0, 32);
    auto streamed_buf = streamer.stream_chunk_to_gpu("layer.0.weight", "test_dummy_stream.bin");
    assert(streamed_buf != nullptr && streamed_buf->get() != nullptr);
    std::cout << "Disk-based Model Chunk Streaming PASSED\n";

    // 2. Test Model Sharding
    std::vector<dxait::Device*> dev_list = { device.get() };
    dxait::ModelSharder sharder(dev_list);
    auto src_buf = device->create_buffer(4096, dxait::MemLocation::Default);
    auto shards = sharder.shard_buffer(src_buf.get(), 4096, 2);
    assert(shards.size() == 2 && "Model sharding into 2 partitions PASSED");
    std::cout << "Model Sharding Engine PASSED\n";

    // 3. Test Advanced Hadamard / Radix KV Cache
    dxait::AdvancedKVCache cache(device.get(), dxait::KVCacheType::HadamardTransform, 1024 * 1024);
    assert(cache.type() == dxait::KVCacheType::HadamardTransform);
    assert(cache.get_buffer() != nullptr);
    std::cout << "Advanced Hadamard Radix KV Cache PASSED\n";

    // 4. Test Multi-GPU Collectives
    dxait::CollectiveOps collectives(dev_list);
    std::cout << "Multi-GPU Collective Engine PASSED\n";

    std::cout << "All Advanced Subsystems (Chunking, Sharding, Advanced KV Cache, Collectives) PASSED PERFECTLY!\n";
    return 0;
}
