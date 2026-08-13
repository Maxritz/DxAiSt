#ifndef DXAIT_DXCHUNK_HPP
#define DXAIT_DXCHUNK_HPP

#include "dxait.hpp"
#include <string>
#include <vector>
#include <memory>
#include <fstream>
#include <future>

namespace dxait {

enum class ShardPolicy {
    AutoVRAMSpill,   // Maximize Dedicated VRAM, spill rest to Shared System RAM
    LayerPipeline,   // Split model layer ranges evenly across VRAM and System RAM
    EvenSplit5050    // Strict 50/50 memory footprint allocation
};

struct TensorChunk {
    std::string name;
    uint64_t file_offset{0};
    uint64_t size_bytes{0};
    MemLocation location{MemLocation::Default};
    bool in_vram{false};
    bool in_ram{false};
};

struct ShardPlan {
    uint32_t total_layers{0};
    uint32_t vram_layers{0};
    uint32_t sysram_layers{0};
    uint64_t vram_bytes{0};
    uint64_t sysram_bytes{0};
};

class ShardPlanner {
public:
    static ShardPlan plan_sharding(const Device* device, uint32_t num_layers, uint64_t bytes_per_layer, ShardPolicy policy);
};

class ChunkStreamer {
public:
    explicit ChunkStreamer(Device* device);
    ~ChunkStreamer();

    void register_chunk(const std::string& name, uint64_t offset, uint64_t size, MemLocation loc = MemLocation::Default);
    bool load_chunk_to_cpu(const std::string& name, const std::string& file_path, void* dest_ptr);
    
    std::unique_ptr<Buffer> stream_chunk_to_gpu(const std::string& name, const std::string& file_path);
    std::future<std::unique_ptr<Buffer>> stream_chunk_async(const std::string& name, const std::string& file_path);

    const std::vector<TensorChunk>& chunks() const { return m_chunks; }

private:
    Device* m_device;
    std::vector<TensorChunk> m_chunks;

    std::unique_ptr<Queue> m_copy_queue;
    std::unique_ptr<Fence> m_copy_fence;
    ComPtr<ID3D12CommandAllocator> m_copy_alloc;
    ComPtr<ID3D12GraphicsCommandList> m_copy_list;
    uint64_t m_fence_val{0};

    // Double Staging Buffers for Pipelined Async Streaming
    std::unique_ptr<Buffer> m_staging_upload_a;
    std::unique_ptr<Buffer> m_staging_upload_b;
    uint64_t m_staging_size{0};
    uint32_t m_active_staging_idx{0};
};

} // namespace dxait

#endif // DXAIT_DXCHUNK_HPP
