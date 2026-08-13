#ifndef DXAIT_DXSHARD_HPP
#define DXAIT_DXSHARD_HPP

#include "dxait.hpp"
#include <vector>
#include <memory>
#include <string>

namespace dxait {

struct ShardConfig {
    uint32_t num_devices{1};
    uint64_t tensor_split_dim{0};
};

struct OffloadConfig {
    float vram_ratio{0.5f}; // 50% in GPU VRAM
    float ram_ratio{0.5f};  // 50% in CPU System RAM
    uint64_t total_model_bytes{0};
};

class ModelSharder {
public:
    explicit ModelSharder(const std::vector<Device*>& devices);
    ~ModelSharder() = default;

    std::vector<std::unique_ptr<Buffer>> shard_buffer(
        Buffer* src_buffer,
        uint64_t total_size,
        uint32_t num_shards
    );

private:
    std::vector<Device*> m_devices;
};

class OffloadPartitionEngine {
public:
    OffloadPartitionEngine(Device* device, const OffloadConfig& config);
    ~OffloadPartitionEngine() = default;

    std::unique_ptr<Buffer> allocate_vram_partition(uint64_t bytes);
    std::unique_ptr<Buffer> allocate_ram_partition(uint64_t bytes);
    void page_swap(Queue* queue, Buffer* ram_buf, Buffer* vram_buf, uint64_t bytes);

private:
    Device* m_device;
    OffloadConfig m_config;
};

} // namespace dxait

#endif // DXAIT_DXSHARD_HPP
