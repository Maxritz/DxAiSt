#include "dxait/dxshard.hpp"

namespace dxait {

ModelSharder::ModelSharder(const std::vector<Device*>& devices) : m_devices(devices) {}

std::vector<std::unique_ptr<Buffer>> ModelSharder::shard_buffer(
    Buffer* src_buffer,
    uint64_t total_size,
    uint32_t num_shards
) {
    (void)src_buffer;
    std::vector<std::unique_ptr<Buffer>> shards;
    if (num_shards == 0) return shards;

    uint64_t shard_size = total_size / num_shards;
    for (uint32_t i = 0; i < num_shards; ++i) {
        Device* dev = (i < m_devices.size()) ? m_devices[i] : m_devices[0];
        shards.push_back(dev->create_buffer(shard_size, MemLocation::Default));
    }
    return shards;
}

OffloadPartitionEngine::OffloadPartitionEngine(Device* device, const OffloadConfig& config)
    : m_device(device), m_config(config) {}

std::unique_ptr<Buffer> OffloadPartitionEngine::allocate_vram_partition(uint64_t bytes) {
    return m_device->create_buffer(bytes, MemLocation::Default);
}

std::unique_ptr<Buffer> OffloadPartitionEngine::allocate_ram_partition(uint64_t bytes) {
    return m_device->create_buffer(bytes, MemLocation::Upload); // System RAM / ReBAR
}

void OffloadPartitionEngine::page_swap(Queue* queue, Buffer* ram_buf, Buffer* vram_buf, uint64_t bytes) {
    auto fence = m_device->create_fence(0);
    ComPtr<ID3D12CommandAllocator> copy_alloc;
    ComPtr<ID3D12GraphicsCommandList> copy_list;

    if (FAILED(m_device->get()->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY, IID_PPV_ARGS(&copy_alloc)))) return;
    if (FAILED(m_device->get()->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, copy_alloc.Get(), nullptr, IID_PPV_ARGS(&copy_list)))) return;

    copy_list->CopyBufferRegion(vram_buf->get(), 0, ram_buf->get(), 0, bytes);
    copy_list->Close();

    ID3D12CommandList* lists[] = { copy_list.Get() };
    queue->execute(lists, 1);
    queue->signal(*fence, 1);
    fence->wait(1);
}

} // namespace dxait
