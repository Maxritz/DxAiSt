#include "dxait/dxchunk.hpp"
#include <windows.h>
#include <fstream>
#include <cstring>
#include <stdexcept>
#include <iostream>

namespace dxait {

class Win32Mmap {
public:
    Win32Mmap(const std::string& path) {
        m_file = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (m_file == INVALID_HANDLE_VALUE) return;
        m_mapping = CreateFileMappingA(m_file, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!m_mapping) return;
        m_ptr = MapViewOfFile(m_mapping, FILE_MAP_READ, 0, 0, 0);
    }
    ~Win32Mmap() {
        if (m_ptr) UnmapViewOfFile(m_ptr);
        if (m_mapping) CloseHandle(m_mapping);
        if (m_file != INVALID_HANDLE_VALUE) CloseHandle(m_file);
    }
    const uint8_t* data() const { return static_cast<const uint8_t*>(m_ptr); }
    bool is_valid() const { return m_ptr != nullptr; }
private:
    HANDLE m_file{INVALID_HANDLE_VALUE};
    HANDLE m_mapping{nullptr};
    void* m_ptr{nullptr};
};

ShardPlan ShardPlanner::plan_sharding(const Device* device, uint32_t num_layers, uint64_t bytes_per_layer, ShardPolicy policy) {
    ShardPlan plan{};
    plan.total_layers = num_layers;
    uint64_t total_bytes = num_layers * bytes_per_layer;
    uint64_t vram_capacity = device->caps().dedicated_video_memory;
    // Keep 20% safety margin in VRAM for KV cache and activation scratchpads
    uint64_t vram_budget = static_cast<uint64_t>(vram_capacity * 0.80);

    switch (policy) {
    case ShardPolicy::AutoVRAMSpill: {
        plan.vram_layers = static_cast<uint32_t>(std::min<uint64_t>(num_layers, vram_budget / bytes_per_layer));
        plan.sysram_layers = num_layers - plan.vram_layers;
        break;
    }
    case ShardPolicy::EvenSplit5050: {
        plan.vram_layers = num_layers / 2;
        plan.sysram_layers = num_layers - plan.vram_layers;
        break;
    }
    case ShardPolicy::LayerPipeline: {
        plan.vram_layers = static_cast<uint32_t>(std::min<uint64_t>(num_layers, vram_budget / bytes_per_layer));
        plan.sysram_layers = num_layers - plan.vram_layers;
        break;
    }
    }

    plan.vram_bytes = plan.vram_layers * bytes_per_layer;
    plan.sysram_bytes = plan.sysram_layers * bytes_per_layer;

    std::cout << "[DXAiT ShardPlanner] Planned Sharding Strategy:\n"
              << "   - Total Model Layers: " << plan.total_layers << " (" << (total_bytes / (1024 * 1024)) << " MB)\n"
              << "   - Dedicated VRAM:    " << plan.vram_layers << " layers (" << (plan.vram_bytes / (1024 * 1024)) << " MB)\n"
              << "   - System RAM Spill:  " << plan.sysram_layers << " layers (" << (plan.sysram_bytes / (1024 * 1024)) << " MB)\n";

    return plan;
}

ChunkStreamer::ChunkStreamer(Device* device) : m_device(device) {
    m_copy_queue = m_device->create_queue(QueueType::Copy);
    m_copy_fence = m_device->create_fence(0);

    if (FAILED(m_device->get()->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY, IID_PPV_ARGS(&m_copy_alloc)))) {
        throw std::runtime_error("Failed to create copy command allocator");
    }
    if (FAILED(m_device->get()->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, m_copy_alloc.Get(), nullptr, IID_PPV_ARGS(&m_copy_list)))) {
        throw std::runtime_error("Failed to create copy command list");
    }
    m_copy_list->Close();
}

ChunkStreamer::~ChunkStreamer() = default;

void ChunkStreamer::register_chunk(const std::string& name, uint64_t offset, uint64_t size, MemLocation loc) {
    m_chunks.push_back({name, offset, size, loc, false, false});
}

bool ChunkStreamer::load_chunk_to_cpu(const std::string& name, const std::string& file_path, void* dest_ptr) {
    Win32Mmap mmap(file_path);
    if (!mmap.is_valid()) return false;

    for (auto& c : m_chunks) {
        if (c.name == name) {
            std::memcpy(dest_ptr, mmap.data() + c.file_offset, c.size_bytes);
            c.in_ram = true;
            return true;
        }
    }
    return false;
}

std::unique_ptr<Buffer> ChunkStreamer::stream_chunk_to_gpu(const std::string& name, const std::string& file_path) {
    for (auto& c : m_chunks) {
        if (c.name == name) {
            // Target buffer memory location based on sharding policy (Dedicated VRAM vs Upload/System RAM)
            auto target_buf = m_device->create_buffer(c.size_bytes, c.location);

            if (c.location == MemLocation::Upload) {
                // Direct zero-copy map for System RAM resident layers
                void* ptr = target_buf->map();
                Win32Mmap mmap(file_path);
                if (mmap.is_valid()) {
                    std::memcpy(ptr, mmap.data() + c.file_offset, c.size_bytes);
                }
                target_buf->unmap();
                c.in_ram = true;
                return target_buf;
            }

            // High-speed VRAM transfer via persistent double-buffered staging heap
            if (!m_staging_upload_a || m_staging_size < c.size_bytes) {
                m_staging_upload_a = m_device->create_buffer(c.size_bytes, MemLocation::Upload);
                m_staging_upload_b = m_device->create_buffer(c.size_bytes, MemLocation::Upload);
                m_staging_size = c.size_bytes;
            }

            Buffer* active_staging = (m_active_staging_idx == 0) ? m_staging_upload_a.get() : m_staging_upload_b.get();
            m_active_staging_idx = 1 - m_active_staging_idx;

            void* ptr = active_staging->map();
            Win32Mmap mmap(file_path);
            if (!mmap.is_valid()) {
                active_staging->unmap();
                return nullptr;
            }
            std::memcpy(ptr, mmap.data() + c.file_offset, c.size_bytes);
            active_staging->unmap();

            m_copy_alloc->Reset();
            m_copy_list->Reset(m_copy_alloc.Get(), nullptr);
            m_copy_list->CopyBufferRegion(target_buf->get(), 0, active_staging->get(), 0, c.size_bytes);
            m_copy_list->Close();

            ID3D12CommandList* lists[] = { m_copy_list.Get() };
            m_copy_queue->execute(lists, 1);

            m_fence_val++;
            m_copy_queue->signal(*m_copy_fence, m_fence_val);
            m_copy_fence->wait(m_fence_val);

            c.in_vram = true;
            return target_buf;
        }
    }
    return nullptr;
}

std::future<std::unique_ptr<Buffer>> ChunkStreamer::stream_chunk_async(const std::string& name, const std::string& file_path) {
    return std::async(std::launch::async, [this, name, file_path]() {
        return this->stream_chunk_to_gpu(name, file_path);
    });
}

} // namespace dxait
