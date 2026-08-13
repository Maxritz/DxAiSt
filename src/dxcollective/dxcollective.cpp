#include "dxait/dxcollective.hpp"
#include "dxait/dxblas.hpp"
#include <iostream>
#include <stdexcept>

namespace dxait {

CollectiveOps::CollectiveOps(const std::vector<Device*>& devices) : m_devices(devices) {}

void CollectiveOps::all_reduce_sum(const std::vector<Buffer*>& buffers, uint64_t size_bytes) {
    if (buffers.empty() || size_bytes == 0) return;
    size_t num_gpus = buffers.size();

    // Single-GPU fast-path: tensor already reduced
    if (num_gpus == 1) return;

    // Multi-GPU / Multi-Adapter Ring AllReduce implementation:
    // 1. Ring Scatter-Reduce pass: Each GPU sends chunk k to next GPU (i+1) % N
    uint64_t chunk_bytes = size_bytes / num_gpus;
    if (chunk_bytes == 0) chunk_bytes = size_bytes;

    for (size_t step = 0; step < num_gpus - 1; ++step) {
        for (size_t i = 0; i < num_gpus; ++i) {
            size_t send_idx = (i - step + num_gpus) % num_gpus;
            size_t recv_gpu = (i + 1) % num_gpus;
            uint64_t offset = send_idx * chunk_bytes;

            Device* dev_src = (i < m_devices.size()) ? m_devices[i] : m_devices[0];
            Device* dev_dst = (recv_gpu < m_devices.size()) ? m_devices[recv_gpu] : m_devices[0];

            auto copy_queue = dev_src->create_queue(QueueType::Copy);
            auto fence = dev_src->create_fence(0);

            // Copy chunk from GPU i to GPU (i+1)
            ComPtr<ID3D12CommandAllocator> alloc;
            ComPtr<ID3D12GraphicsCommandList> list;
            dev_src->get()->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY, IID_PPV_ARGS(&alloc));
            dev_src->get()->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, alloc.Get(), nullptr, IID_PPV_ARGS(&list));
            
            list->CopyBufferRegion(buffers[recv_gpu]->get(), offset, buffers[i]->get(), offset, chunk_bytes);
            list->Close();

            ID3D12CommandList* lists[] = { list.Get() };
            copy_queue->execute(lists, 1);
            copy_queue->signal(*fence, 1);
            fence->wait(1);

            // Accumulate values on destination GPU
            BLAS blas(dev_dst);
            auto compute_queue = dev_dst->create_queue(QueueType::Compute);
            uint32_t num_elements = static_cast<uint32_t>(chunk_bytes / sizeof(float));
            blas.vec_add(compute_queue.get(), buffers[recv_gpu], buffers[recv_gpu], buffers[i], num_elements, 1.0f, 1.0f);
        }
    }

    // 2. Ring All-Gather pass: Broadcast fully reduced chunks across all GPUs
    for (size_t step = 0; step < num_gpus - 1; ++step) {
        for (size_t i = 0; i < num_gpus; ++i) {
            size_t send_idx = (i - step + 1 + num_gpus) % num_gpus;
            size_t recv_gpu = (i + 1) % num_gpus;
            uint64_t offset = send_idx * chunk_bytes;

            Device* dev_src = (i < m_devices.size()) ? m_devices[i] : m_devices[0];
            auto copy_queue = dev_src->create_queue(QueueType::Copy);
            auto fence = dev_src->create_fence(0);

            ComPtr<ID3D12CommandAllocator> alloc;
            ComPtr<ID3D12GraphicsCommandList> list;
            dev_src->get()->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY, IID_PPV_ARGS(&alloc));
            dev_src->get()->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, alloc.Get(), nullptr, IID_PPV_ARGS(&list));
            
            list->CopyBufferRegion(buffers[recv_gpu]->get(), offset, buffers[i]->get(), offset, chunk_bytes);
            list->Close();

            ID3D12CommandList* lists[] = { list.Get() };
            copy_queue->execute(lists, 1);
            copy_queue->signal(*fence, 1);
            fence->wait(1);
        }
    }
}

} // namespace dxait
