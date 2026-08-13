#include "dxait/dxait.hpp"

namespace dxait {

Queue::Queue(ID3D12Device* device, QueueType type) {
    D3D12_COMMAND_QUEUE_DESC desc{};
    desc.Type = static_cast<D3D12_COMMAND_LIST_TYPE>(type);
    desc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
    desc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    desc.NodeMask = 0;

    if (FAILED(device->CreateCommandQueue(&desc, IID_PPV_ARGS(&m_queue)))) {
        throw std::runtime_error("CreateCommandQueue failed");
    }
}

void Queue::execute(ID3D12CommandList* const* lists, uint32_t count) {
    m_queue->ExecuteCommandLists(count, lists);
}

void Queue::signal(Fence& fence, uint64_t value) {
    fence.signal(m_queue.Get(), value);
}

void Queue::wait(Fence& fence, uint64_t value) {
    fence.wait(value);
}

Fence::Fence(ID3D12Device* device, uint64_t initial_val) {
    if (FAILED(device->CreateFence(initial_val, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_fence)))) {
        throw std::runtime_error("CreateFence failed");
    }
    m_event = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    if (!m_event) {
        throw std::runtime_error("CreateEvent failed");
    }
}

void Fence::signal(ID3D12CommandQueue* queue, uint64_t val) {
    queue->Signal(m_fence.Get(), val);
}

void Fence::wait(uint64_t val) {
    if (m_fence->GetCompletedValue() < val) {
        m_fence->SetEventOnCompletion(val, m_event);
        WaitForSingleObject(m_event, INFINITE);
    }
}

bool Fence::is_completed(uint64_t val) const {
    return m_fence->GetCompletedValue() >= val;
}

} // namespace dxait
