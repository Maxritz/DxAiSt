#include "dxait/dxsched.hpp"

namespace dxait {

MultiQueueScheduler::MultiQueueScheduler(Device* device) : m_device(device) {
    m_direct_queue = device->create_queue(QueueType::Direct);
    m_compute_queue = device->create_queue(QueueType::Compute);
    m_direct_fence = device->create_fence(0);
    m_compute_fence = device->create_fence(0);
}

MultiQueueScheduler::~MultiQueueScheduler() {
    wait_all();
}

void MultiQueueScheduler::submit_direct_work(ID3D12CommandList* list) {
    std::lock_guard<std::mutex> lock(m_mutex);
    ID3D12CommandList* lists[] = { list };
    m_direct_queue->execute(lists, 1);
    m_direct_val++;
    m_direct_queue->signal(*m_direct_fence, m_direct_val);
}

void MultiQueueScheduler::submit_compute_work(ID3D12CommandList* list) {
    std::lock_guard<std::mutex> lock(m_mutex);
    ID3D12CommandList* lists[] = { list };
    m_compute_queue->execute(lists, 1);
    m_compute_val++;
    m_compute_queue->signal(*m_compute_fence, m_compute_val);
}

void MultiQueueScheduler::wait_all() {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_direct_val > 0) m_direct_fence->wait(m_direct_val);
    if (m_compute_val > 0) m_compute_fence->wait(m_compute_val);
}

} // namespace dxait
