#ifndef DXAIT_DXSCHED_HPP
#define DXAIT_DXSCHED_HPP

#include "dxait.hpp"
#include <queue>
#include <mutex>
#include <thread>
#include <atomic>

namespace dxait {

class MultiQueueScheduler {
public:
    explicit MultiQueueScheduler(Device* device);
    ~MultiQueueScheduler();

    void submit_direct_work(ID3D12CommandList* list);
    void submit_compute_work(ID3D12CommandList* list);
    void wait_all();

private:
    Device* m_device;
    std::unique_ptr<Queue> m_direct_queue;
    std::unique_ptr<Queue> m_compute_queue;
    std::unique_ptr<Fence> m_direct_fence;
    std::unique_ptr<Fence> m_compute_fence;
    uint64_t m_direct_val{0};
    uint64_t m_compute_val{0};
    std::mutex m_mutex;
};

} // namespace dxait

#endif // DXAIT_DXSCHED_HPP
