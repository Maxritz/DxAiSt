#include "dxait/dxait.hpp"
#include <iostream>
#include <cassert>
#include <cstring>

int main() {
    auto adapters = dxait::Adapter::enumerate();
    if (adapters.empty()) {
        std::cout << "No GPU adapter, skipping memory allocation test.\n";
        return 0;
    }

    auto device = dxait::Adapter::create_device(0);
    assert(device != nullptr);

    // 1. Upload heap allocation
    constexpr uint64_t buf_size = 1024;
    auto upload_buf = device->create_buffer(buf_size, dxait::MemLocation::Upload);
    assert(upload_buf != nullptr && upload_buf->get() != nullptr);

    // Map and write test pattern
    void* ptr = upload_buf->map();
    assert(ptr != nullptr);
    const char test_str[] = "DXAiT D3D12 Compute Fabric Buffer Test";
    std::memcpy(ptr, test_str, sizeof(test_str));
    upload_buf->unmap();

    // 2. Default heap allocation (VRAM) & ReBAR allocation
    auto default_buf = device->create_buffer(buf_size, dxait::MemLocation::Default);
    assert(default_buf != nullptr && default_buf->get() != nullptr);

    auto rebar_buf = device->create_buffer(buf_size, dxait::MemLocation::ReBAR);
    assert(rebar_buf != nullptr && rebar_buf->get() != nullptr);

    // 3. Queue & Fence creation
    auto queue = device->create_queue(dxait::QueueType::Direct);
    assert(queue != nullptr);

    auto fence = device->create_fence(0);
    assert(fence != nullptr);

    queue->signal(*fence, 1);
    fence->wait(1);
    assert(fence->is_completed(1));

    std::cout << "DXAiT Memory Allocation and Queue Fence Test Passed!\n";
    return 0;
}
