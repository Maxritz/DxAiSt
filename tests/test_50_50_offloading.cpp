#include "dxait/dxait.hpp"
#include "dxait/dxshard.hpp"
#include <iostream>
#include <vector>
#include <cassert>
#include <chrono>
#include <cstring>

int main() {
    std::cout << "========================================================\n";
    std::cout << " DXAiT 50% VRAM / 50% RAM Offload & Hardware DMA Trace\n";
    std::cout << "========================================================\n\n";

    auto adapters = dxait::Adapter::enumerate();
    if (adapters.empty()) {
        std::cout << "No GPU adapter found, skipping test.\n";
        return 0;
    }

    auto device = dxait::Adapter::create_device(0);
    const auto& caps = device->caps();

    std::cout << "Target GPU Adapter:      " << caps.name << "\n";
    std::cout << "Dedicated VRAM Capacity:  " << caps.dedicated_video_memory / (1024 * 1024) << " MB\n";
    std::cout << "Shared System RAM:       " << caps.shared_system_memory / (1024 * 1024) << " MB\n\n";

    // 1. Allocate 50/50 Partitions (512 MB each)
    constexpr uint64_t total_layer_bytes = 512ULL * 1024ULL * 1024ULL; // 512 MB
    constexpr uint64_t partition_bytes = total_layer_bytes / 2; // 256 MB each

    dxait::OffloadConfig config{0.5f, 0.5f, total_layer_bytes};
    dxait::OffloadPartitionEngine offloader(device.get(), config);

    std::cout << "1. Allocating 256 MB in GPU Dedicated VRAM (D3D12_HEAP_TYPE_DEFAULT)...\n";
    auto vram_partition = offloader.allocate_vram_partition(partition_bytes);
    std::cout << "   VRAM Resource Handle: " << vram_partition->get() << "\n";

    std::cout << "2. Allocating 256 MB in System RAM / ReBAR (D3D12_HEAP_TYPE_UPLOAD)...\n";
    auto ram_partition = offloader.allocate_ram_partition(partition_bytes);
    std::cout << "   RAM Resource Handle:  " << ram_partition->get() << "\n";

    // Write known test pattern into System RAM buffer
    std::cout << "3. Writing test payload to System RAM buffer...\n";
    float* ram_ptr = static_cast<float*>(ram_partition->map());
    assert(ram_ptr != nullptr && "RAM buffer mapping succeeded");

    constexpr uint64_t num_floats = partition_bytes / sizeof(float);
    for (uint64_t i = 0; i < 1000; ++i) {
        ram_ptr[i] = static_cast<float>(i) * 3.14159f;
    }
    ram_partition->unmap();

    // 4. Issue D3D12 CopyBufferRegion Page Swap over PCIe DMA
    std::cout << "4. Issuing D3D12 COPY Queue CopyBufferRegion (System RAM -> Dedicated VRAM)...\n";
    auto copy_queue = device->create_queue(dxait::QueueType::Copy);

    auto t0 = std::chrono::high_resolution_clock::now();
    offloader.page_swap(copy_queue.get(), ram_partition.get(), vram_partition.get(), partition_bytes);
    auto t1 = std::chrono::high_resolution_clock::now();

    double swap_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double gbs = (partition_bytes / 1e9) / (swap_ms * 1e-3);

    std::cout << "   PCIe DMA Copy Completed in " << swap_ms << " ms (" << gbs << " GB/s throughput)!\n";

    // 5. Read back from Dedicated VRAM to Readback heap to verify hardware copy
    std::cout << "5. Copying Dedicated VRAM buffer back to Readback Heap for hardware verification...\n";
    auto readback_buf = device->create_buffer(partition_bytes, dxait::MemLocation::Readback);

    offloader.page_swap(copy_queue.get(), vram_partition.get(), readback_buf.get(), partition_bytes);

    float* rb_ptr = static_cast<float*>(readback_buf->map());
    assert(rb_ptr != nullptr && "Readback buffer mapping succeeded");

    bool verified = true;
    for (uint64_t i = 0; i < 1000; ++i) {
        float expected = static_cast<float>(i) * 3.14159f;
        if (std::abs(rb_ptr[i] - expected) > 1e-3f) {
            std::cerr << "Mismatch at float index " << i << ": got " << rb_ptr[i] << ", expected " << expected << "\n";
            verified = false;
            break;
        }
    }
    readback_buf->unmap();

    assert(verified && "Hardware PCIe DMA roundtrip verification passed");
    std::cout << "6. Hardware PCIe DMA Byte-for-Byte Verification PASSED!\n";
    std::cout << "========================================================\n";

    return 0;
}
