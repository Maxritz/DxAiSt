#include "dxait/dxait.hpp"
#include <iostream>

int main() {
    std::cout << "========================================================\n";
    std::cout << " DXAiT GPU Architecture & D3D12 Capability Inspector\n";
    std::cout << "========================================================\n\n";

    auto adapters = dxait::Adapter::enumerate();
    std::cout << "Found " << adapters.size() << " D3D12 hardware adapters:\n\n";

    for (size_t i = 0; i < adapters.size(); ++i) {
        const auto& caps = adapters[i];
        std::cout << "Adapter [" << i << "]: " << caps.name << "\n";
        std::cout << "  Vendor ID:            0x" << std::hex << caps.vendor_id << " Device ID: 0x" << caps.device_id << std::dec << "\n";
        std::cout << "  Dedicated VRAM:       " << caps.dedicated_video_memory / (1024 * 1024) << " MB\n";
        std::cout << "  Shared System Memory: " << caps.shared_system_memory / (1024 * 1024) << " MB\n";
        std::cout << "  Wave Lane Count:      [" << caps.wave_min << ", " << caps.wave_max << "]\n";
        std::cout << "  Preferred Wave Size:  " << caps.preferred_wave_size << "\n";
        std::cout << "  WMMA Matrix Hardware: " << (caps.wmma_supported ? "YES (RDNA4 / SM 6.8)" : "NO") << "\n";
        std::cout << "  Int8 Dot4 Primitives: " << (caps.dot4_supported ? "YES (dot4_add_i8)" : "NO") << "\n";
        std::cout << "--------------------------------------------------------\n";
    }
    return 0;
}
