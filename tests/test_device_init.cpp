#include "dxait/dxait.hpp"
#include <iostream>
#include <cassert>

int main() {
    auto adapters = dxait::Adapter::enumerate();
    std::cout << "Found " << adapters.size() << " D3D12 hardware adapters:\n";
    for (const auto& caps : adapters) {
        std::cout << " - " << caps.name << " (VRAM: " << caps.dedicated_video_memory / (1024 * 1024) << " MB)\n";
        std::cout << "   Vendor ID: 0x" << std::hex << caps.vendor_id << " Device ID: 0x" << caps.device_id << std::dec << "\n";
        std::cout << "   Wave Min/Max: [" << caps.wave_min << ", " << caps.wave_max << "], Preferred Wave: " << caps.preferred_wave_size << "\n";
        std::cout << "   WMMA Supported: " << (caps.wmma_supported ? "YES" : "NO") << ", Dot4 Supported: " << (caps.dot4_supported ? "YES" : "NO") << "\n";
    }

    if (adapters.empty()) {
        std::cout << "No physical D3D12 GPU found, skipping device test.\n";
        return 0;
    }

    auto device = dxait::Adapter::create_device(0);
    assert(device != nullptr && "Device creation failed");
    assert(device->get() != nullptr && "Native D3D12 device is null");

    std::cout << "D3D12 Device successfully created!\n";
    return 0;
}
