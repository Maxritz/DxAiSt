#include "dxait/dxait.hpp"
#include <iostream>

namespace dxait {

void TDRGuard::yield_gpu_breather(ID3D12CommandQueue* queue, ID3D12Fence* fence, uint64_t& fence_val) {
    if (!queue || !fence) return;
    fence_val++;
    queue->Signal(fence, fence_val);
    if (fence->GetCompletedValue() < fence_val) {
        HANDLE event = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        if (event) {
            fence->SetEventOnCompletion(fence_val, event);
            WaitForSingleObject(event, 500); // 500ms max breather pulse to prevent WDDM TDR blackout
            CloseHandle(event);
        }
    }
}

bool TDRGuard::is_device_removed(ID3D12Device* device) {
    if (!device) return true;
    HRESULT hr = device->GetDeviceRemovedReason();
    if (FAILED(hr)) {
        std::cerr << "[DXAiT TDRGuard] GPU Device Removal Detected! HRESULT: 0x" << std::hex << hr << std::dec << "\n";
        return true;
    }
    return false;
}

Device::Device(ComPtr<IDXGIAdapter1> adapter) {
    bool created = SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&m_device)));
    if (!created) {
        created = SUCCEEDED(system_d3d12_create_device(adapter.Get(), m_device.ReleaseAndGetAddressOf()));
    }
    if (!created) {
        throw std::runtime_error("D3D12CreateDevice failed");
    }

    DXGI_ADAPTER_DESC1 desc{};
    if (SUCCEEDED(adapter->GetDesc1(&desc))) {
        char name_buf[256] = {};
        WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, name_buf, sizeof(name_buf), nullptr, nullptr);
        m_caps.name = name_buf;
        m_caps.vendor_id = desc.VendorId;
        m_caps.device_id = desc.DeviceId;
        m_caps.dedicated_video_memory = desc.DedicatedVideoMemory;
        m_caps.shared_system_memory = desc.SharedSystemMemory;

        D3D12_FEATURE_DATA_D3D12_OPTIONS1 options1{};
        if (SUCCEEDED(m_device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &options1, sizeof(options1)))) {
            m_caps.wave_ops = options1.WaveOps;
            m_caps.wave_min = options1.WaveLaneCountMin;
            m_caps.wave_max = options1.WaveLaneCountMax;
        }

        if (m_caps.vendor_id == 0x1002 && m_caps.wave_min <= 32 && m_caps.wave_max >= 64) {
            m_caps.arch_family = ArchitectureFamily::AMD_RDNA2;
            m_caps.preferred_wave_size = 64; // Wave64 optimal for RDNA2 decode
            m_caps.dot4_supported = true;
            m_caps.wmma_supported = false;
        }
    }
}

std::unique_ptr<Buffer> Device::create_buffer(uint64_t size, MemLocation loc) {
    return std::make_unique<Buffer>(m_device.Get(), size, loc);
}

std::unique_ptr<Queue> Device::create_queue(QueueType type) {
    return std::make_unique<Queue>(m_device.Get(), type);
}

std::unique_ptr<Fence> Device::create_fence(uint64_t initial_val) {
    return std::make_unique<Fence>(m_device.Get(), initial_val);
}

} // namespace dxait
