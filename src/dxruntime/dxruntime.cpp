#include "dxait/dxait.hpp"
#include <iostream>
#include <string>
#include <windows.h>

namespace dxait {

namespace {
struct ConfigState {
    uint64_t chunk_size_bytes = 1024ull * 1024; // 1 MiB default payload chunk
    double vram_margin_ratio = 0.10;             // keep 10% VRAM headroom
    bool async_prefetch_enabled = true;
};
ConfigState& config_state() {
    static ConfigState s;
    return s;
}
} // namespace

uint64_t Config::get_chunk_size_bytes() { return config_state().chunk_size_bytes; }
void Config::set_chunk_size_bytes(uint64_t bytes) { config_state().chunk_size_bytes = bytes; }
double Config::get_vram_margin_ratio() { return config_state().vram_margin_ratio; }
void Config::set_vram_margin_ratio(double ratio) { config_state().vram_margin_ratio = ratio; }
bool Config::is_async_prefetch_enabled() { return config_state().async_prefetch_enabled; }
void Config::set_async_prefetch_enabled(bool enabled) { config_state().async_prefetch_enabled = enabled; }

void TDRGuard::yield_gpu_breather(ID3D12CommandQueue* queue, ID3D12Fence* fence, uint64_t& fence_val) {
    if (!queue || !fence) return;
    fence_val++;
    queue->Signal(fence, fence_val);
    if (fence->GetCompletedValue() >= fence_val) return;
    HANDLE event = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    if (event) {
        fence->SetEventOnCompletion(fence_val, event);
        WaitForSingleObject(event, 500); //500ms max breather pulse, prevents WDDM TDR blackout
        CloseHandle(event);
    }
}

bool TDRGuard::is_device_removed(ID3D12Device* device) {
    if (!device) return true;
    HRESULT hr = device->GetDeviceRemovedReason();
    if (FAILED(hr)) {
        std::cerr << "[DXAiT TDRGuard] GPU Device Removal Detected! HRESULT: 0x"
                  << std::hex << hr << std::dec << "\n";
        return true;
    }
    return false;
}

Device::Device(ComPtr<IDXGIAdapter1> adapter) {
    //Arch-conditional: RDNA2 system d3d12.dll, RDNA4+ Agility D3D12Core.dll.
    bool created = SUCCEEDED(arch_conditional_create_device(adapter.Get(), m_device.ReleaseAndGetAddressOf()));
    if (!created) created = SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&m_device)));
    if (!created) throw std::runtime_error("D3D12CreateDevice failed");

    DXGI_ADAPTER_DESC1 desc{};
    adapter->GetDesc1(&desc);
    char name_buf[256]{};
    WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, name_buf, sizeof(name_buf), nullptr, nullptr);
    m_caps.name = name_buf;
    m_caps.vendor_id = desc.VendorId;
    m_caps.device_id = desc.DeviceId;
    m_caps.dedicated_video_memory = desc.DedicatedVideoMemory;
    m_caps.shared_system_memory = desc.SharedSystemMemory;

    //Query D3D12Options1 (Wave ops, Wave min/max)
    D3D12_FEATURE_DATA_D3D12_OPTIONS1 options1{};
    if (SUCCEEDED(m_device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &options1, sizeof(options1)))) {
        m_caps.wave_ops = options1.WaveOps;
        m_caps.wave_min = options1.WaveLaneCountMin;
        m_caps.wave_max = options1.WaveLaneCountMax;
    }

    //Query Shader Model
    D3D12_FEATURE_DATA_SHADER_MODEL sm_data{D3D_SHADER_MODEL_6_8};
    D3D12_FEATURE_DATA_SHADER_MODEL fallback{D3D_SHADER_MODEL_6_0};
    if (FAILED(m_device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &sm_data, sizeof(sm_data)))) {
        sm_data.HighestShaderModel = D3D_SHADER_MODEL_6_0;
    }
    m_caps.shader_model = sm_data.HighestShaderModel;

    //Classify arch by device ID (NOT wave counts): RX 9070 RDNA4 reports the same
    //WaveOps/32-64 lane range as RX 6700 RDNA2, so wave-min/max would mislabel it.
    m_caps.arch_family = classify_gpu_architecture(m_caps.vendor_id, m_caps.device_id);
    switch (m_caps.arch_family) {
        case ArchitectureFamily::AMD_RDNA2:
            m_caps.preferred_wave_size = 64; //Wave64 ~25% faster GEMV/decode on RDNA2
            m_caps.dot4_supported = true;
            m_caps.wmma_supported = false;  //RDNA2 has no WMMA matrix hardware
            break;
        case ArchitectureFamily::AMD_RDNA3:
        case ArchitectureFamily::AMD_RDNA4:
            m_caps.preferred_wave_size = 32;
            m_caps.dot4_supported = true;
            m_caps.wmma_supported = true;   //WMMA present on RDNA3+ (MMA_F16 on RDNA4)
            break;
        default:
            m_caps.preferred_wave_size = (m_caps.wave_min <= 32 && m_caps.wave_max >= 64) ? 64 : 32;
            m_caps.dot4_supported = (m_caps.wave_ops != 0);
            m_caps.wmma_supported = false;
            break;
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

//namespace dxait
}
