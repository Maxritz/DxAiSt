#include "dxait/dxait.hpp"
#include <codecvt>
#include <locale>
#include <windows.h>

namespace dxait {

// System D3D12 entry (no Agility). Used when the Agility DLL is absent.
HRESULT system_d3d12_create_device(IDXGIAdapter1* adapter, ID3D12Device** out) {
    static decltype(&D3D12CreateDevice) fn = nullptr;
    if (!fn) {
        HMODULE m = LoadLibraryA("d3d12.dll");
        if (!m) return E_FAIL;
        fn = reinterpret_cast<decltype(&D3D12CreateDevice)>((void*)GetProcAddress(m, "D3D12CreateDevice"));
        if (!fn) return E_FAIL;
    }
    return fn(adapter, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(out));
}


static ArchitectureFamily classify_gpu_architecture(uint32_t vendor_id, uint32_t device_id, uint32_t wave_min, uint32_t wave_max) {
    if (vendor_id == 0x1002) { // AMD
        // RDNA4 (gfx1201) device IDs: e.g. 0x7440, 0x7441, 0x7500...
        if ((device_id & 0xFF00) == 0x7500 || (device_id & 0xFF00) == 0x7400) {
            return ArchitectureFamily::AMD_RDNA4;
        }
        // RDNA3 (gfx1100 family): 0x7400 - 0x743F
        // RDNA2 (gfx1030 / gfx1031 RX 6700 XT): 0x73A0 - 0x73DF
        if (wave_min <= 32 && wave_max >= 64) {
            return ArchitectureFamily::AMD_RDNA2;
        }
        return ArchitectureFamily::AMD_RDNA3;
    } else if (vendor_id == 0x10DE) { // NVIDIA
        return ArchitectureFamily::NVIDIA_Ampere_Ada_Blackwell;
    } else if (vendor_id == 0x8086) { // Intel
        return ArchitectureFamily::Intel_Arc_Xe;
    }
    return ArchitectureFamily::Unknown;
}

std::vector<AdapterCaps> Adapter::enumerate() {
    std::vector<AdapterCaps> caps_list;
    ComPtr<IDXGIFactory6> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return caps_list;

    // Enumerate ALL adapters (EnumAdapters1), not just GPU-preference ones.
    // On headless/RDP boxes the discrete GPU may not be the display adapter, so
    // EnumAdapterByGpuPreference(HIGH_PERFORMANCE) can miss it. llama.cpp does
    // the same full-enumeration route.
    ComPtr<IDXGIAdapter1> adapter;
    for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 desc;
        if (FAILED(adapter->GetDesc1(&desc))) continue;
        // Skip only WARP (software rasterizer); keep everything else including
        // non-display compute-only GPUs.
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;

        ComPtr<ID3D12Device> test_device;
        if (FAILED(system_d3d12_create_device(adapter.Get(), test_device.ReleaseAndGetAddressOf()))) continue;

        AdapterCaps caps{};
        char name_buf[256] = {};
        WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, name_buf, sizeof(name_buf), nullptr, nullptr);
        caps.name = name_buf;
        caps.vendor_id = desc.VendorId;
        caps.device_id = desc.DeviceId;
        caps.dedicated_video_memory = desc.DedicatedVideoMemory;
        caps.shared_system_memory = desc.SharedSystemMemory;

        // Query D3D12 Options1 (Wave ops, Wave min/max)
        D3D12_FEATURE_DATA_D3D12_OPTIONS1 options1{};
        if (SUCCEEDED(test_device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &options1, sizeof(options1)))) {
            caps.wave_ops = options1.WaveOps;
            caps.wave_min = options1.WaveLaneCountMin;
            caps.wave_max = options1.WaveLaneCountMax;
        }

        // Query Shader Model
        D3D12_FEATURE_DATA_SHADER_MODEL sm_data{D3D_SHADER_MODEL_6_8};
        if (FAILED(test_device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &sm_data, sizeof(sm_data)))) {
            sm_data.HighestShaderModel = D3D_SHADER_MODEL_6_0;
        }
        caps.shader_model = sm_data.HighestShaderModel;

        // Query Architecture & Wave preferences
        caps.arch_family = classify_gpu_architecture(caps.vendor_id, caps.device_id, caps.wave_min, caps.wave_max);
        if (caps.arch_family == ArchitectureFamily::AMD_RDNA2) {
            // RDNA2: Wave64 is ~25% faster for GEMV / decode due to instruction issue density
            caps.preferred_wave_size = 64;
            caps.dot4_supported = true;
            caps.wmma_supported = false; // RDNA2 has no WMMA matrix hardware
        } else if (caps.arch_family == ArchitectureFamily::AMD_RDNA4) {
            // RDNA4: Wave32 native + WMMA matrix hardware (MMA_F16 preferred for decode)
            caps.preferred_wave_size = 32;
            caps.dot4_supported = true;
            caps.wmma_supported = true;
        } else {
            caps.preferred_wave_size = caps.wave_min ? caps.wave_min : 32;
        }

        caps_list.push_back(caps);
    }
    return caps_list;
}

std::unique_ptr<Device> Adapter::create_device(uint32_t index) {
    ComPtr<IDXGIFactory6> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return nullptr;

    ComPtr<IDXGIAdapter1> adapter;
    if (FAILED(factory->EnumAdapters1(index, &adapter))) {
        return nullptr;
    }
    return std::make_unique<Device>(adapter);
}

} // namespace dxait
