#include "dxait/dxait.hpp"
#include <string>
#include <windows.h>

namespace dxait {

//RDNA2 (gfx1030/1031) device-id range: 0x73A0 - 0x73DF.
static bool is_rdna2(uint32_t vendor_id, uint32_t device_id) {
    if (vendor_id != 0x1002) return false;
    return (device_id >= 0x73A0 && device_id <= 0x73DF);
}

//Load right D3D12CreateDevice entry point runtime:
//RDNA2 -> system d3d12.dll (Agility not supported there).
//others -> Agility D3D12Core.dll when deployed, else system d3d12.dll.
static decltype(&D3D12CreateDevice) load_d3d12_create(uint32_t vendor_id, uint32_t device_id) {
    static decltype(&D3D12CreateDevice) sys_fn = nullptr;
    static decltype(&D3D12CreateDevice) ag_fn = nullptr;

    if (!sys_fn) {
        HMODULE m = LoadLibraryA("d3d12.dll");
        if (m) sys_fn = reinterpret_cast<decltype(&D3D12CreateDevice)>((void*)GetProcAddress(m, "D3D12CreateDevice"));
    }

    if (is_rdna2(vendor_id, device_id)) return sys_fn; //Agility not supported on RDNA2: system d3d12.

    if (!ag_fn) {
        HMODULE m = LoadLibraryA("D3D12/D3D12Core.dll");
        if (m) ag_fn = reinterpret_cast<decltype(&D3D12CreateDevice)>((void*)GetProcAddress(m, "D3D12CreateDevice"));
    }
    return ag_fn ? ag_fn : sys_fn;
}

//System D3D12 entry (no Agility), used when Agility DLL absent.
HRESULT system_d3d12_create_device(IDXGIAdapter1* adapter, ID3D12Device** out) {
    static decltype(&D3D12CreateDevice) fn = nullptr;
    if (!fn) {
        HMODULE m = LoadLibraryA("d3d12.dll");
        if (!m) return E_FAIL;
        fn = reinterpret_cast<decltype(&D3D12CreateDevice)>((void*)GetProcAddress(m, "D3D12CreateDevice"));
    }
    if (!fn) return E_FAIL;
    return fn(adapter, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(out));
}

//Runtime arch-conditional device creation: RDNA2 system, else Agility.
HRESULT arch_conditional_create_device(IDXGIAdapter1* adapter, ID3D12Device** out) {
    DXGI_ADAPTER_DESC1 desc{};
    if (FAILED(adapter->GetDesc1(&desc))) return E_FAIL;
    auto fn = load_d3d12_create(desc.VendorId, desc.DeviceId);
    if (!fn) return E_FAIL;
    return fn(adapter, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(out));
}

//Classify vendor/device into ArchitectureFamily by device ID, NOT wave-lane counts:
//the RX 9070 (RDNA4, 0x7440) reports the same WaveOps / 32-64 lane range as the RX 6700
//(RDNA2), so a wave-min/max heuristic mislabels RDNA4 as RDNA2 and wrongly clears
//wmma_supported, bypassing the RDNA4 matrix path.
ArchitectureFamily classify_gpu_architecture(uint32_t vendor_id, uint32_t device_id) {
    if (vendor_id == 0x10DE) return ArchitectureFamily::NVIDIA_Ampere_Ada_Blackwell;
    if (vendor_id == 0x8086) return ArchitectureFamily::Intel_Arc_Xe;
    if (vendor_id != 0x1002) return ArchitectureFamily::Unknown; //AMD
    //   RDNA2 (gfx1031): 0x73A0-0x73DF -> dot4 only, no WMMA, Wave64 preferred
    //   RDNA3 (gfx1100):  0x7400-0x74BF -> dot2add + WMMA, Wave32
    //   RDNA4 (gfx1201):  0x74C0-0x7FFF -> dot2add + WMMA (MMA_F16), Wave32
    if (device_id >= 0x73A0 && device_id <= 0x73DF) return ArchitectureFamily::AMD_RDNA2;
    if (device_id >= 0x7400 && device_id <= 0x74BF) return ArchitectureFamily::AMD_RDNA3;
    if (device_id >= 0x74C0) return ArchitectureFamily::AMD_RDNA4;
    return ArchitectureFamily::Unknown;
}

std::vector<AdapterCaps> Adapter::enumerate() {
    std::vector<AdapterCaps> caps_list;
    ComPtr<IDXGIFactory6> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return caps_list;

    //Enumerate ALL adapters (EnumAdapters1), not GPU-preference ones.
    //headless/RDP boxes: discrete GPU may not be the display adapter, and
    //EnumAdapterByGpuPreference(HIGH_PERFORMANCE) misses it (llama.cpp uses the
    //same full-enumeration route).
    ComPtr<IDXGIAdapter1> adapter;
    for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 desc{};
        if (FAILED(adapter->GetDesc1(&desc))) continue;
        //Skip only WARP (software rasterizer); keep everything else including
        //non-display compute-only GPUs.
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;

        ComPtr<ID3D12Device> test_device;
        if (FAILED(arch_conditional_create_device(adapter.Get(), test_device.ReleaseAndGetAddressOf()))) continue;

        AdapterCaps caps{};
        char name_buf[256]{};
        WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, name_buf, sizeof(name_buf), nullptr, nullptr);
        caps.name = name_buf;
        caps.vendor_id = desc.VendorId;
        caps.device_id = desc.DeviceId;
        caps.dedicated_video_memory = desc.DedicatedVideoMemory;
        caps.shared_system_memory = desc.SharedSystemMemory;

        //Query D3D12Options1 (Wave ops, Wave min/max)
        D3D12_FEATURE_DATA_D3D12_OPTIONS1 options1{};
        if (SUCCEEDED(test_device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &options1, sizeof(options1)))) {
            caps.wave_ops = options1.WaveOps;
            caps.wave_min = options1.WaveLaneCountMin;
            caps.wave_max = options1.WaveLaneCountMax;
        }

        //Query Shader Model
        D3D12_FEATURE_DATA_SHADER_MODEL sm_data{D3D_SHADER_MODEL_6_8};
        D3D12_FEATURE_DATA_SHADER_MODEL fallback{D3D_SHADER_MODEL_6_0};
        if (FAILED(test_device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &sm_data, sizeof(sm_data)))) {
            sm_data.HighestShaderModel = D3D_SHADER_MODEL_6_0;
        }
        caps.shader_model = sm_data.HighestShaderModel;

        //Query Architecture Wave preferences (classify by device ID, not wave counts)
        caps.arch_family = classify_gpu_architecture(caps.vendor_id, caps.device_id);
        switch (caps.arch_family) {
            case ArchitectureFamily::AMD_RDNA2:
                caps.preferred_wave_size = 64; //Wave64 ~25% faster GEMV/decode on RDNA2
                caps.dot4_supported = true;
                caps.wmma_supported = false;   //RDNA2 has no WMMA matrix hardware
                break;
            case ArchitectureFamily::AMD_RDNA3:
            case ArchitectureFamily::AMD_RDNA4:
                caps.preferred_wave_size = 32; //Wave32 native on RDNA3/RDNA4
                caps.dot4_supported = true;
                caps.wmma_supported = true;    //WMMA present on RDNA3+ (MMA_F16 on RDNA4)
                break;
            default:
                caps.preferred_wave_size = (caps.wave_min <= 32 && caps.wave_max >= 64) ? 64 : 32;
                caps.dot4_supported = (caps.wave_ops != 0);
                caps.wmma_supported = false;
                break;
        }

        caps_list.push_back(caps);
    }
    return caps_list;
}

std::unique_ptr<Device> Adapter::create_device(uint32_t index) {
    ComPtr<IDXGIFactory6> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return nullptr;
    ComPtr<IDXGIAdapter1> adapter;
    if (FAILED(factory->EnumAdapters1(index, &adapter))) return nullptr;
    return std::make_unique<Device>(adapter);
}

//namespace dxait
}
