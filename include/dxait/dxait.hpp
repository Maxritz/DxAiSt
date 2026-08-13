#ifndef DXAIT_HPP
#define DXAIT_HPP

#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <system_error>

namespace dxait {

using Microsoft::WRL::ComPtr;

// System D3D12 fallback (no Agility): used when the Agility DLL is absent.
HRESULT system_d3d12_create_device(IDXGIAdapter1* adapter, ID3D12Device** out);

// Runtime arch-conditional: RDNA2 -> system d3d12.dll, others -> Agility if deployed.
HRESULT arch_conditional_create_device(IDXGIAdapter1* adapter, ID3D12Device** out);

enum class QueueType {
    Direct = D3D12_COMMAND_LIST_TYPE_DIRECT,
    Compute = D3D12_COMMAND_LIST_TYPE_COMPUTE,
    Copy = D3D12_COMMAND_LIST_TYPE_COPY
};

enum class MemLocation {
    Default = D3D12_HEAP_TYPE_DEFAULT,
    Upload = D3D12_HEAP_TYPE_UPLOAD,
    Readback = D3D12_HEAP_TYPE_READBACK,
    ReBAR = 3 // Custom Heap: CPU Write-Combine directly in VRAM (PCIe ReBAR)
};

enum class ArchitectureFamily {
    Unknown = 0,
    AMD_RDNA2, // Wave32/64 capable, sdot4/sudot4, v_dot2, no WMMA. GEMV prefers Wave64.
    AMD_RDNA3, // Wave32, WMMA capable.
    AMD_RDNA4, // Wave32 native, WMMA capable (BEST_FATTN_KERNEL_MMA_F16 optimized).
    NVIDIA_Ampere_Ada_Blackwell, // Wave32, Tensor Cores.
    Intel_Arc_Xe // Wave16/32, XMX.
};

struct AdapterCaps {
    std::string name;
    uint32_t vendor_id{0};
    uint32_t device_id{0};
    uint64_t dedicated_video_memory{0};
    uint64_t shared_system_memory{0};
    D3D_FEATURE_LEVEL feature_level{D3D_FEATURE_LEVEL_11_0};
    D3D_SHADER_MODEL shader_model{D3D_SHADER_MODEL_6_0};
    uint32_t wave_min{0};
    uint32_t wave_max{0};
    bool wave_ops{false};
    bool fp16{false};
    bool int8{false};
    bool wmma_supported{false};
    bool dot4_supported{false};
    bool enhanced_barriers{false};
    bool work_graphs{false};
    bool direct_storage{true};
    bool ray_tracing{false};
    bool mesh_shaders{false};
    bool vrs_tier2{false};
    ArchitectureFamily arch_family{ArchitectureFamily::Unknown};
    uint32_t preferred_wave_size{32};
};

class Config {
public:
    static uint64_t get_chunk_size_bytes();
    static void set_chunk_size_bytes(uint64_t bytes);

    static double get_vram_margin_ratio();
    static void set_vram_margin_ratio(double ratio);

    static bool is_async_prefetch_enabled();
    static void set_async_prefetch_enabled(bool enabled);
};

// GPU TDR (Timeout Detection & Recovery) Blackout Safeguard Engine
class TDRGuard {
public:
    static void yield_gpu_breather(ID3D12CommandQueue* queue, ID3D12Fence* fence, uint64_t& fence_val);
    static bool is_device_removed(ID3D12Device* device);
};

class Device;

class Adapter {
public:
    static std::vector<AdapterCaps> enumerate();
    static std::unique_ptr<Device> create_device(uint32_t index = 0);
};

class Fence {
public:
    Fence(ID3D12Device* device, uint64_t initial_val = 0);
    ~Fence() = default;

    void signal(ID3D12CommandQueue* queue, uint64_t val);
    void wait(uint64_t val);
    bool is_completed(uint64_t val) const;

private:
    ComPtr<ID3D12Fence> m_fence;
    HANDLE m_event{nullptr};
};

class Buffer {
public:
    Buffer(ID3D12Device* device, uint64_t size_bytes, MemLocation loc);
    ~Buffer() = default;

    ID3D12Resource* get() const { return m_resource.Get(); }
    uint64_t size() const { return m_size; }
    void* map();
    void unmap();
    bool make_resident(ID3D12Device* device);
    bool evict(ID3D12Device* device);

private:
    ComPtr<ID3D12Resource> m_resource;
    uint64_t m_size{0};
    MemLocation m_location;
};

class Queue {
public:
    Queue(ID3D12Device* device, QueueType type);
    ~Queue() = default;

    void execute(ID3D12CommandList* const* lists, uint32_t count);
    void signal(Fence& fence, uint64_t value);
    void wait(Fence& fence, uint64_t value);
    ID3D12CommandQueue* get() const { return m_queue.Get(); }

private:
    ComPtr<ID3D12CommandQueue> m_queue;
};

class Device {
public:
    explicit Device(ComPtr<IDXGIAdapter1> adapter);
    ~Device() = default;

    ID3D12Device* get() const { return m_device.Get(); }
    const AdapterCaps& caps() const { return m_caps; }

    std::unique_ptr<Buffer> create_buffer(uint64_t size_bytes, MemLocation loc = MemLocation::Default);
    std::unique_ptr<Queue> create_queue(QueueType type = QueueType::Direct);
    std::unique_ptr<Fence> create_fence(uint64_t initial_val = 0);

private:
    ComPtr<IDXGIAdapter1> m_adapter;
    ComPtr<ID3D12Device> m_device;
    AdapterCaps m_caps;
};

} // namespace dxait

#endif // DXAIT_HPP
