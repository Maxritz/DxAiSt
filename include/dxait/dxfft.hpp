#ifndef DXAIT_DXFFT_HPP
#define DXAIT_DXFFT_HPP

#include "dxait.hpp"
#include "dxjit.hpp"

namespace dxait {

class FFTOps {
public:
    explicit FFTOps(Device* device);
    ~FFTOps() = default;

    // Iterative radix-2 Cooley-Tukey FFT. n must be a power of two.
    // Requires a fence wait per call (serialised), see dxmath for the ring-buffer upgrade path.
    void fft_1d_radix2(
        Queue* queue,
        Buffer* out_real,
        Buffer* out_imag,
        Buffer* in_real,
        Buffer* in_imag,
        uint32_t n
    );

private:
    Device* m_device;
    PipelineCache m_pso_cache;
    ComPtr<ID3D12RootSignature> m_root_sig;
    ComPtr<ID3D12CommandAllocator> m_cmd_alloc;
    ComPtr<ID3D12GraphicsCommandList> m_cmd_list;
    std::unique_ptr<Fence> m_fence;
    uint64_t m_fence_val{0};

    void init_root_signature();
};

} // namespace dxait

#endif // DXAIT_DXFFT_HPP
