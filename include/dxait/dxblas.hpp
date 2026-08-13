#ifndef DXAIT_DXBLAS_HPP
#define DXAIT_DXBLAS_HPP

#include "dxait.hpp"
#include "dxjit.hpp"

namespace dxait {

class BLAS {
public:
    explicit BLAS(Device* device);
    ~BLAS() = default;

    void vec_add(
        Queue* queue,
        Buffer* out_buf,
        Buffer* in1_buf,
        Buffer* in2_buf,
        uint32_t count,
        float alpha = 1.0f,
        float beta = 1.0f
    );

    // Cutlass-style tiled GEMM: shared memory staging, register accumulation.
    // Computes C = A * B for MxK * KxN row-major floats. Tile is 16x16.
    void gemm(
        Queue* queue,
        Buffer* out_buf,
        Buffer* in_a,
        Buffer* in_b,
        uint32_t M,
        uint32_t N,
        uint32_t K
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

#endif // DXAIT_DXBLAS_HPP
