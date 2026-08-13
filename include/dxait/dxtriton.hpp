#ifndef DXAIT_DXTRITON_HPP
#define DXAIT_DXTRITON_HPP

#include "dxait.hpp"
#include "dxjit.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

namespace dxait {

struct TritonKernelSpec {
    std::string kernel_name;
    uint32_t block_m{64};
    uint32_t block_n{64};
    uint32_t block_k{32};
    uint32_t num_warps{4};
    uint32_t wave_size{32};
    bool use_wmma{false};
    bool use_fp16{true};
};

class TritonJIT {
public:
    explicit TritonJIT(Device* device);
    ~TritonJIT();

    // 1. Generate HLSL Compute Shader Source from Triton Kernel Specification
    std::string generate_triton_matmul_hlsl(const TritonKernelSpec& spec);
    std::string generate_triton_attention_hlsl(const TritonKernelSpec& spec);

    // 2. JIT Compile & Create Pipeline State Object (PSO)
    ComPtr<ID3D12PipelineState> compile_triton_kernel(
        const std::string& hlsl_source,
        const std::string& entry_point,
        ID3D12RootSignature* root_sig
    );

    // 3. Dispatch JIT Compiled Triton Compute Kernel
    void dispatch_triton_kernel(
        Queue* queue,
        ID3D12PipelineState* pso,
        ID3D12RootSignature* root_sig,
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

    void init_root_signature();
    // ponytail: Triton-to-HLSL code generation template; upgrade path is full LLVM-based Triton IR to HLSL/DXIL backend compiler
};

} // namespace dxait

#endif // DXAIT_DXTRITON_HPP
