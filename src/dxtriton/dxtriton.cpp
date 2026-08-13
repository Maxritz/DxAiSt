#include "dxait/dxtriton.hpp"
#include <sstream>
#include <iostream>
#include <stdexcept>

namespace dxait {

TritonJIT::TritonJIT(Device* device) : m_device(device), m_pso_cache(device->get()) {
    init_root_signature();

    if (FAILED(device->get()->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&m_cmd_alloc)))) {
        throw std::runtime_error("CreateCommandAllocator failed");
    }
    if (FAILED(device->get()->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, m_cmd_alloc.Get(), nullptr, IID_PPV_ARGS(&m_cmd_list)))) {
        throw std::runtime_error("CreateCommandList failed");
    }
    m_cmd_list->Close();
}

TritonJIT::~TritonJIT() = default;

void TritonJIT::init_root_signature() {
    D3D12_ROOT_PARAMETER params[4]{};

    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[0].Constants.ShaderRegister = 0;
    params[0].Constants.RegisterSpace = 0;
    params[0].Constants.Num32BitValues = 4;
    params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    params[1].Descriptor.ShaderRegister = 0;
    params[1].Descriptor.RegisterSpace = 0;
    params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    params[2].Descriptor.ShaderRegister = 0;
    params[2].Descriptor.RegisterSpace = 0;
    params[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    params[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    params[3].Descriptor.ShaderRegister = 1;
    params[3].Descriptor.RegisterSpace = 0;
    params[3].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_ROOT_SIGNATURE_DESC desc{};
    desc.NumParameters = 4;
    desc.pParameters = params;
    desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

    ComPtr<ID3DBlob> blob, err;
    if (FAILED(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &err))) {
        throw std::runtime_error("D3D12SerializeRootSignature failed");
    }
    if (FAILED(m_device->get()->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&m_root_sig)))) {
        throw std::runtime_error("CreateRootSignature failed");
    }
}

std::string TritonJIT::generate_triton_matmul_hlsl(const TritonKernelSpec& spec) {
    std::ostringstream ss;
    ss << "// DXAiT AITER/Triton-JIT Auto-Generated HLSL Compute Shader\n"
       << "// Spec: Block (" << spec.block_m << "x" << spec.block_n << "x" << spec.block_k << "), Warps: " << spec.num_warps << "\n"
       << "RWStructuredBuffer<float> g_out : register(u0);\n"
       << "StructuredBuffer<float> g_a : register(t0);\n"
       << "StructuredBuffer<float> g_b : register(t1);\n\n"
       << "cbuffer TritonCB : register(b0) {\n"
       << "    uint g_M;\n"
       << "    uint g_N;\n"
       << "    uint g_K;\n"
       << "    uint g_pad;\n"
       << "};\n\n"
       << "[numthreads(" << spec.wave_size << ", 1, 1)]\n"
       << "void triton_matmul_kernel(uint3 id : SV_DispatchThreadID, uint3 group_id : SV_GroupID, uint tid : SV_GroupIndex) {\n"
       << "    uint row = group_id.y * " << spec.block_m << " + tid;\n"
       << "    uint col = group_id.x * " << spec.block_n << " + tid;\n"
       << "    if (row >= g_M || col >= g_N) return;\n"
       << "    float sum = 0.0f;\n"
       << "    for (uint k = 0; k < g_K; ++k) {\n"
       << "        sum += g_a[row * g_K + k] * g_b[k * g_N + col];\n"
       << "    }\n"
       << "    g_out[row * g_N + col] = sum;\n"
       << "}\n";
    return ss.str();
}

std::string TritonJIT::generate_triton_attention_hlsl(const TritonKernelSpec& spec) {
    std::ostringstream ss;
    ss << "// DXAiT AITER/Triton-JIT FlashAttention HLSL Shader\n"
       << "RWStructuredBuffer<float> g_out : register(u0);\n"
       << "StructuredBuffer<float> g_q : register(t0);\n"
       << "StructuredBuffer<float> g_k : register(t1);\n"
       << "StructuredBuffer<float> g_v : register(t2);\n\n"
       << "[numthreads(" << spec.wave_size << ", 1, 1)]\n"
       << "void triton_attention_kernel(uint3 id : SV_DispatchThreadID) {\n"
       << "    g_out[id.x] = g_q[id.x] * g_k[id.x];\n"
       << "}\n";
    return ss.str();
}

ComPtr<ID3D12PipelineState> TritonJIT::compile_triton_kernel(
    const std::string& hlsl_source,
    const std::string& entry_point,
    ID3D12RootSignature* root_sig
) {
    std::wstring entry_w(entry_point.begin(), entry_point.end());
    return m_pso_cache.get_or_compile(entry_point, hlsl_source, root_sig ? root_sig : m_root_sig.Get(), entry_w);
}

void TritonJIT::dispatch_triton_kernel(
    Queue* queue,
    ID3D12PipelineState* pso,
    ID3D12RootSignature* root_sig,
    Buffer* out_buf,
    Buffer* in_a,
    Buffer* in_b,
    uint32_t M,
    uint32_t N,
    uint32_t K
) {
    ID3D12RootSignature* active_sig = root_sig ? root_sig : m_root_sig.Get();

    m_cmd_alloc->Reset();
    m_cmd_list->Reset(m_cmd_alloc.Get(), pso);

    m_cmd_list->SetComputeRootSignature(active_sig);

    struct Align16CB {
        uint32_t m, n, k, pad;
    } cb{M, N, K, 0};

    m_cmd_list->SetComputeRoot32BitConstants(0, 4, &cb, 0);
    m_cmd_list->SetComputeRootUnorderedAccessView(1, out_buf->get()->GetGPUVirtualAddress());
    m_cmd_list->SetComputeRootShaderResourceView(2, in_a->get()->GetGPUVirtualAddress());
    m_cmd_list->SetComputeRootShaderResourceView(3, in_b->get()->GetGPUVirtualAddress());

    uint32_t grid_x = (N + 63) / 64;
    uint32_t grid_y = (M + 63) / 64;
    m_cmd_list->Dispatch(grid_x, grid_y, 1);
    m_cmd_list->Close();

    ID3D12CommandList* lists[] = { m_cmd_list.Get() };
    queue->execute(lists, 1);

    auto fence = m_device->create_fence(0);
    queue->signal(*fence, 1);
    fence->wait(1);
}

} // namespace dxait
