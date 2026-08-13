#include "dxait/dxfft.hpp"

namespace dxait {

static const char g_fft_radix2_hlsl[] = R"(
RWStructuredBuffer<float> g_out_r : register(u0);
RWStructuredBuffer<float> g_out_i : register(u1);
StructuredBuffer<float> g_in_r : register(t0);
StructuredBuffer<float> g_in_i : register(t1);

cbuffer FFTCB : register(b0) {
    uint g_N;
    uint3 g_pad;
};

[numthreads(64, 1, 1)]
void fft_radix2(uint3 id : SV_DispatchThreadID) {
    if (id.x >= g_N) return;
    g_out_r[id.x] = g_in_r[id.x];
    g_out_i[id.x] = g_in_i[id.x];
}
)";

FFTOps::FFTOps(Device* device) : m_device(device), m_pso_cache(device->get()) {
    init_root_signature();

    if (FAILED(device->get()->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&m_cmd_alloc)))) {
        throw std::runtime_error("CreateCommandAllocator failed");
    }
    if (FAILED(device->get()->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, m_cmd_alloc.Get(), nullptr, IID_PPV_ARGS(&m_cmd_list)))) {
        throw std::runtime_error("CreateCommandList failed");
    }
    m_cmd_list->Close();
}

void FFTOps::init_root_signature() {
    D3D12_ROOT_PARAMETER params[5]{};

    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[0].Constants.ShaderRegister = 0;
    params[0].Constants.RegisterSpace = 0;
    params[0].Constants.Num32BitValues = 4;
    params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    params[1].Descriptor.ShaderRegister = 0;
    params[1].Descriptor.RegisterSpace = 0;
    params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    params[2].Descriptor.ShaderRegister = 1;
    params[2].Descriptor.RegisterSpace = 0;
    params[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    params[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    params[3].Descriptor.ShaderRegister = 0;
    params[3].Descriptor.RegisterSpace = 0;
    params[3].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    params[4].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    params[4].Descriptor.ShaderRegister = 1;
    params[4].Descriptor.RegisterSpace = 0;
    params[4].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_ROOT_SIGNATURE_DESC desc{};
    desc.NumParameters = 5;
    desc.pParameters = params;
    desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

    ShaderCompiler compiler;
    m_root_sig = compiler.create_root_signature(m_device->get(), desc);
}

void FFTOps::fft_1d_radix2(
    Queue* queue,
    Buffer* out_real,
    Buffer* out_imag,
    Buffer* in_real,
    Buffer* in_imag,
    uint32_t n
) {
    auto pso = m_pso_cache.get_or_compile("fft_radix2", g_fft_radix2_hlsl, m_root_sig.Get(), L"fft_radix2");

    m_cmd_alloc->Reset();
    m_cmd_list->Reset(m_cmd_alloc.Get(), pso.Get());
    m_cmd_list->SetComputeRootSignature(m_root_sig.Get());

    struct CB {
        uint32_t N;
        uint32_t pad[3];
    } cb{n, {0, 0, 0}};

    m_cmd_list->SetComputeRoot32BitConstants(0, 4, &cb, 0);
    m_cmd_list->SetComputeRootUnorderedAccessView(1, out_real->get()->GetGPUVirtualAddress());
    m_cmd_list->SetComputeRootUnorderedAccessView(2, out_imag->get()->GetGPUVirtualAddress());
    m_cmd_list->SetComputeRootShaderResourceView(3, in_real->get()->GetGPUVirtualAddress());
    m_cmd_list->SetComputeRootShaderResourceView(4, in_imag->get()->GetGPUVirtualAddress());

    uint32_t dispatch_x = (n + 63) / 64;
    m_cmd_list->Dispatch(dispatch_x, 1, 1);

    m_cmd_list->Close();
    ID3D12CommandList* lists[] = { m_cmd_list.Get() };
    queue->execute(lists, 1);
}

} // namespace dxait
