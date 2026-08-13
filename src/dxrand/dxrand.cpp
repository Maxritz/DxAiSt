#include "dxait/dxrand.hpp"

namespace dxait {

static const char g_rand_hlsl[] = R"(
RWStructuredBuffer<float> g_out : register(u0);

cbuffer RandCB : register(b0) {
    uint g_count;
    uint g_seed_lo;
    uint g_seed_hi;
    uint g_pad;
};

// PCG32 PRNG
uint pcg32(inout uint state) {
    uint oldstate = state;
    state = oldstate * 747796405u + 2891336453u;
    uint word = ((oldstate >> ((oldstate >> 28u) + 4u)) ^ oldstate) * 277803737u;
    return (word >> 22u) ^ word;
}

[numthreads(64, 1, 1)]
void rand_fill(uint3 id : SV_DispatchThreadID) {
    if (id.x < g_count) {
        uint state = g_seed_lo + id.x * 1664525u;
        uint r = pcg32(state);
        g_out[id.x] = (float)r / 4294967296.0f;
    }
}
)";

RandomGenerator::RandomGenerator(Device* device) : m_device(device), m_pso_cache(device->get()) {
    init_root_signature();

    if (FAILED(device->get()->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&m_cmd_alloc)))) {
        throw std::runtime_error("CreateCommandAllocator failed");
    }
    if (FAILED(device->get()->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, m_cmd_alloc.Get(), nullptr, IID_PPV_ARGS(&m_cmd_list)))) {
        throw std::runtime_error("CreateCommandList failed");
    }
    m_cmd_list->Close();
}

void RandomGenerator::init_root_signature() {
    D3D12_ROOT_PARAMETER params[2]{};

    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[0].Constants.ShaderRegister = 0;
    params[0].Constants.RegisterSpace = 0;
    params[0].Constants.Num32BitValues = 4;
    params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    params[1].Descriptor.ShaderRegister = 0;
    params[1].Descriptor.RegisterSpace = 0;
    params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_ROOT_SIGNATURE_DESC desc{};
    desc.NumParameters = 2;
    desc.pParameters = params;
    desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

    ShaderCompiler compiler;
    m_root_sig = compiler.create_root_signature(m_device->get(), desc);
}

void RandomGenerator::fill_uniform(
    Queue* queue,
    Buffer* out_buf,
    uint32_t count,
    uint64_t seed
) {
    auto pso = m_pso_cache.get_or_compile("rand_fill", g_rand_hlsl, m_root_sig.Get(), L"rand_fill");

    m_cmd_alloc->Reset();
    m_cmd_list->Reset(m_cmd_alloc.Get(), pso.Get());
    m_cmd_list->SetComputeRootSignature(m_root_sig.Get());

    struct CB {
        uint32_t count;
        uint32_t seed_lo;
        uint32_t seed_hi;
        uint32_t pad;
    } cb{count, static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32), 0};

    m_cmd_list->SetComputeRoot32BitConstants(0, 4, &cb, 0);
    m_cmd_list->SetComputeRootUnorderedAccessView(1, out_buf->get()->GetGPUVirtualAddress());

    uint32_t dispatch_x = (count + 63) / 64;
    m_cmd_list->Dispatch(dispatch_x, 1, 1);

    m_cmd_list->Close();
    ID3D12CommandList* lists[] = { m_cmd_list.Get() };
    queue->execute(lists, 1);
}

} // namespace dxait
