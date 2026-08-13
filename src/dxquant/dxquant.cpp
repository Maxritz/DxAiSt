#include "dxait/dxquant.hpp"
#include <cmath>
#include <algorithm>

namespace dxait {

static uint16_t float_to_fp16(float v) {
    uint32_t f32;
    std::memcpy(&f32, &v, sizeof(v));
    uint32_t sign = (f32 >> 16) & 0x8000;
    int32_t exp = ((f32 >> 23) & 0xFF) - 127 + 15;
    uint32_t frac = (f32 >> 13) & 0x03FF;
    if (exp <= 0) return static_cast<uint16_t>(sign);
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00);
    return static_cast<uint16_t>(sign | (exp << 10) | frac);
}

static float fp16_to_float(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h & 0x7C00) >> 10;
    uint32_t frac = (h & 0x03FF) << 13;
    if (exp == 0) return 0.0f;
    uint32_t f32 = sign | ((exp - 15 + 127) << 23) | frac;
    float v;
    std::memcpy(&v, &f32, sizeof(v));
    return v;
}

void quantize_row_q8_0(const float* src, BlockQ8_0* dst, uint64_t k) {
    const uint32_t nb = static_cast<uint32_t>(k / 32);
    for (uint32_t i = 0; i < nb; ++i) {
        float amax = 0.0f;
        for (int j = 0; j < 32; ++j) {
            amax = (std::max)(amax, std::abs(src[i * 32 + j]));
        }
        const float d = amax / 127.0f;
        const float id = d ? 1.0f / d : 0.0f;
        dst[i].d = float_to_fp16(d);
        for (int j = 0; j < 32; ++j) {
            int val = static_cast<int>(std::round(src[i * 32 + j] * id));
            dst[i].qs[j] = static_cast<int8_t>((std::min)(127, (std::max)(-127, val)));
        }
    }
}

void dequantize_row_q8_0(const BlockQ8_0* src, float* dst, uint64_t k) {
    const uint32_t nb = static_cast<uint32_t>(k / 32);
    for (uint32_t i = 0; i < nb; ++i) {
        const float d = fp16_to_float(src[i].d);
        for (int j = 0; j < 32; ++j) {
            dst[i * 32 + j] = src[i].qs[j] * d;
        }
    }
}

void quantize_row_q4_0(const float* src, BlockQ4_0* dst, uint64_t k) {
    const uint32_t nb = static_cast<uint32_t>(k / 32);
    for (uint32_t i = 0; i < nb; ++i) {
        float amax = 0.0f;
        for (int j = 0; j < 32; ++j) {
            amax = (std::max)(amax, std::abs(src[i * 32 + j]));
        }
        const float d = amax / -8.0f;
        const float id = d ? 1.0f / d : 0.0f;
        dst[i].d = float_to_fp16(d);
        for (int j = 0; j < 16; ++j) {
            const float x0 = src[i * 32 + j] * id;
            const float x1 = src[i * 32 + j + 16] * id;
            const uint8_t xi0 = static_cast<uint8_t>((std::min)(15, (std::max)(0, static_cast<int>(x0 + 8.5f))));
            const uint8_t xi1 = static_cast<uint8_t>((std::min)(15, (std::max)(0, static_cast<int>(x1 + 8.5f))));
            dst[i].qs[j] = xi0 | (xi1 << 4);
        }
    }
}

void dequantize_row_q4_0(const BlockQ4_0* src, float* dst, uint64_t k) {
    const uint32_t nb = static_cast<uint32_t>(k / 32);
    for (uint32_t i = 0; i < nb; ++i) {
        const float d = fp16_to_float(src[i].d);
        for (int j = 0; j < 16; ++j) {
            const uint8_t vi = src[i].qs[j];
            const int8_t x0 = (vi & 0x0F) - 8;
            const int8_t x1 = (vi >> 4) - 8;
            dst[i * 32 + j] = x0 * d;
            dst[i * 32 + j + 16] = x1 * d;
        }
    }
}

static const char g_dequant_q4_0_hlsl[] = R"(
RWStructuredBuffer<float> g_out : register(u0);
ByteAddressBuffer g_in_q4 : register(t0);

cbuffer DequantCB : register(b0) {
    uint g_num_blocks;
    uint3 g_pad;
};

[numthreads(64, 1, 1)]
void dequant_q4_0(uint3 id : SV_DispatchThreadID) {
    if (id.x >= g_num_blocks) return;

    uint block_idx = id.x;
    uint byte_offset = block_idx * 18; // 2 bytes scale + 16 bytes qs
    uint d_raw = g_in_q4.Load(byte_offset) & 0xFFFF;

    uint out_base = block_idx * 32;
    for (uint j = 0; j < 16; ++j) {
        uint val = (g_in_q4.Load(byte_offset + 2 + j) & 0xFF);
        int x0 = int(val & 0x0F) - 8;
        int x1 = int(val >> 4) - 8;
        g_out[out_base + j] = float(x0);
        g_out[out_base + j + 16] = float(x1);
    }
}
)";

QuantOps::QuantOps(Device* device) : m_device(device), m_pso_cache(device->get()) {
    init_root_signature();

    if (FAILED(device->get()->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&m_cmd_alloc)))) {
        throw std::runtime_error("CreateCommandAllocator failed");
    }
    if (FAILED(device->get()->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, m_cmd_alloc.Get(), nullptr, IID_PPV_ARGS(&m_cmd_list)))) {
        throw std::runtime_error("CreateCommandList failed");
    }
    m_cmd_list->Close();
}

void QuantOps::init_root_signature() {
    D3D12_ROOT_PARAMETER params[3]{};

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

    D3D12_ROOT_SIGNATURE_DESC desc{};
    desc.NumParameters = 3;
    desc.pParameters = params;
    desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

    ShaderCompiler compiler;
    m_root_sig = compiler.create_root_signature(m_device->get(), desc);
}

void QuantOps::dequantize_q4_0_gpu(
    Queue* queue,
    Buffer* out_buf,
    Buffer* in_q4_buf,
    uint32_t num_blocks
) {
    auto pso = m_pso_cache.get_or_compile("dequant_q4_0", g_dequant_q4_0_hlsl, m_root_sig.Get(), L"dequant_q4_0");

    m_cmd_alloc->Reset();
    m_cmd_list->Reset(m_cmd_alloc.Get(), pso.Get());
    m_cmd_list->SetComputeRootSignature(m_root_sig.Get());

    struct CB {
        uint32_t num_blocks;
        uint32_t pad[3];
    } cb{num_blocks, {0, 0, 0}};

    m_cmd_list->SetComputeRoot32BitConstants(0, 4, &cb, 0);
    m_cmd_list->SetComputeRootUnorderedAccessView(1, out_buf->get()->GetGPUVirtualAddress());
    m_cmd_list->SetComputeRootShaderResourceView(2, in_q4_buf->get()->GetGPUVirtualAddress());

    uint32_t dispatch_x = (num_blocks + 63) / 64;
    m_cmd_list->Dispatch(dispatch_x, 1, 1);

    m_cmd_list->Close();
    ID3D12CommandList* lists[] = { m_cmd_list.Get() };
    queue->execute(lists, 1);
}

} // namespace dxait
