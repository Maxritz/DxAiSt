#include "dxait/dxjit.hpp"
#include <stdexcept>

namespace dxait {

ShaderCompiler::ShaderCompiler() {
    if (FAILED(DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(&m_utils)))) {
        throw std::runtime_error("Failed to create IDxcUtils");
    }
    if (FAILED(DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&m_compiler)))) {
        throw std::runtime_error("Failed to create IDxcCompiler3");
    }
}

ComPtr<IDxcBlob> ShaderCompiler::compile_hlsl(
    const std::string& hlsl_source,
    const std::wstring& entry_point,
    const std::wstring& target_profile,
    const std::vector<ShaderCompileMacro>& macros
) {
    ComPtr<IDxcBlobEncoding> source_blob;
    if (FAILED(m_utils->CreateBlob(hlsl_source.data(), static_cast<UINT32>(hlsl_source.size()), CP_UTF8, &source_blob))) {
        throw std::runtime_error("Failed to create source blob");
    }

    std::vector<LPCWSTR> arguments;
    arguments.push_back(L"-E");
    arguments.push_back(entry_point.c_str());
    arguments.push_back(L"-T");
    arguments.push_back(target_profile.c_str());
    arguments.push_back(DXC_ARG_OPTIMIZATION_LEVEL3);

    std::vector<std::wstring> macro_strings;
    for (const auto& m : macros) {
        macro_strings.push_back(m.name + L"=" + m.value);
    }
    for (const auto& ms : macro_strings) {
        arguments.push_back(L"-D");
        arguments.push_back(ms.c_str());
    }

    DxcBuffer source_buffer{};
    source_buffer.Ptr = source_blob->GetBufferPointer();
    source_buffer.Size = source_blob->GetBufferSize();
    source_buffer.Encoding = CP_UTF8;

    ComPtr<IDxcResult> result;
    if (FAILED(m_compiler->Compile(&source_buffer, arguments.data(), static_cast<UINT32>(arguments.size()), nullptr, IID_PPV_ARGS(&result)))) {
        throw std::runtime_error("DXC compilation failed");
    }

    HRESULT status;
    result->GetStatus(&status);
    if (FAILED(status)) {
        ComPtr<IDxcBlobUtf8> errors;
        result->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&errors), nullptr);
        if (errors && errors->GetStringLength() > 0) {
            throw std::runtime_error(std::string("Shader compilation errors:\n") + errors->GetStringPointer());
        }
        throw std::runtime_error("Shader compilation failed without error string");
    }

    ComPtr<IDxcBlob> shader_blob;
    result->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&shader_blob), nullptr);
    return shader_blob;
}

ComPtr<ID3D12PipelineState> ShaderCompiler::create_compute_pso(
    ID3D12Device* device,
    ID3D12RootSignature* root_sig,
    IDxcBlob* dxil_blob
) {
    D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
    desc.pRootSignature = root_sig;
    desc.CS.pShaderBytecode = dxil_blob->GetBufferPointer();
    desc.CS.BytecodeLength = dxil_blob->GetBufferSize();
    desc.NodeMask = 0;

    ComPtr<ID3D12PipelineState> pso;
    if (FAILED(device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&pso)))) {
        throw std::runtime_error("CreateComputePipelineState failed");
    }
    return pso;
}

ComPtr<ID3D12RootSignature> ShaderCompiler::create_root_signature(
    ID3D12Device* device,
    const D3D12_ROOT_SIGNATURE_DESC& desc
) {
    ComPtr<ID3DBlob> sig_blob;
    ComPtr<ID3DBlob> err_blob;
    if (FAILED(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &sig_blob, &err_blob))) {
        if (err_blob) {
            throw std::runtime_error(static_cast<const char*>(err_blob->GetBufferPointer()));
        }
        throw std::runtime_error("D3D12SerializeRootSignature failed");
    }

    ComPtr<ID3D12RootSignature> root_sig;
    if (FAILED(device->CreateRootSignature(0, sig_blob->GetBufferPointer(), sig_blob->GetBufferSize(), IID_PPV_ARGS(&root_sig)))) {
        throw std::runtime_error("CreateRootSignature failed");
    }
    return root_sig;
}

PipelineCache::PipelineCache(ID3D12Device* device) : m_device(device) {}

ComPtr<ID3D12PipelineState> PipelineCache::get_or_compile(
    const std::string& key,
    const std::string& hlsl_source,
    ID3D12RootSignature* root_sig,
    const std::wstring& entry,
    const std::vector<ShaderCompileMacro>& macros
) {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_cache.find(key);
    if (it != m_cache.end()) return it->second;

    auto blob = m_compiler.compile_hlsl(hlsl_source, entry, L"cs_6_6", macros);
    auto pso = m_compiler.create_compute_pso(m_device, root_sig, blob.Get());
    m_cache[key] = pso;
    return pso;
}

} // namespace dxait
