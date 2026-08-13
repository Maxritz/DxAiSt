#ifndef DXAIT_DXJIT_HPP
#define DXAIT_DXJIT_HPP

#include "dxait.hpp"
#include <dxcapi.h>

#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <mutex>

namespace dxait {

struct ShaderCompileMacro {
    std::wstring name;
    std::wstring value;
};

class ShaderCompiler {
public:
    ShaderCompiler();
    ~ShaderCompiler() = default;

    ComPtr<IDxcBlob> compile_hlsl(
        const std::string& hlsl_source,
        const std::wstring& entry_point = L"main",
        const std::wstring& target_profile = L"cs_6_6",
        const std::vector<ShaderCompileMacro>& macros = {}
    );

    ComPtr<ID3D12PipelineState> create_compute_pso(
        ID3D12Device* device,
        ID3D12RootSignature* root_sig,
        IDxcBlob* dxil_blob
    );

    ComPtr<ID3D12RootSignature> create_root_signature(
        ID3D12Device* device,
        const D3D12_ROOT_SIGNATURE_DESC& desc
    );

private:
    ComPtr<IDxcUtils> m_utils;
    ComPtr<IDxcCompiler3> m_compiler;
};

class PipelineCache {
public:
    explicit PipelineCache(ID3D12Device* device);
    ~PipelineCache() = default;

    ComPtr<ID3D12PipelineState> get_or_compile(
        const std::string& key,
        const std::string& hlsl_source,
        ID3D12RootSignature* root_sig,
        const std::wstring& entry = L"main",
        const std::vector<ShaderCompileMacro>& macros = {}
    );

private:
    ID3D12Device* m_device;
    ShaderCompiler m_compiler;
    std::unordered_map<std::string, ComPtr<ID3D12PipelineState>> m_cache;
    std::mutex m_mutex;
};

} // namespace dxait

#endif // DXAIT_DXJIT_HPP
