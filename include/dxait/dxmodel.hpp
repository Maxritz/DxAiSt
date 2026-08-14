#ifndef DXAIT_DXMODEL_HPP
#define DXAIT_DXMODEL_HPP

#include "dxait.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

namespace dxait {

enum class TensorDataType {
    FP32 = 0,
    FP16,
    BF16,
    FP8_E4M3,
    FP8_E5M2,
    Q8_0,
    Q4_0,
    Q4_K,
    Q5_K,
    Q6_K,
    AWQ_INT4,
    GPTQ_INT4,
    INT8
};

enum class ModelFormat {
    Unknown = 0,
    GGUF,
    Safetensors
};

enum class AttentionType {
    MHA = 0, // Multi-Head Attention
    GQA,     // Grouped-Query Attention
    MQA,     // Multi-Query Attention
    SWA,     // Sliding Window Attention
    PagedAttn,
    FlashAttn2,
    MLA,     // DeepSeek Multi-Head Latent Attention (Compressed KV latent c_KV + decoupled RoPE q^R, k^R)
    MTP,     // DeepSeek-V3 Multi-Token Prediction (Depth-d parallel draft heads)
    dflash   // DeepSeek-R1 DFlash Sparse Latent Attention
};

struct TensorInfo {
    std::string name;
    TensorDataType type{TensorDataType::FP32};
    std::vector<uint64_t> dims;
    uint64_t offset_bytes{0};
    uint64_t size_bytes{0};
};

struct ModelArchConfig {
    std::string arch_name{"deepseek_v3"};
    AttentionType attention_type{AttentionType::MLA};
    uint32_t num_layers{61};
    uint32_t num_mtp_depth{1};        // DeepSeek-V3 MTP speculative token depth (d=1)
    uint32_t kv_compressed_dim{512};  // MLA c_KV latent compression dimension
    uint32_t q_compressed_dim{1536}; // MLA c_Q latent compression dimension
    uint32_t rope_head_dim{64};       // Decoupled RoPE dimension
    uint32_t nop_head_dim{128};      // Non-positional projection dimension
};

class ModelLoader {
public:
    ModelLoader() = default;
    ~ModelLoader() = default;

    bool load_file(const std::string& filepath);
    bool parse_gguf(const std::string& filepath);
    bool parse_safetensors(const std::string& filepath);

    ModelFormat format() const { return m_format; }
    const ModelArchConfig& arch_config() const { return m_arch_config; }
    const std::unordered_map<std::string, TensorInfo>& tensors() const { return m_tensors; }
    uint32_t tensor_count() const { return static_cast<uint32_t>(m_tensors.size()); }

private:
    ModelFormat m_format{ModelFormat::Unknown};
    std::string m_filepath;
    ModelArchConfig m_arch_config{};
    std::unordered_map<std::string, TensorInfo> m_tensors;
};

} // namespace dxait

#endif // DXAIT_DXMODEL_HPP
