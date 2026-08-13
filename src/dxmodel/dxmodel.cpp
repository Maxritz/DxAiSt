#include "dxait/dxmodel.hpp"
#include <windows.h>
#include <iostream>
#include <fstream>
#include <cstring>

namespace dxait {

enum GGUFType {
    GGUF_TYPE_UINT8 = 0,
    GGUF_TYPE_INT8 = 1,
    GGUF_TYPE_UINT16 = 2,
    GGUF_TYPE_INT16 = 3,
    GGUF_TYPE_UINT32 = 4,
    GGUF_TYPE_INT32 = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL = 7,
    GGUF_TYPE_STRING = 8,
    GGUF_TYPE_ARRAY = 9,
    GGUF_TYPE_UINT64 = 10,
    GGUF_TYPE_INT64 = 11,
    GGUF_TYPE_FLOAT64 = 12
};

class MemoryMappedFile {
public:
    MemoryMappedFile(const std::string& path) {
        m_file = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (m_file == INVALID_HANDLE_VALUE) return;

        LARGE_INTEGER size;
        if (!GetFileSizeEx(m_file, &size)) return;
        m_size = size.QuadPart;

        m_mapping = CreateFileMappingA(m_file, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!m_mapping) return;

        m_ptr = MapViewOfFile(m_mapping, FILE_MAP_READ, 0, 0, 0);
    }

    ~MemoryMappedFile() {
        if (m_ptr) UnmapViewOfFile(m_ptr);
        if (m_mapping) CloseHandle(m_mapping);
        if (m_file != INVALID_HANDLE_VALUE) CloseHandle(m_file);
    }

    bool is_valid() const { return m_ptr != nullptr && m_size > 0; }
    const uint8_t* data() const { return static_cast<const uint8_t*>(m_ptr); }
    uint64_t size() const { return m_size; }

private:
    HANDLE m_file{INVALID_HANDLE_VALUE};
    HANDLE m_mapping{nullptr};
    void* m_ptr{nullptr};
    uint64_t m_size{0};
};

static std::string read_gguf_string(const uint8_t* data, uint64_t max_size, uint64_t& offset) {
    if (offset + 8 > max_size) return "";
    uint64_t str_len = *reinterpret_cast<const uint64_t*>(data + offset);
    offset += 8;
    if (offset + str_len > max_size) return "";

    std::string result(reinterpret_cast<const char*>(data + offset), static_cast<size_t>(str_len));
    offset += str_len;
    return result;
}

static void skip_gguf_val(const uint8_t* data, uint64_t max_size, uint64_t& offset, uint32_t val_type) {
    switch (val_type) {
    case GGUF_TYPE_UINT8: case GGUF_TYPE_INT8: case GGUF_TYPE_BOOL: offset += 1; break;
    case GGUF_TYPE_UINT16: case GGUF_TYPE_INT16: offset += 2; break;
    case GGUF_TYPE_UINT32: case GGUF_TYPE_INT32: case GGUF_TYPE_FLOAT32: offset += 4; break;
    case GGUF_TYPE_UINT64: case GGUF_TYPE_INT64: case GGUF_TYPE_FLOAT64: offset += 8; break;
    case GGUF_TYPE_STRING: {
        if (offset + 8 <= max_size) {
            uint64_t str_len = *reinterpret_cast<const uint64_t*>(data + offset);
            offset += 8 + str_len;
        }
        break;
    }
    case GGUF_TYPE_ARRAY: {
        if (offset + 12 <= max_size) {
            uint32_t elem_type = *reinterpret_cast<const uint32_t*>(data + offset);
            uint64_t array_len = *reinterpret_cast<const uint64_t*>(data + offset + 4);
            offset += 12;
            for (uint64_t i = 0; i < array_len; ++i) {
                skip_gguf_val(data, max_size, offset, elem_type);
            }
        }
        break;
    }
    default: break;
    }
}

bool ModelLoader::load_file(const std::string& filepath) {
    m_filepath = filepath;
    if (filepath.rfind(".gguf") != std::string::npos) {
        return parse_gguf(filepath);
    } else if (filepath.rfind(".safetensors") != std::string::npos) {
        return parse_safetensors(filepath);
    } else if (filepath.rfind(".pte") != std::string::npos) {
        return parse_pte(filepath);
    }
    return parse_gguf(filepath);
}

bool ModelLoader::parse_gguf(const std::string& filepath) {
    MemoryMappedFile mmap(filepath);
    if (!mmap.is_valid()) return false;

    if (mmap.size() < 24) return false;
    const uint32_t magic = *reinterpret_cast<const uint32_t*>(mmap.data());
    if (magic != 0x46554747) return false; // "GGUF"

    const uint32_t version = *reinterpret_cast<const uint32_t*>(mmap.data() + 4);
    const uint64_t tensor_count = *reinterpret_cast<const uint64_t*>(mmap.data() + 8);
    const uint64_t kv_count = *reinterpret_cast<const uint64_t*>(mmap.data() + 16);
    (void)version;

    uint64_t offset = 24;

    for (uint64_t i = 0; i < kv_count; ++i) {
        std::string key = read_gguf_string(mmap.data(), mmap.size(), offset);
        if (offset + 4 > mmap.size()) break;
        uint32_t val_type = *reinterpret_cast<const uint32_t*>(mmap.data() + offset);
        offset += 4;
        skip_gguf_val(mmap.data(), mmap.size(), offset, val_type);
    }

    bool has_mtp_layers = false;
    bool has_mla_tensors = false;

    // Parse GGUF Tensor descriptors and detect DeepSeek MLA / MTP layers
    for (uint64_t i = 0; i < tensor_count; ++i) {
        std::string t_name = read_gguf_string(mmap.data(), mmap.size(), offset);
        if (offset + 4 > mmap.size()) break;
        uint32_t n_dims = *reinterpret_cast<const uint32_t*>(mmap.data() + offset);
        offset += 4;

        TensorInfo info;
        info.name = t_name;
        for (uint32_t d = 0; d < n_dims; ++d) {
            if (offset + 8 > mmap.size()) break;
            uint64_t dim_val = *reinterpret_cast<const uint64_t*>(mmap.data() + offset);
            offset += 8;
            info.dims.push_back(dim_val);
        }

        if (offset + 12 > mmap.size()) break;
        uint32_t ggml_type = *reinterpret_cast<const uint32_t*>(mmap.data() + offset);
        uint64_t t_offset = *reinterpret_cast<const uint64_t*>(mmap.data() + offset + 4);
        offset += 12;

        info.offset_bytes = t_offset;
        info.type = (ggml_type == 7) ? TensorDataType::Q8_0 : ((ggml_type == 2) ? TensorDataType::Q4_0 : TensorDataType::FP16);
        
        if (t_name.find("mtp") != std::string::npos || t_name.find("speculative") != std::string::npos) {
            has_mtp_layers = true;
        }
        if (t_name.find("kv_a") != std::string::npos || t_name.find("kv_b") != std::string::npos || t_name.find("q_a") != std::string::npos) {
            has_mla_tensors = true;
        }

        m_tensors[t_name] = info;
    }

    if (has_mtp_layers) {
        m_arch_config.arch_name = "deepseek_v3_mtp";
        m_arch_config.attention_type = AttentionType::MTP;
        m_arch_config.num_mtp_depth = 1;
    } else if (has_mla_tensors) {
        m_arch_config.arch_name = "deepseek_v3_mla";
        m_arch_config.attention_type = AttentionType::MLA;
        m_arch_config.kv_compressed_dim = 512;
    } else {
        m_arch_config.arch_name = "transformer_gqa";
        m_arch_config.attention_type = AttentionType::GQA;
    }

    m_format = ModelFormat::GGUF;
    std::cout << "[DXAiT ModelLoader Engine] Parsed " << m_tensors.size() << " tensors (Arch: "
              << m_arch_config.arch_name << ", AttentionType: " << static_cast<int>(m_arch_config.attention_type)
              << ") from " << filepath << "\n";
    return true;
}

bool ModelLoader::parse_safetensors(const std::string& filepath) {
    MemoryMappedFile mmap(filepath);
    if (!mmap.is_valid()) return false;
    m_format = ModelFormat::Safetensors;
    m_arch_config.arch_name = "deepseek_r1_dflash";
    m_arch_config.attention_type = AttentionType::dflash;
    return true;
}

bool ModelLoader::parse_pte(const std::string& filepath) {
    (void)filepath;
    m_format = ModelFormat::PTE;
    return true;
}

bool ModelLoader::parse_pytorch_bin(const std::string& filepath) {
    (void)filepath;
    m_format = ModelFormat::PyTorchBin;
    return true;
}

bool ModelLoader::parse_onnx(const std::string& filepath) {
    (void)filepath;
    m_format = ModelFormat::ONNX;
    return true;
}

} // namespace dxait
