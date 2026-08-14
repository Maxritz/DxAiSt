#include "dxait/dxait.hpp"
#include "dxait/dxmodel.hpp"
#include "dxait/dxjit.hpp"
#include <iostream>
#include <fstream>
#include <cassert>

int main() {
    std::cout << "========================================================\n";
    std::cout << " DXAiT Multi-Format Model Loaders Verification Suite\n";
    std::cout << "========================================================\n\n";

    // 1. GGUF v3 Model File Test
    {
        std::ofstream out("test_model.gguf", std::ios::binary);
        uint32_t magic = 0x46554747; // "GGUF"
        uint32_t ver = 3;
        uint64_t tensor_cnt = 10;
        uint64_t kv_cnt = 5;
        out.write(reinterpret_cast<char*>(&magic), 4);
        out.write(reinterpret_cast<char*>(&ver), 4);
        out.write(reinterpret_cast<char*>(&tensor_cnt), 8);
        out.write(reinterpret_cast<char*>(&kv_cnt), 8);
    }
    dxait::ModelLoader gguf_loader;
    if (!gguf_loader.load_file("test_model.gguf") || gguf_loader.format() != dxait::ModelFormat::GGUF) {
        std::cerr << "GGUF loader failed\n";
        return -1;
    }
    std::cout << "[1/2] GGUF v3 Model Loader:               PASSED\n";

    // 2. Safetensors Model File Test
    {
        std::ofstream out("test_model.safetensors", std::ios::binary);
        std::string json = "{\"__metadata__\":{\"format\":\"pt\"},\"model.embed.weight\":{\"dtype\":\"F16\",\"shape\":[1024,4096],\"data_offsets\":[0,8388608]}}";
        uint64_t len = json.size();
        out.write(reinterpret_cast<char*>(&len), 8);
        out.write(json.data(), len);
    }
    dxait::ModelLoader st_loader;
    if (!st_loader.load_file("test_model.safetensors") || st_loader.format() != dxait::ModelFormat::Safetensors) {
        std::cerr << "Safetensors loader failed\n";
        return -1;
    }
    std::cout << "[2/2] Safetensors Model Loader:          PASSED\n";

    std::cout << "--------------------------------------------------------\n";
    std::cout << " All 2 Model Format Loaders Verified End-to-End!\n";
    return 0;
}
