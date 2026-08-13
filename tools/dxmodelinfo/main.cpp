#include "dxait/dxait.hpp"
#include "dxait/dxmodel.hpp"
#include <iostream>
#include <filesystem>
#include <vector>

namespace fs = std::filesystem;

void inspect_path(const std::string& path_str) {
    if (!fs::exists(path_str)) {
        std::cout << "Directory " << path_str << " not found, skipping.\n";
        return;
    }

    std::cout << "Scanning directory: " << path_str << "\n";
    dxait::ModelLoader loader;

    uint32_t count = 0;
    for (const auto& entry : fs::recursive_directory_iterator(path_str, fs::directory_options::skip_permission_denied)) {
        if (!entry.is_regular_file()) continue;
        std::string ext = entry.path().extension().string();
        if (ext == ".gguf" || ext == ".safetensors" || ext == ".pte" || ext == ".onnx" || ext == ".bin") {
            std::string filepath = entry.path().string();
            if (loader.load_file(filepath)) {
                std::string fmt_name = "Unknown";
                if (loader.format() == dxait::ModelFormat::GGUF) fmt_name = "GGUF";
                else if (loader.format() == dxait::ModelFormat::Safetensors) fmt_name = "Safetensors";
                else if (loader.format() == dxait::ModelFormat::PTE) fmt_name = "PyTorch (.pte)";
                else if (loader.format() == dxait::ModelFormat::ONNX) fmt_name = "ONNX";
                else if (loader.format() == dxait::ModelFormat::PyTorchBin) fmt_name = "PyTorch (.bin)";

                std::cout << "  - " << entry.path().filename().string() << " [" << fmt_name << "]\n";
                count++;
                if (count >= 5) break; // Limit display to 5 files per dir
            }
        }
    }
    if (count == 0) {
        std::cout << "  (No model files matched in " << path_str << ")\n";
    }
}

int main() {
    std::cout << "========================================================\n";
    std::cout << " DXAiT Multi-Format Model Directory Inspector\n";
    std::cout << "========================================================\n\n";

    inspect_path("E:\\OLLAMA-Models\\GGUF");
    inspect_path("C:\\Users\\rr\\.cache\\huggingface\\hub");
    inspect_path("W:\\locomotion");

    return 0;
}
