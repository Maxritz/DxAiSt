// dxai_ingest: offline GGUF -> aligned weights.bin + TensorCoordinate index (dxstream).
#include "dxait/dxstream.hpp"
#include <windows.h>
#include <filesystem>
#include <iostream>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: dxai_ingest <model.gguf> <out_dir> [alignment]\n";
        return 2;
    }
    const std::string gguf = argv[1];
    const std::string out_dir = argv[2];
    uint32_t alignment = argc > 3 ? static_cast<uint32_t>(std::stoul(argv[3])) : 65536;

    if (!std::filesystem::exists(gguf)) {
        std::cerr << "input not found: " << gguf << "\n";
        return 2;
    }
    std::filesystem::create_directories(out_dir);

    LARGE_INTEGER sz{};
    HANDLE h = CreateFileA(gguf.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
    if (h != INVALID_HANDLE_VALUE) {
        GetFileSizeEx(h, &sz);
        CloseHandle(h);
    }

    dxait::IngestResult res;
    if (!dxait::ingest_gguf(gguf, out_dir, alignment, res)) {
        std::cerr << "ingest failed for " << gguf << "\n";
        return 1;
    }

    dxait::TensorIndex index;
    index.build(res.coords, res.payload_size, alignment);
    std::string idx_path = (std::filesystem::path(out_dir) / "index.bin").string();
    if (!index.save(idx_path)) {
        std::cerr << "index save failed\n";
        return 1;
    }

    std::cout << "[DXAiT INGEST] source=" << gguf
              << " source_bytes=" << sz.QuadPart
              << " payload=" << res.payload_path
              << " payload_bytes=" << res.payload_size
              << " tensors=" << res.coords.size()
              << " alignment=" << alignment
              << " index=" << idx_path << "\n";
    return 0;
}