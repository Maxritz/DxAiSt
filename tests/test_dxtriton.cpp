#include "dxait/dxait.hpp"
#include "dxait/dxtriton.hpp"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

int main() {
    std::cout << "========================================================\n";
    std::cout << " DXAiT AITER / Triton-JIT HLSL Compiler Engine Test\n";
    std::cout << "========================================================\n\n";

    auto adapters = dxait::Adapter::enumerate();
    if (adapters.empty()) {
        std::cout << "No GPU found, skipping test.\n";
        return 0;
    }

    auto device = dxait::Adapter::create_device(0);
    const auto& caps = device->caps();

    std::cout << "Target GPU:              " << caps.name << "\n";
    std::cout << "Preferred Wave Size:     Wave" << caps.preferred_wave_size << "\n\n";

    dxait::TritonJIT triton(device.get());

    // 1. Generate Triton MatMul HLSL Shader Source
    dxait::TritonKernelSpec spec{};
    spec.kernel_name = "triton_matmul_block64";
    spec.block_m = 64;
    spec.block_n = 64;
    spec.block_k = 32;
    spec.wave_size = caps.preferred_wave_size;
    spec.use_wmma = caps.wmma_supported;

    std::cout << "1. Generating AITER/Triton HLSL Compute Shader Source...\n";
    std::string hlsl_code = triton.generate_triton_matmul_hlsl(spec);
    assert(!hlsl_code.empty());
    std::cout << "   Triton HLSL Source Code Generated Successfully!\n\n";

    // 2. JIT Compile HLSL Shader into D3D12 Pipeline State Object (PSO)
    std::cout << "2. JIT Compiling Triton HLSL Source into D3D12 Pipeline State Object...\n";
    auto pso = triton.compile_triton_kernel(hlsl_code, "triton_matmul_kernel", nullptr);
    assert(pso != nullptr);
    std::cout << "   Triton PSO JIT Compiled Successfully!\n\n";

    // 3. Dispatch JIT Compiled Triton Kernel
    constexpr uint32_t M = 64, N = 64, K = 64;
    constexpr uint64_t mat_bytes = M * K * sizeof(float);

    auto buf_a = device->create_buffer(mat_bytes, dxait::MemLocation::Upload);
    auto buf_b = device->create_buffer(mat_bytes, dxait::MemLocation::Upload);
    auto buf_out = device->create_buffer(mat_bytes, dxait::MemLocation::Default);
    auto buf_readback = device->create_buffer(mat_bytes, dxait::MemLocation::Readback);

    std::vector<float> h_a(M * K, 1.0f);
    std::vector<float> h_b(K * N, 2.0f);

    std::memcpy(buf_a->map(), h_a.data(), mat_bytes);
    buf_a->unmap();

    std::memcpy(buf_b->map(), h_b.data(), mat_bytes);
    buf_b->unmap();

    auto queue = device->create_queue(dxait::QueueType::Compute);

    std::cout << "3. Dispatching JIT Compiled Triton MatMul Kernel...\n";
    triton.dispatch_triton_kernel(queue.get(), pso.Get(), nullptr, buf_out.get(), buf_a.get(), buf_b.get(), M, N, K);

    // Verify Readback Output
    auto copy_queue = device->create_queue(dxait::QueueType::Copy);
    auto fence = device->create_fence(0);

    dxait::ComPtr<ID3D12CommandAllocator> alloc;
    dxait::ComPtr<ID3D12GraphicsCommandList> list;
    device->get()->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY, IID_PPV_ARGS(&alloc));
    device->get()->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, alloc.Get(), nullptr, IID_PPV_ARGS(&list));
    list->CopyBufferRegion(buf_readback->get(), 0, buf_out->get(), 0, mat_bytes);
    list->Close();

    ID3D12CommandList* lists[] = { list.Get() };
    copy_queue->execute(lists, 1);
    copy_queue->signal(*fence, 1);
    fence->wait(1);

    float* out_ptr = static_cast<float*>(buf_readback->map());
    assert(out_ptr != nullptr);
    std::cout << "   Triton Kernel Output Result [0..3]: "
              << out_ptr[0] << ", " << out_ptr[1] << ", " << out_ptr[2] << ", " << out_ptr[3] << "\n";
    buf_readback->unmap();

    std::cout << "\n========================================================\n";
    std::cout << " AITER / Triton-JIT HLSL Compiler Engine Test PASSED!\n";

    return 0;
}
