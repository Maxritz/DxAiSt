#include "dxait/dxait.hpp"
#include "dxait/dxcache.hpp"
#include <cstdio>
#include <cmath>

int main() {
    printf("DXAiT AdvancedKVCache Hadamard Transform Test\n");
    printf("==============================================\n\n");

    auto adapters = dxait::Adapter::enumerate();
    if (adapters.empty()) { printf("No GPU found, skipping test.\n"); return 0; }
    auto device = dxait::Adapter::create_device(0);

    constexpr uint32_t n = 4;
    constexpr uint64_t bytes = n * sizeof(float);

    dxait::AdvancedKVCache cache(device.get(), dxait::KVCacheType::HadamardTransform, bytes);
    if (cache.type() != dxait::KVCacheType::HadamardTransform) {
        printf("FAILED: cache type mismatch\n");
        return 1;
    }

    auto upload = device->create_buffer(bytes, dxait::MemLocation::Upload);
    auto readback = device->create_buffer(bytes, dxait::MemLocation::Readback);
    float* p = (float*)upload->map();
    p[0] = 1.0f; p[1] = 2.0f; p[2] = 3.0f; p[3] = 4.0f;
    upload->unmap();

    // Single compute queue + shared fence for seed copy, transform, readback copy.
    // Keeps all ops on one queue so GPU ordering is guaranteed (no cross-queue race).
    auto q = device->create_queue(dxait::QueueType::Compute);
    auto fence = device->create_fence(0);
    uint64_t fv = 0;

    dxait::ComPtr<ID3D12CommandAllocator> alloc;
    dxait::ComPtr<ID3D12GraphicsCommandList> list;
    device->get()->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&alloc));
    device->get()->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, alloc.Get(), nullptr, IID_PPV_ARGS(&list));

    // 1. upload -> cache buffer
    list->CopyBufferRegion(cache.get_buffer()->get(), 0, upload->get(), 0, bytes);
    list->Close();
    ID3D12CommandList* ls1[] = { list.Get() };
    q->execute(ls1, 1);
    q->signal(*fence, ++fv);
    fence->wait(fv);

    // 1b. verify seed landed: cache -> readback
    alloc->Reset();
    list->Reset(alloc.Get(), nullptr);
    list->CopyBufferRegion(readback->get(), 0, cache.get_buffer()->get(), 0, bytes);
    list->Close();
    ID3D12CommandList* ls0[] = { list.Get() };
    q->execute(ls0, 1);
    q->signal(*fence, ++fv);
    fence->wait(fv);
    printf("Applying hadamard transform to [1,2,3,4]...\n");
    cache.apply_hadamard_transform(q.get());

    // 2. cache buffer -> readback
    alloc->Reset();
    list->Reset(alloc.Get(), nullptr);
    list->CopyBufferRegion(readback->get(), 0, cache.get_buffer()->get(), 0, bytes);
    list->Close();
    ID3D12CommandList* ls2[] = { list.Get() };
    q->execute(ls2, 1);
    q->signal(*fence, ++fv);
    fence->wait(fv);

    const float expected[4] = {10.0f, -2.0f, -4.0f, 0.0f};
    float* r = (float*)readback->map();
    bool ok = true;
    for (int i = 0; i < 4; ++i) {
        printf("  out[%d] = %.4f (expected %.4f)\n", i, r[i], expected[i]);
        if (std::fabs(r[i] - expected[i]) > 1e-3f) ok = false;
    }
    readback->unmap();

    printf("\nResult: %s\n", ok ? "Hadamard transform PASSED" : "FAILED");
    return ok ? 0 : 1;
}
