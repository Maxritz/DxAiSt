#include "dxait/dxait.hpp"
#include "dxait/dxtrace.hpp"
#include <cstdio>

int main() {
    printf("DXAiT Trace Marker Smoke Test\n");
    printf("==============================\n\n");

    auto adapters = dxait::Adapter::enumerate();
    if (adapters.empty()) { printf("No GPU found, skipping test.\n"); return 0; }
    auto device = dxait::Adapter::create_device(0);

    auto q = device->create_queue(dxait::QueueType::Compute);
    auto fence = device->create_fence(0);
    dxait::ComPtr<ID3D12CommandAllocator> alloc;
    dxait::ComPtr<ID3D12GraphicsCommandList> list;
    device->get()->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&alloc));
    device->get()->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, alloc.Get(), nullptr, IID_PPV_ARGS(&list));

    printf("Recording trace markers...\n");
    {
        dxait::ProfileScope scope(list.Get(), "bench_scope");
        dxait::trace_begin_event(list.Get(), "begin_marker");
        dxait::trace_end_event(list.Get());
    }

    list->Close();
    ID3D12CommandList* ls[] = { list.Get() };
    q->execute(ls, 1);
    q->signal(*fence, 1);
    fence->wait(1);

    printf("Recorded and executed command list with markers.\n\n");
    printf("Result: Trace marker smoke test PASSED\n");
    return 0;
}
