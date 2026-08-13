#include "dxait/dxait.hpp"
#include "dxait/dxmath.hpp"
#include <cstdio>

int main() {
    auto adapters = dxait::Adapter::enumerate();
    if (adapters.empty()) { printf("no gpu\n"); return 0; }
    auto device = dxait::Adapter::create_device(0);
    try {
        dxait::MathOps math(device.get());
        auto q = device->create_queue(dxait::QueueType::Compute);
        auto fence = device->create_fence(0);

        constexpr uint32_t num_rows = 2, row_dim = 64;
        constexpr uint64_t buf_bytes = num_rows * row_dim * sizeof(float);
        constexpr uint64_t weight_bytes = row_dim * sizeof(float);

        auto in = device->create_buffer(buf_bytes, dxait::MemLocation::Upload);
        auto weight = device->create_buffer(weight_bytes, dxait::MemLocation::Upload);
        auto out = device->create_buffer(buf_bytes, dxait::MemLocation::Readback);
        float* p = (float*)in->map();
        for (uint32_t i = 0; i < num_rows * row_dim; ++i) p[i] = 1.0f;
        in->unmap();

        printf("dispatch 1\n"); fflush(stdout);
        math.rms_norm(q.get(), out.get(), in.get(), weight.get(), num_rows, row_dim);
        q->signal(*fence, 1);
        fence->wait(1);
        printf("dispatch 1 done removed=0x%08X\n", (unsigned)device->get()->GetDeviceRemovedReason()); fflush(stdout);

        printf("dispatch 2 (same math)\n"); fflush(stdout);
        math.rms_norm(q.get(), out.get(), in.get(), weight.get(), num_rows, row_dim);
        q->signal(*fence, 2);
        fence->wait(2);
        printf("dispatch 2 done removed=0x%08X\n", (unsigned)device->get()->GetDeviceRemovedReason()); fflush(stdout);

        printf("softmax\n"); fflush(stdout);
        math.softmax(q.get(), out.get(), in.get(), num_rows, row_dim, 1.0f);
        q->signal(*fence, 3);
        fence->wait(3);
        printf("softmax done removed=0x%08X\n", (unsigned)device->get()->GetDeviceRemovedReason()); fflush(stdout);
    } catch (const std::exception& e) {
        printf("EXC: %s\n", e.what()); fflush(stdout);
    }
    printf("END\n"); fflush(stdout);
    return 0;
}
