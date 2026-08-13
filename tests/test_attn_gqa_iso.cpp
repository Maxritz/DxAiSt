#include "dxait/dxait.hpp"
#include "dxait/dxattention.hpp"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

int main() {
    auto adapters = dxait::Adapter::enumerate();
    if (adapters.empty()) { printf("no gpu\n"); return 0; }
    auto device = dxait::Adapter::create_device(0);

    constexpr uint32_t seq = 8, dim = 4, n_qh = 2, n_kvh = 1;
    constexpr float scale = 0.5f;
    constexpr uint64_t qb = (uint64_t)n_qh * seq * dim * sizeof(float);
    constexpr uint64_t kvb = (uint64_t)n_qh * seq * dim * sizeof(float);

    std::vector<float> q(n_qh * seq * dim), k(n_qh * seq * dim), v(n_qh * seq * dim);
    for (uint32_t i = 0; i < q.size(); ++i) q[i] = ((float)i * 0.13f) - 1.0f;
    for (uint32_t i = 0; i < k.size(); ++i) k[i] = ((float)i * 0.07f) + 0.5f;
    for (uint32_t i = 0; i < v.size(); ++i) v[i] = ((float)i * 0.03f) + 0.1f;

    auto qb0 = device->create_buffer(qb, dxait::MemLocation::Upload);
    auto kb0 = device->create_buffer(kvb, dxait::MemLocation::Upload);
    auto vb0 = device->create_buffer(kvb, dxait::MemLocation::Upload);
    auto ob0 = device->create_buffer(qb, dxait::MemLocation::Readback);
    std::memcpy(qb0->map(), q.data(), qb); qb0->unmap();
    std::memcpy(kb0->map(), k.data(), kvb); kb0->unmap();
    std::memcpy(vb0->map(), v.data(), kvb); vb0->unmap();

    dxait::AttentionOps attn(device.get());
    auto cq = device->create_queue(dxait::QueueType::Compute);
    auto fence = device->create_fence(0);

    dxait::AttentionConfig cfg;
    cfg.mechanism = dxait::AttentionMechanism::GQA;
    cfg.num_q_heads = n_qh;
    cfg.num_kv_heads = n_kvh;
    cfg.head_dim = dim;
    cfg.seq_len = seq;
    cfg.scale = scale;

    printf("GQA alone, fresh process, first dispatch\n"); fflush(stdout);
    attn.dispatch_attention(cq.get(), ob0.get(), qb0.get(), kb0.get(), vb0.get(), cfg);
    cq->signal(*fence, 1);
    fence->wait(1);
    HRESULT r = device->get()->GetDeviceRemovedReason();
    printf("removed=0x%08X\n", (unsigned)r); fflush(stdout);

    float* o = (float*)ob0->map();
    printf("out[0..3] = %.4f %.4f %.4f %.4f\n", o[0], o[1], o[2], o[3]);
    ob0->unmap();
    return 0;
}
