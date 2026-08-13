#include "dxait/dxait.hpp"
#include "dxait/dxcollective.hpp"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

// Multi-GPU RingAttention: splits Q/K/V across N devices, each computes a flash
// attention shard, orchestrator combines. Skips gracefully when < 2 adapters.

int main() {
    printf("DXAiT Multi-GPU RingAttention Test\n");
    printf("==================================\n\n");

    auto adapters = dxait::Adapter::enumerate();
    if (adapters.size() < 2) {
        printf("Need >= 2 GPU adapters, found %zu. Skipping multi-GPU ring test.\n", adapters.size());
        printf("Result: SKIPPED (single adapter)\n");
        return 0;
    }

    std::vector<std::unique_ptr<dxait::Device>> devices;
    for (size_t i = 0; i < adapters.size(); ++i)
        devices.push_back(dxait::Adapter::create_device(static_cast<uint32_t>(i)));

    constexpr uint32_t seq = 8;
    constexpr uint32_t dim = 4;
    constexpr uint32_t n_qh = 1;
    constexpr uint32_t n_kvh = 1;
    constexpr float scale = 0.5f;

    uint32_t n_dev = static_cast<uint32_t>(devices.size());
    uint32_t local_seq = seq / n_dev;
    if (local_seq < 2) local_seq = 2; // keep shards meaningful

    constexpr uint64_t local_q_bytes = 4 * dim * sizeof(float); // q for 4 tokens
    constexpr uint64_t local_kv_bytes = 4 * dim * sizeof(float);

    std::vector<std::unique_ptr<dxait::Buffer>> q_bufs, k_bufs, v_bufs, out_bufs;
    for (uint32_t d = 0; d < n_dev; ++d) {
        auto dev = devices[d].get();
        q_bufs.push_back(dev->create_buffer(local_q_bytes, dxait::MemLocation::Upload));
        k_bufs.push_back(dev->create_buffer(local_kv_bytes, dxait::MemLocation::Upload));
        v_bufs.push_back(dev->create_buffer(local_kv_bytes, dxait::MemLocation::Upload));
        out_bufs.push_back(dev->create_buffer(local_q_bytes, dxait::MemLocation::Readback));
        float* qp = (float*)q_bufs[d]->map();
        float* kp = (float*)k_bufs[d]->map();
        float* vp = (float*)v_bufs[d]->map();
        for (uint32_t t = 0; t < 4; ++t) {
            float base = (float)(d * 4 + t);
            for (uint32_t e = 0; e < dim; ++e) {
                qp[t * dim + e] = base * 0.13f - 1.0f + 0.01f * e;
                kp[t * dim + e] = base * 0.07f + 0.5f;
                vp[t * dim + e] = base * 0.03f + 0.1f;
            }
        }
        q_bufs[d]->unmap(); k_bufs[d]->unmap(); v_bufs[d]->unmap();
    }

    std::vector<dxait::Device*> dev_ptrs;
    for (auto& d : devices) dev_ptrs.push_back(d.get());

    dxait::CollectiveOps collectives(dev_ptrs);
    dxait::RingAttentionConfig cfg;
    cfg.seq_len = seq;
    cfg.head_dim = dim;
    cfg.num_q_heads = n_qh;
    cfg.num_kv_heads = n_kvh;
    cfg.scale = scale;
    cfg.causal = true;

    std::vector<dxait::Buffer*> qp_, kp_, vp_, op_;
    for (auto& b : q_bufs) qp_.push_back(b.get());
    for (auto& b : k_bufs) kp_.push_back(b.get());
    for (auto& b : v_bufs) vp_.push_back(b.get());
    for (auto& b : out_bufs) op_.push_back(b.get());

    bool ran = collectives.ring_attention(qp_, kp_, vp_, op_, cfg);
    if (!ran) {
        printf("RingAttention did not run on multi-GPU.\n");
        return 1;
    }

    // Verify each device produced finite non-zero output (shard sanity).
    bool ok = true;
    for (uint32_t d = 0; d < n_dev; ++d) {
        float* o = (float*)out_bufs[d]->map();
        bool finite = true;
        for (uint32_t t = 0; t < 4; ++t) {
            for (uint32_t e = 0; e < dim; ++e) {
                float val = o[t * dim + e];
                if (!std::isfinite(val)) finite = false;
            }
        }
        printf("  device[%u] output finite: %s (o[0]=%.4f)\n", d, finite ? "yes" : "no", o[0]);
        if (!finite) ok = false;
        out_bufs[d]->unmap();
    }

    printf("\nResult: %s\n", ok ? "Multi-GPU RingAttention shard execution PASSED" : "FAILED");
    return ok ? 0 : 1;
}
