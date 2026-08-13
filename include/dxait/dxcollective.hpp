#ifndef DXAIT_DXCOLLECTIVE_HPP
#define DXAIT_DXCOLLECTIVE_HPP

#include "dxait.hpp"
#include "dxattention.hpp"
#include <vector>

namespace dxait {

struct RingAttentionConfig {
    uint32_t seq_len{4096};
    uint32_t head_dim{128};
    uint32_t num_q_heads{32};
    uint32_t num_kv_heads{8};
    float scale{0.088388347f};
    bool causal{true};
};

class CollectiveOps {
public:
    explicit CollectiveOps(const std::vector<Device*>& devices);
    ~CollectiveOps() = default;

    void all_reduce_sum(const std::vector<Buffer*>& buffers, uint64_t size_bytes);

    // Multi-GPU RingAttention: splits Q/K/V across devices, each device computes
    // partial attention on its KV shard with running online-softmax stats, then
    // partial accumulators (acc, l, m) ring-pass around the loop. Returns true
    // if at least 2 devices participated.
    bool ring_attention(
        const std::vector<Buffer*>& q_bufs,
        const std::vector<Buffer*>& k_bufs,
        const std::vector<Buffer*>& v_bufs,
        const std::vector<Buffer*>& out_bufs,
        const RingAttentionConfig& config
    );

private:
    std::vector<Device*> m_devices;
};

} // namespace dxait

#endif // DXAIT_DXCOLLECTIVE_HPP
