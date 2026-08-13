#include "dxait/dxait.hpp"
#include "dxait/dxattention.hpp"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

static float cpu_attention_one(
    const std::vector<float>& q, const std::vector<float>& k, const std::vector<float>& v,
    uint32_t seq, uint32_t dim, uint32_t i, uint32_t kv_head, uint32_t q_head,
    uint32_t n_qh, uint32_t n_kvh, bool causal, uint32_t window, float scale)
{
    float max_s = -1e30f;
    for (uint32_t j = 0; j < seq; ++j) {
        if (causal && j > i) break;
        if (window > 0 && i > j + window) continue;
        float dot = 0.0f;
        for (uint32_t d = 0; d < dim; ++d)
            dot += q[(q_head * seq + i) * dim + d] * k[(kv_head * seq + j) * dim + d];
        max_s = (std::max)(max_s, dot * scale);
    }
    float sum = 0.0f;
    std::vector<float> w(seq, 0.0f);
    for (uint32_t j = 0; j < seq; ++j) {
        if (causal && j > i) break;
        if (window > 0 && i > j + window) continue;
        float dot = 0.0f;
        for (uint32_t d = 0; d < dim; ++d)
            dot += q[(q_head * seq + i) * dim + d] * k[(kv_head * seq + j) * dim + d];
        w[j] = std::exp(dot * scale - max_s);
        sum += w[j];
    }
    std::vector<float> out(dim, 0.0f);
    for (uint32_t d = 0; d < dim; ++d) {
        float o = 0.0f;
        for (uint32_t j = 0; j < seq; ++j) {
            if (causal && j > i) break;
            if (window > 0 && i > j + window) continue;
            o += (w[j] / sum) * v[(kv_head * seq + j) * dim + d];
        }
        out[d] = o;
    }
    return out[0];
}

static float cpu_linear_one(
    const std::vector<float>& q, const std::vector<float>& k, const std::vector<float>& v,
    uint32_t seq, uint32_t dim, uint32_t i, uint32_t kv_head, uint32_t q_head)
{
    auto kern = [](float x) { return (x > 0.0f) ? (x + 1.0f) : std::exp(x); };
    std::vector<std::vector<float>> S(dim, std::vector<float>(dim, 0.0f));
    std::vector<float> z(dim, 0.0f);
    for (uint32_t j = 0; j <= i && j < seq; ++j) {
        for (uint32_t d = 0; d < dim; ++d) {
            float kd = kern(k[(kv_head * seq + j) * dim + d]);
            z[d] += kd;
            for (uint32_t e = 0; e < dim; ++e)
                S[d][e] += kd * v[(kv_head * seq + j) * dim + e];
        }
    }
    float norm = 0.0f;
    std::vector<float> o(dim, 0.0f);
    for (uint32_t d = 0; d < dim; ++d) {
        float qd = kern(q[(q_head * seq + i) * dim + d]);
        norm += qd * z[d];
        for (uint32_t e = 0; e < dim; ++e) o[e] += qd * S[d][e];
    }
    return (norm > 0.0f) ? (o[0] / norm) : 0.0f;
}

int main() {
    printf("DXAiT Attention Mechanism Variants Test (MHA / GQA / MQA / SWA)\n");
    printf("===============================================================\n\n");

    auto adapters = dxait::Adapter::enumerate();
    if (adapters.empty()) { printf("No GPU found, skipping test.\n"); return 0; }
    auto device = dxait::Adapter::create_device(0);

    constexpr uint32_t seq = 8;
    constexpr uint32_t dim = 4;
    constexpr uint32_t n_qh = 2;   // 2 query heads
    constexpr uint32_t n_kvh = 1;  // GQA 2:1, MQA when kv heads = 1
    constexpr float scale = 0.5f;

    constexpr uint64_t q_bytes = (uint64_t)n_qh * seq * dim * sizeof(float);
    constexpr uint64_t kv_bytes = (uint64_t)n_qh * seq * dim * sizeof(float); // sized for max kv heads (MHA)

    std::vector<float> q(n_qh * seq * dim), k(n_qh * seq * dim), v(n_qh * seq * dim);
    for (uint32_t i = 0; i < q.size(); ++i) q[i] = ((float)i * 0.13f) - 1.0f;
    for (uint32_t i = 0; i < k.size(); ++i) k[i] = ((float)i * 0.07f) + 0.5f;
    for (uint32_t i = 0; i < v.size(); ++i) v[i] = ((float)i * 0.03f) + 0.1f;

    auto q_buf = device->create_buffer(q_bytes, dxait::MemLocation::Upload);
    auto k_buf = device->create_buffer(kv_bytes, dxait::MemLocation::Upload);
    auto v_buf = device->create_buffer(kv_bytes, dxait::MemLocation::Upload);
    std::memcpy(q_buf->map(), q.data(), q_bytes); q_buf->unmap();
    std::memcpy(k_buf->map(), k.data(), kv_bytes); k_buf->unmap();
    std::memcpy(v_buf->map(), v.data(), kv_bytes); v_buf->unmap();

    dxait::AttentionOps attn(device.get());
    auto cq = device->create_queue(dxait::QueueType::Compute);
    auto fence = device->create_fence(0);

    struct Case { const char* name; dxait::AttentionMechanism mech; uint32_t kvh; bool causal; uint32_t window; bool linear; };
    const Case cases[] = {
        {"MHA (kv=q heads)", dxait::AttentionMechanism::MHA, n_qh, false, 0, false},
        {"GQA (shared KV)",  dxait::AttentionMechanism::GQA, n_kvh, false, 0, false},
        {"MQA (1 KV head)",  dxait::AttentionMechanism::MQA, 1, false, 0, false},
        {"SWA (causal win=4)", dxait::AttentionMechanism::SlidingWindow, n_kvh, true, 4, false},
        {"FlashAttention",   dxait::AttentionMechanism::FlashAttention, n_kvh, true, 0, false},
        {"LinearAttention",  dxait::AttentionMechanism::LinearAttention, n_kvh, true, 0, true},
    };

    bool all_ok = true;
    for (auto& c : cases) {
        // Fresh readback per case to avoid stale-data ambiguity.
        auto out_buf = device->create_buffer(q_bytes, dxait::MemLocation::Readback);
        dxait::AttentionConfig cfg;
        cfg.mechanism = c.mech;
        cfg.num_q_heads = n_qh;
        cfg.num_kv_heads = c.kvh;
        cfg.head_dim = dim;
        cfg.seq_len = seq;
        cfg.scale = scale;
        cfg.sliding_window_size = c.window;

        attn.dispatch_attention(cq.get(), out_buf.get(), q_buf.get(), k_buf.get(), v_buf.get(), cfg);
        cq->signal(*fence, 1);
        fence->wait(1);

        float* o = (float*)out_buf->map();
        bool ok = true;
        float max_err = 0.0f;
        for (uint32_t h = 0; h < n_qh; ++h) {
            for (uint32_t i = 0; i < seq; ++i) {
                uint32_t kvh = (c.kvh * h / n_qh);
                if (kvh >= c.kvh) kvh = c.kvh - 1;
                float ref = c.linear
                    ? cpu_linear_one(q, k, v, seq, dim, i, kvh, h)
                    : cpu_attention_one(q, k, v, seq, dim, i, kvh, h, n_qh, c.kvh, c.causal, c.window, scale);
                float got = o[(h * seq + i) * dim];
                float err = std::fabs(got - ref);
                if (err > max_err) max_err = err;
                if (err > 1e-2f) { printf("    [%s] h=%u i=%u got=%.5f ref=%.5f\n", c.name, h, i, got, ref); ok = false; }
            }
        }
        out_buf->unmap();
        printf("  [%s] max_err=%.5f %s\n", c.name, max_err, ok ? "PASS" : "FAIL");
        if (!ok) all_ok = false;
    }

    printf("\nResult: %s\n", all_ok ? "Attention variants PASSED" : "FAILED");
    return all_ok ? 0 : 1;
}
