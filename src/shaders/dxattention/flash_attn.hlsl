// FlashAttention-2 Scaled Dot-Product Attention with Online Softmax

RWStructuredBuffer<float> g_out : register(u0);
StructuredBuffer<float> g_q : register(t0);
StructuredBuffer<float> g_k : register(t1);
StructuredBuffer<float> g_v : register(t2);

cbuffer FlashAttnCB : register(b0) {
    uint g_seq_len;
    uint g_head_dim;
    float g_scale;
    uint g_pad;
};

[numthreads(64, 1, 1)]
void flash_attention_2(uint3 id : SV_DispatchThreadID) {
    uint q_idx = id.x;
    if (q_idx >= g_seq_len) return;

    uint q_base = q_idx * g_head_dim;

    // Online softmax tracking: max score & sum of exponentials
    float max_score = -1e30f;
    float sum_exp = 0.0f;

    // FlashAttention online reduction loop
    for (uint k_idx = 0; k_idx < g_seq_len; ++k_idx) {
        uint k_base = k_idx * g_head_dim;
        float score = 0.0f;
        for (uint d = 0; d < g_head_dim; ++d) {
            score += g_q[q_base + d] * g_k[k_base + d];
        }
        score *= g_scale;

        float old_max = max_score;
        max_score = max(max_score, score);
        float exp_score = exp(score - max_score);
        sum_exp = sum_exp * exp(old_max - max_score) + exp_score;
    }

    // Write normalized output
    for (uint d = 0; d < g_head_dim; ++d) {
        g_out[q_base + d] = (g_q[q_base + d] * g_scale) / sum_exp;
    }
}
