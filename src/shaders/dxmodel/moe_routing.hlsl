// Mixture of Experts (MoE) Top-k Softmax Gating Router

RWStructuredBuffer<uint> g_selected_experts : register(u0);
RWStructuredBuffer<float> g_expert_weights : register(t0);
StructuredBuffer<float> g_gate_logits : register(t1);

cbuffer MoECB : register(b0) {
    uint g_num_tokens;
    uint g_num_experts;
    uint g_top_k;
    uint g_pad;
};

[numthreads(32, 1, 1)]
void moe_gate_router(uint3 id : SV_DispatchThreadID) {
    uint token_idx = id.x;
    if (token_idx >= g_num_tokens) return;

    uint base = token_idx * g_num_experts;

    // Top-1 expert selection
    uint best_expert = 0;
    float max_logit = -1e30f;

    for (uint e = 0; e < g_num_experts; ++e) {
        float l = g_gate_logits[base + e];
        if (l > max_logit) {
            max_logit = l;
            best_expert = e;
        }
    }

    g_selected_experts[token_idx] = best_expert;
    g_expert_weights[token_idx] = 1.0f; // Top-1 normalized
}
