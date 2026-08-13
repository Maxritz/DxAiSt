// TensorRT-LLM Speculative Decoding Draft Verification Compute Shader

RWStructuredBuffer<uint> g_accept_mask : register(u0);
StructuredBuffer<float> g_target_probs : register(t0);
StructuredBuffer<float> g_draft_probs : register(t1);
StructuredBuffer<uint> g_draft_tokens : register(t2);

cbuffer SpeculativeCB : register(b0) {
    uint g_num_draft_tokens;
    uint g_vocab_size;
    float g_random_val;
    uint g_pad;
};

[numthreads(32, 1, 1)]
void speculative_verify_kernel(uint3 id : SV_DispatchThreadID) {
    uint token_idx = id.x;
    if (token_idx >= g_num_draft_tokens) return;

    uint draft_token = g_draft_tokens[token_idx];
    uint prob_offset = token_idx * g_vocab_size + draft_token;

    float p_target = g_target_probs[prob_offset];
    float p_draft = g_draft_probs[prob_offset];

    float accept_ratio = p_draft > 0.0f ? min(1.0f, p_target / p_draft) : 0.0f;

    // Accept token if random value <= acceptance probability
    if (g_random_val <= accept_ratio) {
        g_accept_mask[token_idx] = 1;
    } else {
        g_accept_mask[token_idx] = 0;
    }
}
