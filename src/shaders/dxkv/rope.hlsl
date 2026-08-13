// Rotary Positional Embedding (RoPE) Compute Shader

RWStructuredBuffer<float> g_vec : register(u0);

cbuffer RoPECB : register(b0) {
    uint g_seq_len;
    uint g_head_dim;
    uint g_num_heads;
    float g_theta_base;
};

[numthreads(32, 1, 1)]
void rope_kernel(uint3 id : SV_DispatchThreadID) {
    uint head_idx = id.y;
    uint pos = id.x;
    uint pair_idx = id.z;

    if (pos >= g_seq_len || pair_idx >= (g_head_dim / 2)) return;

    uint base_idx = (pos * g_num_heads + head_idx) * g_head_dim + pair_idx * 2;
    float freq = 1.0f / pow(g_theta_base, (float)(pair_idx * 2) / (float)g_head_dim);
    float angle = (float)pos * freq;

    float cos_a = cos(angle);
    float sin_a = sin(angle);

    float x0 = g_vec[base_idx + 0];
    float x1 = g_vec[base_idx + 1];

    g_vec[base_idx + 0] = x0 * cos_a - x1 * sin_a;
    g_vec[base_idx + 1] = x0 * sin_a + x1 * cos_a;
}
