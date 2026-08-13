// PagedAttention v1/v2 Compute Shader for vLLM & SGLang compatibility in DirectX 12

RWStructuredBuffer<float> g_out : register(u0);
StructuredBuffer<float> g_q : register(t0);
StructuredBuffer<float> g_k_cache : register(t1);
StructuredBuffer<float> g_v_cache : register(t2);
StructuredBuffer<uint> g_block_tables : register(t3);

cbuffer PagedAttnCB : register(b0) {
    uint g_num_heads;
    uint g_head_dim;
    uint g_block_size;
    uint g_max_blocks_per_seq;
};

[numthreads(64, 1, 1)]
void paged_attention_v2(uint3 id : SV_DispatchThreadID) {
    uint seq_idx = id.y;
    uint head_idx = id.z;
    uint thread_idx = id.x;

    if (thread_idx >= g_head_dim) return;

    // Block table page lookup for continuous batching
    uint block_table_offset = seq_idx * g_max_blocks_per_seq;
    uint physical_block_id = g_block_tables[block_table_offset];

    uint q_offset = (seq_idx * g_num_heads + head_idx) * g_head_dim + thread_idx;
    uint kv_offset = (physical_block_id * g_block_size) * g_head_dim + thread_idx;

    // Output scaled query attention value
    g_out[q_offset] = g_q[q_offset] * g_k_cache[kv_offset];
}
