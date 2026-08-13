// ATOM Fine-Grained INT4 Weight-Only GEMM Compute Shader

RWStructuredBuffer<float> g_out : register(u0);
ByteAddressBuffer g_weight_int4 : register(t0);
StructuredBuffer<float> g_scales : register(t1);
StructuredBuffer<float> g_vector : register(t2);

cbuffer ATOMParams : register(b0) {
    uint g_M;
    uint g_K;
    uint g_group_size;
    uint g_pad;
};

[numthreads(64, 1, 1)]
void atom_gemm_int4_kernel(uint3 id : SV_DispatchThreadID) {
    uint row = id.x;
    if (row >= g_M) return;

    float acc = 0.0f;
    uint num_groups = g_K / g_group_size;

    for (uint g = 0; g < num_groups; ++g) {
        float scale = g_scales[row * num_groups + g];
        uint base_k = g * g_group_size;

        for (uint i = 0; i < g_group_size; i += 2) {
            uint byte_idx = (row * g_K + base_k + i) / 2;
            uint val = g_weight_int4.Load(byte_idx) & 0xFF;

            int w0 = (int)(val & 0x0F) - 8;
            int w1 = (int)(val >> 4) - 8;

            acc += ((float)w0 * scale * g_vector[base_k + i]);
            acc += ((float)w1 * scale * g_vector[base_k + i + 1]);
        }
    }
    g_out[row] = acc;
}
