// Elementwise compute operations for DXAiT

RWStructuredBuffer<float> g_out : register(u0);
StructuredBuffer<float> g_in1 : register(t0);
StructuredBuffer<float> g_in2 : register(t1);

cbuffer ElementwiseCB : register(b0) {
    uint g_count;
    float g_alpha;
    float g_beta;
    uint g_pad;
};

[numthreads(64, 1, 1)]
void vec_add(uint3 id : SV_DispatchThreadID) {
    if (id.x < g_count) {
        g_out[id.x] = g_alpha * g_in1[id.x] + g_beta * g_in2[id.x];
    }
}
