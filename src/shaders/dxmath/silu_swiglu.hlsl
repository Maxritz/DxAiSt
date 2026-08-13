// SiLU and SwiGLU Activation Compute Shaders

RWStructuredBuffer<float> g_out : register(u0);
StructuredBuffer<float> g_gate : register(t0);
StructuredBuffer<float> g_up : register(t1);

cbuffer ActivationCB : register(b0) {
    uint g_count;
    uint3 g_pad;
};

float silu(float x) {
    return x / (1.0f + exp(-x));
}

[numthreads(64, 1, 1)]
void silu_kernel(uint3 id : SV_DispatchThreadID) {
    if (id.x < g_count) {
        g_out[id.x] = silu(g_gate[id.x]);
    }
}

[numthreads(64, 1, 1)]
void swiglu_kernel(uint3 id : SV_DispatchThreadID) {
    if (id.x < g_count) {
        g_out[id.x] = silu(g_gate[id.x]) * g_up[id.x];
    }
}
