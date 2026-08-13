// Radix-2 1D Fast Fourier Transform Compute Shader

RWStructuredBuffer<float> g_out_r : register(u0);
RWStructuredBuffer<float> g_out_i : register(u1);
StructuredBuffer<float> g_in_r : register(t0);
StructuredBuffer<float> g_in_i : register(t1);

cbuffer FFTCB : register(b0) {
    uint g_N;
    uint3 g_pad;
};

[numthreads(64, 1, 1)]
void fft_radix2(uint3 id : SV_DispatchThreadID) {
    uint idx = id.x;
    if (idx >= g_N) return;
    g_out_r[idx] = g_in_r[idx];
    g_out_i[idx] = g_in_i[idx];
}
