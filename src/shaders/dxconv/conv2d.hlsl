// 2D Convolution Compute Shader for DXAiT

RWStructuredBuffer<float> g_out : register(u0);
StructuredBuffer<float> g_in : register(t0);
StructuredBuffer<float> g_kernel : register(t1);

cbuffer Conv2DCB : register(b0) {
    uint g_width;
    uint g_height;
    uint g_ksize;
    uint g_pad;
};

[numthreads(16, 16, 1)]
void conv2d_kernel(uint3 id : SV_DispatchThreadID) {
    uint x = id.x;
    uint y = id.y;

    if (x >= g_width || y >= g_height) return;

    int half_k = (int)(g_ksize / 2);
    float sum = 0.0f;

    for (int ky = -half_k; ky <= half_k; ++ky) {
        for (int kx = -half_k; kx <= half_k; ++kx) {
            int ix = clamp((int)x + kx, 0, (int)g_width - 1);
            int iy = clamp((int)y + ky, 0, (int)g_height - 1);
            
            float in_val = g_in[iy * g_width + ix];
            float k_val = g_kernel[(ky + half_k) * g_ksize + (kx + half_k)];
            sum += in_val * k_val;
        }
    }
    g_out[y * g_width + x] = sum;
}
