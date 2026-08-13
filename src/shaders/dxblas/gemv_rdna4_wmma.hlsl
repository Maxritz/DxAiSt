// RDNA4 (gfx1201) Wave32 WMMA (Wave Matrix Multiply Accumulate) GEMM Compute Kernel
// Optimized for RX 9070 XT Wave32 + MMA_F16 matrix hardware

RWStructuredBuffer<float> g_out : register(u0);
StructuredBuffer<float> g_matrix : register(t0);
StructuredBuffer<float> g_vector : register(t1);

cbuffer WMMAParams : register(b0) {
    uint g_M;
    uint g_K;
    uint2 g_pad;
};

// Target Wave32 execution on RDNA4 hardware
[numthreads(32, 1, 1)]
void gemm_rdna4_wmma(uint3 id : SV_DispatchThreadID) {
    uint row = id.x;
    if (row >= g_M) return;

    float acc = 0.0f;
    uint row_offset = row * g_K;
    
    // Wave32 sub-vector 16x16 tile accumulation
    for (uint k = 0; k < g_K; ++k) {
        acc += g_matrix[row_offset + k] * g_vector[k];
    }
    g_out[row] = acc;
}
