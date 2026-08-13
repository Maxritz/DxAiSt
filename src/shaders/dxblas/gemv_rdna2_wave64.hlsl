// RDNA2 (gfx1031) Wave64 Dot4 Quantized Q8_0 GEMV Compute Kernel
// Optimized for RX 6700 XT 384 GB/s memory bandwidth & Wave64 execution

RWStructuredBuffer<float> g_out : register(u0);
ByteAddressBuffer g_weight_q8 : register(t0);
StructuredBuffer<float> g_vector : register(t1);

cbuffer GEMVParams : register(b0) {
    uint g_M;
    uint g_K;
    uint2 g_pad;
};

// Target Wave64 execution on RDNA2 hardware
[numthreads(64, 1, 1)]
void gemv_q8_0_rdna2_wave64(uint3 id : SV_DispatchThreadID, uint lane_id : SV_GroupIndex) {
    uint row = id.x;
    if (row >= g_M) return;

    uint num_blocks = g_K / 32;
    uint row_byte_offset = row * num_blocks * 34; // 34 bytes per Q8_0 block (2B scale + 32B int8)

    float sum = 0.0f;
    for (uint b = 0; b < num_blocks; ++b) {
        uint block_offset = row_byte_offset + b * 34;
        uint d_raw = g_weight_q8.Load(block_offset) & 0xFFFF;
        
        // RDNA2 native sdot4 / sudot4 4-wide int8 dot product simulation
        [unroll]
        for (uint i = 0; i < 32; i += 4) {
            uint w_packed = g_weight_q8.Load(block_offset + 2 + i);
            float v0 = g_vector[b * 32 + i + 0];
            float v1 = g_vector[b * 32 + i + 1];
            float v2 = g_vector[b * 32 + i + 2];
            float v3 = g_vector[b * 32 + i + 3];

            int w0 = (int)(w_packed & 0xFF) - 128;
            int w1 = (int)((w_packed >> 8) & 0xFF) - 128;
            int w2 = (int)((w_packed >> 16) & 0xFF) - 128;
            int w3 = (int)((w_packed >> 24) & 0xFF) - 128;

            sum += (w0 * v0 + w1 * v1 + w2 * v2 + w3 * v3);
        }
    }
    g_out[row] = sum;
}
