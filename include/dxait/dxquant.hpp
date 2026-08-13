#ifndef DXAIT_DXQUANT_HPP
#define DXAIT_DXQUANT_HPP

#include "dxait.hpp"
#include "dxjit.hpp"
#include <cstdint>

namespace dxait {

// Q8_0 Block representation: 32 int8 weights + 1 fp16 scale (34 bytes total)
struct alignas(2) BlockQ8_0 {
    uint16_t d; // fp16 scale factor
    int8_t qs[32]; // quantized 8-bit weights
};

// Q4_0 Block representation: 32 4-bit weights packed into 16 bytes + 1 fp16 scale (18 bytes total)
struct alignas(2) BlockQ4_0 {
    uint16_t d; // fp16 scale factor
    uint8_t qs[16]; // packed nibbles
};

void quantize_row_q8_0(const float* src, BlockQ8_0* dst, uint64_t k);
void dequantize_row_q8_0(const BlockQ8_0* src, float* dst, uint64_t k);

void quantize_row_q4_0(const float* src, BlockQ4_0* dst, uint64_t k);
void dequantize_row_q4_0(const BlockQ4_0* src, float* dst, uint64_t k);

class QuantOps {
public:
    explicit QuantOps(Device* device);
    ~QuantOps() = default;

    void dequantize_q4_0_gpu(
        Queue* queue,
        Buffer* out_buf,
        Buffer* in_q4_buf,
        uint32_t num_blocks
    );

private:
    Device* m_device;
    PipelineCache m_pso_cache;
    ComPtr<ID3D12RootSignature> m_root_sig;
    ComPtr<ID3D12CommandAllocator> m_cmd_alloc;
    ComPtr<ID3D12GraphicsCommandList> m_cmd_list;

    void init_root_signature();
};

} // namespace dxait

#endif // DXAIT_DXQUANT_HPP
