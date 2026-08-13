#ifndef DXAIT_DXMATH_HPP
#define DXAIT_DXMATH_HPP

#include "dxait.hpp"
#include "dxjit.hpp"

namespace dxait {

struct SamplingParams {
    float temperature{1.0f};
    float top_p{1.0f};
    uint32_t top_k{0};
    float min_p{0.0f};
    float repetition_penalty{1.0f};
};

class MathOps {
public:
    explicit MathOps(Device* device);
    ~MathOps() = default;

    void rms_norm(
        Queue* queue,
        Buffer* out_buf,
        Buffer* in_buf,
        Buffer* weight_buf,
        uint32_t num_rows,
        uint32_t row_dim,
        float eps = 1e-5f
    );

    void softmax(
        Queue* queue,
        Buffer* out_buf,
        Buffer* in_buf,
        uint32_t num_rows,
        uint32_t row_dim,
        float temperature = 1.0f
    );

    void rope(
        Queue* queue,
        Buffer* out_buf,
        Buffer* in_buf,
        uint32_t num_rows,
        uint32_t head_dim,
        uint32_t pos,
        float theta = 10000.0f
    );

    uint32_t sample(
        Queue* queue,
        Buffer* logits_buf,
        uint32_t vocab_size,
        const SamplingParams& params
    );

private:
    Device* m_device;
    PipelineCache m_pso_cache;
    ComPtr<ID3D12RootSignature> m_root_sig;
    ComPtr<ID3D12CommandAllocator> m_cmd_alloc;
    ComPtr<ID3D12GraphicsCommandList> m_cmd_list;
    std::unique_ptr<Fence> m_fence;
    uint64_t m_fence_val{0};

    void init_root_signature();
    void sync_wait();
};

} // namespace dxait

#endif // DXAIT_DXMATH_HPP
