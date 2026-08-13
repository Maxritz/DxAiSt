#ifndef DXAIT_DXSPECULATIVE_HPP
#define DXAIT_DXSPECULATIVE_HPP

#include "dxait.hpp"
#include "dxjit.hpp"
#include <vector>

namespace dxait {

class SpeculativeEngine {
public:
    explicit SpeculativeEngine(Device* device);
    ~SpeculativeEngine() = default;

    void verify_draft_tokens(
        Queue* queue,
        Buffer* accept_mask_buf,
        Buffer* target_probs_buf,
        Buffer* draft_probs_buf,
        Buffer* draft_tokens_buf,
        uint32_t num_draft_tokens,
        uint32_t vocab_size,
        float random_val
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

#endif // DXAIT_DXSPECULATIVE_HPP
