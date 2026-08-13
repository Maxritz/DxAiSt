#ifndef DXAIT_DXRAND_HPP
#define DXAIT_DXRAND_HPP

#include "dxait.hpp"
#include "dxjit.hpp"

namespace dxait {

class RandomGenerator {
public:
    explicit RandomGenerator(Device* device);
    ~RandomGenerator() = default;

    void fill_uniform(
        Queue* queue,
        Buffer* out_buf,
        uint32_t count,
        uint64_t seed = 1337
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

#endif // DXAIT_DXRAND_HPP
