#ifndef DXAIT_DXBARRIER_HPP
#define DXAIT_DXBARRIER_HPP

#include "dxait.hpp"
#include <vector>

namespace dxait {

class BarrierBatch {
public:
    BarrierBatch() = default;
    ~BarrierBatch() = default;

    void add_transition(
        ID3D12Resource* resource,
        D3D12_RESOURCE_STATES state_before,
        D3D12_RESOURCE_STATES state_after,
        UINT subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES
    );

    void add_uav(ID3D12Resource* resource);

    void flush(ID3D12GraphicsCommandList* cmd_list);
    void clear();

private:
    std::vector<D3D12_RESOURCE_BARRIER> m_barriers;
};

} // namespace dxait

#endif // DXAIT_DXBARRIER_HPP
