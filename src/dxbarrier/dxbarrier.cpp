#include "dxait/dxbarrier.hpp"

namespace dxait {

void BarrierBatch::add_transition(
    ID3D12Resource* resource,
    D3D12_RESOURCE_STATES state_before,
    D3D12_RESOURCE_STATES state_after,
    UINT subresource
) {
    if (state_before == state_after) return;
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    b.Transition.pResource = resource;
    b.Transition.StateBefore = state_before;
    b.Transition.StateAfter = state_after;
    b.Transition.Subresource = subresource;
    m_barriers.push_back(b);
}

void BarrierBatch::add_uav(ID3D12Resource* resource) {
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    b.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    b.UAV.pResource = resource;
    m_barriers.push_back(b);
}

void BarrierBatch::flush(ID3D12GraphicsCommandList* cmd_list) {
    if (m_barriers.empty()) return;
    cmd_list->ResourceBarrier(static_cast<UINT>(m_barriers.size()), m_barriers.data());
    m_barriers.clear();
}

void BarrierBatch::clear() {
    m_barriers.clear();
}

} // namespace dxait
