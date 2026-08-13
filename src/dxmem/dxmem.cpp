#include "dxait/dxait.hpp"

namespace dxait {

Buffer::Buffer(ID3D12Device* device, uint64_t size_bytes, MemLocation loc)
    : m_size(size_bytes), m_location(loc) {
    
    D3D12_HEAP_PROPERTIES heap_props{};
    if (loc == MemLocation::ReBAR) {
        heap_props.Type = D3D12_HEAP_TYPE_CUSTOM;
        heap_props.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE;
        heap_props.MemoryPoolPreference = D3D12_MEMORY_POOL_L1; // VRAM
    } else {
        heap_props.Type = static_cast<D3D12_HEAP_TYPE>(loc);
        heap_props.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
        heap_props.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    }
    heap_props.CreationNodeMask = 1;
    heap_props.VisibleNodeMask = 1;

    D3D12_RESOURCE_DESC res_desc{};
    res_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    res_desc.Alignment = 0;
    res_desc.Width = size_bytes;
    res_desc.Height = 1;
    res_desc.DepthOrArraySize = 1;
    res_desc.MipLevels = 1;
    res_desc.Format = DXGI_FORMAT_UNKNOWN;
    res_desc.SampleDesc.Count = 1;
    res_desc.SampleDesc.Quality = 0;
    res_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    res_desc.Flags = D3D12_RESOURCE_FLAG_NONE;

    D3D12_RESOURCE_STATES initial_state = D3D12_RESOURCE_STATE_COMMON;
    if (loc == MemLocation::Upload || loc == MemLocation::ReBAR) initial_state = D3D12_RESOURCE_STATE_GENERIC_READ;
    else if (loc == MemLocation::Readback) initial_state = D3D12_RESOURCE_STATE_COPY_DEST;

    if (FAILED(device->CreateCommittedResource(
            &heap_props,
            D3D12_HEAP_FLAG_NONE,
            &res_desc,
            initial_state,
            nullptr,
            IID_PPV_ARGS(&m_resource)))) {
        throw std::runtime_error("CreateCommittedResource failed");
    }
}

void* Buffer::map() {
    void* ptr = nullptr;
    D3D12_RANGE read_range{0, 0};
    if (FAILED(m_resource->Map(0, &read_range, &ptr))) {
        return nullptr;
    }
    return ptr;
}

void Buffer::unmap() {
    D3D12_RANGE write_range{0, m_size};
    m_resource->Unmap(0, &write_range);
}

bool Buffer::make_resident(ID3D12Device* device) {
    ID3D12Pageable* pageable = m_resource.Get();
    return SUCCEEDED(device->MakeResident(1, &pageable));
}

bool Buffer::evict(ID3D12Device* device) {
    ID3D12Pageable* pageable = m_resource.Get();
    return SUCCEEDED(device->Evict(1, &pageable));
}

} // namespace dxait
