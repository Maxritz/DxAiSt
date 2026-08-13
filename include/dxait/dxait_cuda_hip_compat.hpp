#ifndef DXAIT_CUDA_HIP_COMPAT_HPP
#define DXAIT_CUDA_HIP_COMPAT_HPP

#include "dxait.hpp"
#include <cstdint>

// Drop-in CUDA / HIP / ROCm Compatibility Layer for DXAiT
namespace dxait {

typedef Device* dxaitDevice_t;
typedef Queue* dxaitStream_t;
typedef Buffer* dxaitPtr_t;
typedef Fence* dxaitEvent_t;

inline DXResult dxaitMalloc(Device* device, dxaitPtr_t* ptr, uint64_t size) {
    if (!device || !ptr) return DX_ERROR_INVALID_ARGUMENT;
    auto buf = device->create_buffer(size, MemLocation::Default);
    *ptr = buf.release();
    return DX_SUCCESS;
}

inline DXResult dxaitFree(dxaitPtr_t ptr) {
    delete ptr;
    return DX_SUCCESS;
}

inline DXResult dxaitStreamSynchronize(Queue* queue, Fence* fence, uint64_t val) {
    if (!queue || !fence) return DX_ERROR_INVALID_ARGUMENT;
    queue->signal(*fence, val);
    fence->wait(val);
    return DX_SUCCESS;
}

} // namespace dxait

#endif // DXAIT_CUDA_HIP_COMPAT_HPP
