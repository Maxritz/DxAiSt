#ifndef DXAIT_TYPES_H
#define DXAIT_TYPES_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum DXResult {
    DX_SUCCESS = 0,
    DX_ERROR_INVALID_ARGUMENT = -1,
    DX_ERROR_OUT_OF_MEMORY = -2,
    DX_ERROR_DEVICE_REMOVED = -3,
    DX_ERROR_UNSUPPORTED = -4,
    DX_ERROR_UNKNOWN = -99
} DXResult;

typedef enum DXQueueType {
    DX_QUEUE_DIRECT = 0,
    DX_QUEUE_COMPUTE = 1,
    DX_QUEUE_COPY = 2
} DXQueueType;

typedef enum DXMemLocation {
    DX_MEM_DEFAULT = 0, // VRAM
    DX_MEM_UPLOAD = 1,  // System RAM -> GPU write
    DX_MEM_READBACK = 2 // GPU -> System RAM read
} DXMemLocation;

typedef struct DXRuntime DXRuntime;
typedef struct DXDevice DXDevice;
typedef struct DXQueue DXQueue;
typedef struct DXBuffer DXBuffer;
typedef struct DXFence DXFence;

typedef struct DXAdapterCaps {
    char name[256];
    uint32_t vendor_id;
    uint32_t device_id;
    uint64_t dedicated_video_memory;
    uint64_t shared_system_memory;
    uint32_t feature_level;
    uint32_t shader_model;
    uint32_t wave_min;
    uint32_t wave_max;
    bool wave_ops;
    bool fp16;
    bool int8;
    bool execute_indirect;
} DXAdapterCaps;

#ifdef __cplusplus
}
#endif

#endif // DXAIT_TYPES_H
