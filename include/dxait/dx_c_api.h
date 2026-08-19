#ifndef DXAIT_DX_C_API_H
#define DXAIT_DX_C_API_H

#include <windows.h>
#include <stdint.h>

#if defined(DXAIT_DLL_BUILD)
#define DX_CAPI __declspec(dllexport)
#elif defined(DXAIT_DLL_USE)
#define DX_CAPI __declspec(dllimport)
#else
#define DX_CAPI
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct dx_device_s dx_device;
typedef struct dx_queue_s dx_queue;
typedef struct dx_buffer_s dx_buffer;

/* Device lifecycle */
DX_CAPI HRESULT dx_create_device(uint32_t index, dx_device** out);
DX_CAPI void dx_destroy_device(dx_device* dev);

/* Queue: device owns one compute queue; handle stays valid until device destroy */
DX_CAPI HRESULT dx_device_queue(dx_device* dev, dx_queue** out);
DX_CAPI void dx_destroy_queue(dx_queue* q);

/* Buffers. loc: 1=Default 2=Upload 3=Readback */
DX_CAPI HRESULT dx_create_buffer(dx_device* dev, uint64_t bytes, int loc, dx_buffer** out);
DX_CAPI void* dx_buffer_map(dx_buffer* b);
DX_CAPI void dx_buffer_unmap(dx_buffer* b);
DX_CAPI void dx_destroy_buffer(dx_buffer* b);

/* Transfers (Default buffers go through staging + GPU copy) */
DX_CAPI HRESULT dx_upload(dx_device* dev, dx_buffer* dst, uint64_t off, const void* src, uint64_t bytes);
DX_CAPI HRESULT dx_download(dx_device* dev, dx_buffer* src, uint64_t off, void* dst, uint64_t bytes);

/* LA op layer */
DX_CAPI HRESULT dx_la_elementwise(dx_device* dev, dx_queue* q, dx_buffer* out, dx_buffer* a, dx_buffer* b, uint32_t count, int op, float alpha, float beta);
DX_CAPI HRESULT dx_la_activation(dx_device* dev, dx_queue* q, dx_buffer* out, dx_buffer* in, uint32_t count, int act, float alpha);
DX_CAPI HRESULT dx_la_rmsnorm(dx_device* dev, dx_queue* q, dx_buffer* out, dx_buffer* in, dx_buffer* gamma, uint32_t rows, uint32_t dim, float eps);
DX_CAPI HRESULT dx_la_softmax(dx_device* dev, dx_queue* q, dx_buffer* out, dx_buffer* in, uint32_t rows, uint32_t dim);
DX_CAPI HRESULT dx_la_reduce(dx_device* dev, dx_queue* q, dx_buffer* out, dx_buffer* in, uint32_t rows, uint32_t dim, int op);
DX_CAPI HRESULT dx_la_gemm_f16_dot2(dx_device* dev, dx_queue* q, dx_buffer* out, dx_buffer* a, dx_buffer* b, uint32_t M, uint32_t N, uint32_t K);
DX_CAPI HRESULT dx_la_gemm_f16_wmma(dx_device* dev, dx_queue* q, dx_buffer* out, dx_buffer* a, dx_buffer* b, uint32_t M, uint32_t N, uint32_t K);

/* Synchronization and diagnostics */
DX_CAPI HRESULT dx_wait(dx_device* dev);
DX_CAPI HRESULT dx_device_desc(dx_device* dev, char* buf, int cap);
DX_CAPI const char* dx_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* DXAIT_DX_C_API_H */
