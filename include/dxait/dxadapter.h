#ifndef DXAIT_DXADAPTER_H
#define DXAIT_DXADAPTER_H

#include "dxait_types.h"

#ifdef __cplusplus
extern "C" {
#endif

DXResult dxadapter_get_count(uint32_t *out_count);
DXResult dxadapter_get_caps(uint32_t index, DXAdapterCaps *out_caps);
DXResult dxadapter_create_device(uint32_t index, DXDevice **out_device);
void dxdevice_destroy(DXDevice *device);

#ifdef __cplusplus
}
#endif

#endif // DXAIT_DXADAPTER_H
