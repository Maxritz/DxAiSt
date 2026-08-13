#include "dxait/dxtrace.hpp"
#include <wrl/client.h>
#include <d3d12.h>

namespace dxait {

// D3D12 native event markers via ID3D12GraphicsCommandList::BeginEvent /
// EndEvent. Works with PIX and GPU timing tools, no external runtime needed.
// No-op when debug layer is off.

void trace_begin_event(ID3D12GraphicsCommandList* cmd_list, const char* name) {
    if (!cmd_list || !name) return;
    wchar_t wname[256] = {};
    MultiByteToWideChar(CP_UTF8, 0, name, -1, wname, 256);
    cmd_list->BeginEvent(0, wname, (UINT)(wcslen(wname) * sizeof(wchar_t)));
}

void trace_end_event(ID3D12GraphicsCommandList* cmd_list) {
    if (!cmd_list) return;
    cmd_list->EndEvent();
}

ProfileScope::ProfileScope(ID3D12GraphicsCommandList* cmd_list, const char* name)
    : m_cmd_list(cmd_list) {
    trace_begin_event(m_cmd_list, name);
}

ProfileScope::~ProfileScope() {
    trace_end_event(m_cmd_list);
}

} // namespace dxait
