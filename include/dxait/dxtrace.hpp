#ifndef DXAIT_DXTRACE_HPP
#define DXAIT_DXTRACE_HPP

#include "dxait.hpp"
#include <string>

namespace dxait {

class ProfileScope {
public:
    ProfileScope(ID3D12GraphicsCommandList* cmd_list, const char* name);
    ~ProfileScope();

private:
    ID3D12GraphicsCommandList* m_cmd_list;
};

void trace_begin_event(ID3D12GraphicsCommandList* cmd_list, const char* name);
void trace_end_event(ID3D12GraphicsCommandList* cmd_list);

} // namespace dxait

#endif // DXAIT_DXTRACE_HPP
