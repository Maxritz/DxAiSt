#ifndef DXAIT_DXCOLLECTIVE_HPP
#define DXAIT_DXCOLLECTIVE_HPP

#include "dxait.hpp"
#include <vector>

namespace dxait {

class CollectiveOps {
public:
    explicit CollectiveOps(const std::vector<Device*>& devices);
    ~CollectiveOps() = default;

    void all_reduce_sum(const std::vector<Buffer*>& buffers, uint64_t size_bytes);

private:
    std::vector<Device*> m_devices;
};

} // namespace dxait

#endif // DXAIT_DXCOLLECTIVE_HPP
