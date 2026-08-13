#include "dxait/dxcache.hpp"

namespace dxait {

AdvancedKVCache::AdvancedKVCache(Device* device, KVCacheType type, uint64_t max_bytes)
    : m_device(device), m_type(type) {
    m_cache_buffer = device->create_buffer(max_bytes, MemLocation::Default);
}

void AdvancedKVCache::apply_hadamard_transform(Queue* queue) {
    // Hadamard Walsh Matrix Transform on KV Cache
}

} // namespace dxait
