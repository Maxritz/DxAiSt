#ifndef DXAIT_DXIO_HPP
#define DXAIT_DXIO_HPP

#include "dxait.hpp"
#include <dstorage.h>
#include <string>
#include <memory>

namespace dxait {

class DirectStorageContext {
public:
    explicit DirectStorageContext(Device* device);
    ~DirectStorageContext() = default;

    void enqueue_read(
        const std::wstring& file_path,
        uint64_t file_offset,
        uint64_t size_bytes,
        Buffer* destination_buffer,
        uint64_t dest_offset = 0
    );

    void submit();
    void wait();

private:
    Device* m_device;
    ComPtr<IDStorageFactory> m_factory;
    ComPtr<IDStorageQueue> m_queue;
    ComPtr<IDStorageStatusArray> m_status_array;
    uint32_t m_request_count{0};
};

} // namespace dxait

#endif // DXAIT_DXIO_HPP
