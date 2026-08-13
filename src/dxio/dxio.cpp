#include "dxait/dxio.hpp"
#include <stdexcept>

namespace dxait {

DirectStorageContext::DirectStorageContext(Device* device) : m_device(device) {
    if (FAILED(DStorageGetFactory(IID_PPV_ARGS(&m_factory)))) {
        throw std::runtime_error("DStorageGetFactory failed");
    }

    DSTORAGE_QUEUE_DESC queue_desc{};
    queue_desc.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
    queue_desc.Capacity = DSTORAGE_MAX_QUEUE_CAPACITY;
    queue_desc.Priority = DSTORAGE_PRIORITY_NORMAL;
    queue_desc.Device = device->get();

    if (FAILED(m_factory->CreateQueue(&queue_desc, IID_PPV_ARGS(&m_queue)))) {
        throw std::runtime_error("IDStorageFactory::CreateQueue failed");
    }

    if (FAILED(m_factory->CreateStatusArray(1, nullptr, IID_PPV_ARGS(&m_status_array)))) {
        throw std::runtime_error("CreateStatusArray failed");
    }
}

void DirectStorageContext::enqueue_read(
    const std::wstring& file_path,
    uint64_t file_offset,
    uint64_t size_bytes,
    Buffer* destination_buffer,
    uint64_t dest_offset
) {
    ComPtr<IDStorageFile> file;
    if (FAILED(m_factory->OpenFile(file_path.c_str(), IID_PPV_ARGS(&file)))) {
        throw std::runtime_error("IDStorageFactory::OpenFile failed");
    }

    DSTORAGE_REQUEST request{};
    request.Options.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
    request.Options.DestinationType = DSTORAGE_REQUEST_DESTINATION_MULTIPLE_SUBRESOURCES;
    request.Source.File.Source = file.Get();
    request.Source.File.Offset = file_offset;
    request.Source.File.Size = static_cast<UINT32>(size_bytes);
    request.Destination.MultipleSubresources.Resource = destination_buffer->get();
    request.Destination.MultipleSubresources.FirstSubresource = 0;
    request.UncompressedSize = static_cast<UINT32>(size_bytes);

    m_queue->EnqueueRequest(&request);
    m_request_count++;
}

void DirectStorageContext::submit() {
    if (m_request_count > 0) {
        m_queue->EnqueueStatus(m_status_array.Get(), 0);
        m_queue->Submit();
    }
}

void DirectStorageContext::wait() {
    if (m_request_count > 0) {
        while (!m_status_array->IsComplete(0)) {
            Sleep(1);
        }
        m_request_count = 0;
    }
}

} // namespace dxait
