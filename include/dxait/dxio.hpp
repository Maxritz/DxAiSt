#ifndef DXAIT_DXIO_HPP
#define DXAIT_DXIO_HPP

#include "dxait.hpp"
#include <dstorage.h>
#include <string>
#include <unordered_map>
#include <deque>
#include <cstdint>

namespace dxait {

enum class BypassPath {
    Unsupported, // volume/FS cannot BypassIO (storage driver incompatible / filter blocks)
    Unaligned,   // volume supports BypassIO but request offset/size breaks alignment
    Enabled,     // BypassIO engaged for this request
    DisabledByConfig
};

struct FetchRecord {
    std::string expert;
    uint64_t offset{0};
    uint64_t size{0};
    BypassPath path{BypassPath::Unaligned};
    HRESULT status{S_OK};
};

class DirectStorageContext {
public:
    explicit DirectStorageContext(Device* device, bool enable_bypassio = true);
    ~DirectStorageContext() = default;

    void set_debug_flags(DSTORAGE_DEBUG dv);
    DSTORAGE_DEBUG debug_flags() const { return m_debug_flags; }

    // Caches the IDStorageFile; first open also runs the FS_BPIO_OP_QUERY probe.
    ComPtr<IDStorageFile> open_file(const std::wstring& file_path);

    // Enqueues a raw byte read from file into a VRAM buffer via GPU virtual
    // address destination. destination_buffer must be Default/ReBAR heap.
    // Logs the per-fetch line on completion (wait_for_request).
    uint64_t enqueue_read(
        const std::wstring& file_path,
        uint64_t file_offset,
        uint64_t size_bytes,
        ID3D12Resource* destination_resource,
        uint64_t dest_offset = 0,
        const std::string& expert = ""
    );

    void submit();
    // Enqueue a DS fence signal that fires when all previously enqueued requests
    // complete, then submit. Returns the fence value; compute queue waits on it.
    void enqueue_signal_and_submit(ID3D12Fence* fence, uint64_t value);
    // Blocks until request (or all enqueued so far) complete; logs per-fetch lines.
    void wait_for_request(uint64_t request_id);
    void wait_all();

    // Capability probe for the volume behind the given file.
    struct BypassCapability {
        bool queried{false};
        bool compatible_storage{false};
        bool filter_blocked{false};
        std::wstring failing_driver;
        std::wstring failure_reason;
    };
    const BypassCapability& bypass_capability(const std::wstring& file_path);

    Device* device() const { return m_device; }

    // A/B toggle: when false, DStorageSetConfiguration1 DisableBypassIO=TRUE.
    bool bypassio_enabled() const { return m_enable_bypassio; }

    uint64_t requests_submitted() const { return m_total_submitted; }

private:
    Device* m_device;
    bool m_enable_bypassio{true};
    DSTORAGE_DEBUG m_debug_flags{DSTORAGE_DEBUG_NONE};
    ComPtr<IDStorageFactory> m_factory;
    ComPtr<IDStorageQueue> m_queue;
    static constexpr uint32_t k_status_capacity = 128;
    ComPtr<IDStorageStatusArray> m_status_array;

    std::unordered_map<std::wstring, ComPtr<IDStorageFile>> m_files;
    std::unordered_map<std::wstring, BypassCapability> m_capabilities;

    struct Pending {
        std::wstring file_path;
        std::string expert;
        uint64_t offset{0};
        uint64_t size{0};
        uint32_t status_slot{0};
        uint64_t request_id{0};
        BypassPath path{BypassPath::Unaligned};
    };
    std::deque<Pending> m_pending;
    uint64_t m_total_submitted{0};
    uint64_t m_statuses_issued{0};
};

// Direct OS-level BypassIO capability probe (FSCTL_MANAGE_BYPASS_IO QUERY).
// Returns capability for the file's backing volume. Independent of DirectStorage.
DirectStorageContext::BypassCapability probe_file_bypassio(const std::wstring& file_path);

} // namespace dxait

#endif // DXAIT_DXIO_HPP