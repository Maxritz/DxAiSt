#include "dxait/dxio.hpp"
#include <windows.h>
#include <winioctl.h>
#include <iostream>
#include <stdexcept>
#include <iomanip>
#include <mutex>

namespace dxait {

namespace {

std::string wide_to_utf8(const std::wstring& w) {
    if (w.empty()) return "";
    int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), -1, nullptr, 0, nullptr, nullptr);
    std::string s(n - 1, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.c_str(), -1, s.data(), n, nullptr, nullptr);
    return s;
}

const char* path_name(BypassPath p) {
    switch (p) {
        case BypassPath::Unsupported:   return "UNSUPPORTED";
        case BypassPath::Unaligned:     return "STAGED";
        case BypassPath::Enabled:       return "BYPASSIO";
        case BypassPath::DisabledByConfig: return "STAGED";
    }
    return "?";
}

// DStorageSetConfiguration1 may only be called once per process, before any
// factory exists. All DirectStorageContexts share the process config; the first
// construction wins. The BypassIO toggle is therefore process-scoped (set via
// the constructor of the first context / DXAIT_DSTORAGE_DISABLE_BYPASSIO env).
std::once_flag g_config_once;
bool g_config_bypassio_on = true;
void ensure_process_config(bool enable_bypassio) {
    std::call_once(g_config_once, [enable_bypassio] {
        g_config_bypassio_on = enable_bypassio;
        DSTORAGE_CONFIGURATION1 cfg{};
        cfg.DisableBypassIO = !enable_bypassio;
        cfg.ForceMappingLayer = false;
        cfg.DisableTelemetry = false;
        cfg.DisableGpuDecompressionMetacommand = false;
        cfg.DisableGpuDecompression = false;
        cfg.NumSubmitThreads = 4;
        cfg.NumBuiltInCpuDecompressionThreads = 4;
        HRESULT hr = DStorageSetConfiguration1(&cfg);
        if (FAILED(hr)) {
            std::cerr << "[DX12 DSTORAGE] DStorageSetConfiguration1 failed: "
                      << std::hex << hr << std::dec << "\n";
        }
    });
}

} // namespace

static void log_fetch(const FetchRecord& r, const char* stage) {
    std::cout << "[DX12 DSTORAGE] expert=" << r.expert
              << " offset=" << r.offset << " size=" << r.size
              << " path=" << path_name(r.path)
              << " status=" << std::hex << std::showbase << (uint32_t)r.status << std::dec
              << " (" << stage << ")\n";
}

DirectStorageContext::BypassCapability probe_file_bypassio(const std::wstring& file_path) {
    DirectStorageContext::BypassCapability cap{};
    HANDLE h = CreateFileW(file_path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE,
                           nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) return cap;
    cap.queried = true;

    FS_BPIO_INPUT in{};
    in.Operation = FS_BPIO_OP_QUERY;
    in.InFlags = FSBPIO_INFL_None;
    FS_BPIO_OUTPUT out{};
    DWORD returned = 0;
    BOOL ok = DeviceIoControl(h, FSCTL_MANAGE_BYPASS_IO, &in, sizeof(in), &out, sizeof(out), &returned, nullptr);
    if (ok && returned >= sizeof(out)) {
        cap.compatible_storage = (out.OutFlags & FSBPIO_OUTFL_COMPATIBLE_STORAGE_DRIVER) != 0;
        cap.filter_blocked = (out.OutFlags & FSBPIO_OUTFL_FILTER_ATTACH_BLOCKED) != 0;
        if (out.Query.FailingDriverNameLen > 0)
            cap.failing_driver.assign(out.Query.FailingDriverName, out.Query.FailingDriverNameLen);
        if (out.Query.FailureReasonLen > 0)
            cap.failure_reason.assign(out.Query.FailureReason, out.Query.FailureReasonLen);
    } else {
        // Older FT/SKIPPED storage stack: treat as unsupported for logging purposes.
        cap.compatible_storage = false;
    }
    CloseHandle(h);
    return cap;
}

DirectStorageContext::DirectStorageContext(Device* device, bool enable_bypassio)
    : m_device(device), m_enable_bypassio(enable_bypassio) {

    // Process-global, one-shot: the first constructor's choice wins.
    ensure_process_config(enable_bypassio);
    m_enable_bypassio = g_config_bypassio_on; // effective value for path logging

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

    if (FAILED(m_factory->CreateStatusArray(k_status_capacity, nullptr, IID_PPV_ARGS(&m_status_array)))) {
        throw std::runtime_error("CreateStatusArray failed");
    }
}

void DirectStorageContext::set_debug_flags(DSTORAGE_DEBUG dv) {
    m_debug_flags = dv;
    m_factory->SetDebugFlags(dv);
}

const DirectStorageContext::BypassCapability& DirectStorageContext::bypass_capability(const std::wstring& file_path) {
    auto it = m_capabilities.find(file_path);
    if (it == m_capabilities.end()) {
        BypassCapability cap = probe_file_bypassio(file_path);
        cap.queried = true;
        m_capabilities[file_path] = cap;
        return m_capabilities[file_path];
    }
    return it->second;
}

ComPtr<IDStorageFile> DirectStorageContext::open_file(const std::wstring& file_path) {
    auto it = m_files.find(file_path);
    if (it != m_files.end()) return it->second;

    ComPtr<IDStorageFile> file;
    if (FAILED(m_factory->OpenFile(file_path.c_str(), IID_PPV_ARGS(&file)))) {
        throw std::runtime_error("IDStorageFactory::OpenFile failed: " + wide_to_utf8(file_path));
    }

    // Capability probe on first open.
    if (m_capabilities.find(file_path) == m_capabilities.end()) {
        BypassCapability cap = probe_file_bypassio(file_path);
        cap.queried = true;
        m_capabilities[file_path] = cap;
        std::cout << "[DX12 DSTORAGE] bypassio_probe file=" << wide_to_utf8(file_path)
                  << " compatible_storage=" << (cap.compatible_storage ? "YES" : "NO")
                  << " filter_blocked=" << (cap.filter_blocked ? "YES" : "NO")
                  << " failing_driver=\"" << wide_to_utf8(cap.failing_driver)
                  << "\" reason=\"" << wide_to_utf8(cap.failure_reason) << "\"\n";
    }

    m_files[file_path] = file;
    return file;
}

uint64_t DirectStorageContext::enqueue_read(
    const std::wstring& file_path,
    uint64_t file_offset,
    uint64_t size_bytes,
    ID3D12Resource* destination_resource,
    uint64_t dest_offset,
    const std::string& expert
) {
    if (size_bytes > UINT32_MAX) {
        throw std::runtime_error("enqueue_read: size exceeds UINT32_MAX; split into chunks");
    }

    ComPtr<IDStorageFile> file = open_file(file_path);

    DSTORAGE_REQUEST request{};
    request.Options.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
    request.Options.DestinationType = DSTORAGE_REQUEST_DESTINATION_BUFFER;
    request.Source.File.Source = file.Get();
    request.Source.File.Offset = file_offset;
    request.Source.File.Size = static_cast<UINT32>(size_bytes);
    request.Destination.Buffer.Resource = destination_resource;
    request.Destination.Buffer.Offset = dest_offset;
    request.Destination.Buffer.Size = static_cast<UINT32>(size_bytes);
    request.UncompressedSize = 0;

    m_queue->EnqueueRequest(&request);

    uint32_t slot = static_cast<uint32_t>(m_total_submitted % k_status_capacity);
    Pending p;
    p.file_path = file_path;
    p.expert = expert;
    p.offset = file_offset;
    p.size = size_bytes;
    p.status_slot = slot;
    p.request_id = m_total_submitted;

    auto capit = m_capabilities.find(file_path);
    bool vol_supported = false;
    if (capit != m_capabilities.end()) {
        const auto& cap = capit->second;
        vol_supported = cap.compatible_storage && !cap.filter_blocked;
    }
    constexpr uint64_t k_align = 4096; // BypassIO requires 4K-aligned offset+size
    bool aligned = (file_offset % k_align == 0) && (size_bytes % k_align == 0);
    if (!m_enable_bypassio) {
        p.path = BypassPath::DisabledByConfig;
    } else if (!vol_supported) {
        p.path = BypassPath::Unsupported;
    } else if (!aligned) {
        p.path = BypassPath::Unaligned;
    } else {
        p.path = BypassPath::Enabled;
    }

    m_pending.push_back(std::move(p));
    return m_total_submitted++;
}

void DirectStorageContext::submit() {
    m_queue->Submit();
}

void DirectStorageContext::enqueue_signal_and_submit(ID3D12Fence* fence, uint64_t value) {
    // Issue status tracking for every pending request not yet tracked, so
    // wait_all() can report per-fetch completion even on the signal path.
    while (m_statuses_issued < m_total_submitted) {
        for (auto& p : m_pending) {
            if (p.request_id == m_statuses_issued) {
                m_queue->EnqueueStatus(m_status_array.Get(), p.status_slot);
                break;
            }
        }
        ++m_statuses_issued;
    }
    m_queue->EnqueueSignal(fence, value);
    m_queue->Submit();
}

namespace {
constexpr DWORD k_wait_timeout_ms = 5000;
}

void DirectStorageContext::wait_for_request(uint64_t request_id) {
    auto deadline = GetTickCount64() + k_wait_timeout_ms;
    for (auto it = m_pending.begin(); it != m_pending.end();) {
        if (it->request_id == request_id) {
            while (!m_status_array->IsComplete(it->status_slot)) {
                if (GetTickCount64() > deadline) {
                    FetchRecord rec{it->expert, it->offset, it->size, it->path,
                                    (HRESULT)0x8000FFFFL /* E_UNEXPECTED */};
                    log_fetch(rec, "TIMEOUT — DS rejected request (e.g. unaligned offset); no completion");
                    m_pending.erase(it);
                    return;
                }
                Sleep(1); // yield to DS IO threads; short slice avoids busy-spin
            }
            HRESULT hr = m_status_array->GetHResult(it->status_slot);
            FetchRecord rec{it->expert, it->offset, it->size, it->path, hr};
            log_fetch(rec, "done");
            m_pending.erase(it);
            return;
        }
        ++it;
    }
}

void DirectStorageContext::wait_all() {
    auto deadline = GetTickCount64() + k_wait_timeout_ms;
    for (auto& p : m_pending) {
        while (!m_status_array->IsComplete(p.status_slot)) {
            if (GetTickCount64() > deadline) {
                for (auto& rest : m_pending) {
                    FetchRecord rec{rest.expert, rest.offset, rest.size, rest.path,
                                    (HRESULT)0x8000FFFFL /* E_UNEXPECTED */};
                    log_fetch(rec, "TIMEOUT — DS rejected request; no completion");
                }
                m_pending.clear();
                return;
            }
            Sleep(1);
        }
        HRESULT hr = m_status_array->GetHResult(p.status_slot);
        FetchRecord rec{p.expert, p.offset, p.size, p.path, hr};
        log_fetch(rec, "done");
    }
    m_pending.clear();
}

} // namespace dxait