#ifndef DXAIT_DXIOCP_HPP
#define DXAIT_DXIOCP_HPP

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>

#include "dxait.hpp"
#include <wrl/client.h>
#include <string>
#include <vector>
#include <unordered_set>
#include <atomic>
#include <cstdint>
#include <thread>

namespace dxait {

// Chunk request sent by the client: op 0 = fetch index blob, op 1 = fetch [offset,size).
#pragma pack(push, 1)
struct ChunkRequest {
    uint32_t op;         // 0 = INDEX, 1 = CHUNK
    uint32_t padding;
    uint64_t offset;
    uint64_t size;
};
struct ChunkHeader {
    uint32_t op;
    uint32_t flags;      // bit0 = zstd compressed
    uint64_t size;       // payload bytes following this header
};
#pragma pack(pop)

struct IocpStats {
    uint64_t chunks_received{0};
    uint64_t bytes_received{0};
    uint64_t zstd_engaged{0};
    uint64_t latency_wsarecv_total_ns{0};
    uint64_t copy_async_issued{0};
    uint64_t copy_submit_ns{0};  // CPU time recording+executing the copy cmd list
    uint64_t copy_wait_ns{0};    // CPU time blocked waiting on the copy fence
    uint64_t copies_that_waited{0};
};

// Streaming chunk server: serves the payload file (index blob op0, byte ranges
// op1) over a listener with a worker thread per connection. Client side owns the
// IOCP path; server is the remote payload provider.
class IocpStreamServer {
public:
    IocpStreamServer(const std::string& payload_path, const std::vector<uint8_t>& index_blob,
                     bool zstd_enabled = false, uint16_t port = 9095);
    ~IocpStreamServer();
    bool start();
    void stop();
    bool is_running() const { return m_running.load(); }
    uint16_t port() const { return m_port; }

private:
    struct Conn {
        SOCKET sock{INVALID_SOCKET};
        std::thread thread;
    };
    std::string m_payload;
    std::vector<uint8_t> m_index;
    bool m_zstd{false};
    uint16_t m_port;
    SOCKET m_listen{INVALID_SOCKET};
    std::atomic<bool> m_running{false};
    std::vector<std::thread> m_accept_threads;
    std::vector<Conn> m_conns;

    void accept_loop();
    void serve_conn(Conn* c);
};

// IOCP chunk client: requests chunks, receives each into a pinned UPLOAD-heap
// staging buffer via an overlapped WSARecv completed through a completion port,
// optionally Zstd-decompresses (only when zstd.dll is loadable AND the chunk is
// flagged), then issues an async COPY-queue buffer copy from pinned RAM into the
// VRAM slot. Records per-chunk WSARecv completion latency and whether the DMA
// copy ever blocked the CPU.
class IocpStreamClient {
public:
    IocpStreamClient(Device* device, const std::string& server_ip, uint16_t port,
                     uint64_t slot_size = 128ull * 1024 * 1024);
    ~IocpStreamClient();
    IocpStreamClient(const IocpStreamClient&) = delete;
    IocpStreamClient& operator=(const IocpStreamClient&) = delete;

    bool connect(const std::string& ip, uint16_t port);

    // One-time COPY-queue warmup: absorb driver first-write-to-VRAM latency for
    // this device's COPY queue AND the destination ring, outside timed fetches.
    void warm_copy_queue();
    // Warm a specific VRAM destination buffer (write `mb` MB, wait, untimed).
    void warm_vram(ID3D12Resource* slot, uint32_t mb);

    // Fetch index blob (op0). Returns false on network error.
    bool fetch_index(std::vector<uint8_t>& out);
    // Fetch [offset,size) into the pinned staging pool then starst an async
    // COPY-queue transfer into the VRAM slot. Returns the copy fence value.
    uint64_t fetch_chunk(uint64_t offset, uint64_t size, ID3D12Resource* vram_slot);

    // Block until the returned fence value completes.
    void wait_copy(uint64_t fence_value);
    void* staging_ptr() const;
    ID3D12Resource* staging_resource() const { return m_staging ? m_staging->get() : nullptr; }

    const IocpStats& stats() const { return m_stats; }
    uint16_t port() const { return m_port; }

private:
    Device* m_device;
    std::string m_server_ip;
    uint16_t m_port;
    uint64_t m_slot_size;

    SOCKET m_sock{INVALID_SOCKET};
    HANDLE m_iocp{nullptr};
    std::unique_ptr<Buffer> m_staging; // pinned UPLOAD-heap pool (recv target)

    std::unique_ptr<Queue> m_copy_queue;
    std::unique_ptr<Fence> m_copy_fence;
    uint64_t m_copy_fence_value{0};
    uint64_t m_last_copy_fv{0};

    // Persistent COPY command allocator/list. Reused for every chunk so we
    // never free the allocator while a queued copy may still reference it.
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> m_copy_alloc;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> m_copy_list;
    std::unordered_set<ID3D12Resource*> m_copied_slots; // VRAM slots already in COPY_DEST

    IocpStats m_stats;

    bool connect_socket(const std::string& ip, uint16_t port);
    void recv_exact(uint8_t* dst, uint64_t n); // overlapped WSARecv via IOCP
    void send_request(const ChunkRequest& req);
    void ensure_copy_cmd();
    // Record+submit+signal a VRAM copy; returns the fence value (sets m_last_copy_fv).
    uint64_t submit_copy(ID3D12Resource* dst, ID3D12Resource* src, uint64_t size);
    static bool zstd_loadable();
    static size_t zstd_decompress(uint8_t* dst, uint64_t dst_cap, const uint8_t* src, uint64_t src_len);
};

} // namespace dxait

#endif // DXAIT_DXIOCP_HPP