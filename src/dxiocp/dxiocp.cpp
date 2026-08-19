#include "dxait/dxiocp.hpp"
#include <windows.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <cstring>
#include <atomic>

#pragma comment(lib, "ws2_32.lib")

namespace dxait {

namespace {
void wsa_start_once() {
    static bool done = false;
    if (!done) {
        WSADATA wsa;
        if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
            throw std::runtime_error("WSAStartup failed");
        }
        done = true;
    }
}

std::string winsock_error() {
    int err = WSAGetLastError();
    char buf[256]{};
    snprintf(buf, sizeof(buf), "WSA error %d", err);
    return buf;
}
} // namespace

// ---- IocpStreamServer ----

IocpStreamServer::IocpStreamServer(const std::string& payload_path, const std::vector<uint8_t>& index_blob,
                                   bool zstd_enabled, uint16_t port)
    : m_payload(payload_path), m_index(index_blob), m_zstd(zstd_enabled), m_port(port) {}

IocpStreamServer::~IocpStreamServer() {
    stop();
}

void IocpStreamServer::stop() {
    m_running.store(false);
    if (m_listen != INVALID_SOCKET) {
        closesocket(m_listen);
        m_listen = INVALID_SOCKET;
    }
    for (auto& t : m_accept_threads) {
        if (t.joinable()) t.join();
    }
    m_accept_threads.clear();
    for (auto& c : m_conns) {
        if (c.sock != INVALID_SOCKET) closesocket(c.sock);
        if (c.thread.joinable()) c.thread.join();
    }
    m_conns.clear();
}

bool IocpStreamServer::start() {
    wsa_start_once();
    m_listen = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (m_listen == INVALID_SOCKET) {
        std::cerr << "[DXIOCP] server socket() " << winsock_error() << "\n";
        return false;
    }
    int one = 1;
    setsockopt(m_listen, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&one), sizeof(one));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons(m_port);
    if (bind(m_listen, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == SOCKET_ERROR) {
        std::cerr << "[DXIOCP] server bind(port " << m_port << ") " << winsock_error() << "\n";
        return false;
    }
    if (listen(m_listen, 4) == SOCKET_ERROR) {
        std::cerr << "[DXIOCP] server listen() " << winsock_error() << "\n";
        return false;
    }

    // Discover the actual bound port (handles port 0).
    sockaddr_in bound{};
    int bound_len = sizeof(bound);
    if (getsockname(m_listen, reinterpret_cast<sockaddr*>(&bound), &bound_len) == 0) {
        m_port = ntohs(bound.sin_port);
    }

    m_running.store(true);
    m_accept_threads.emplace_back([this] { accept_loop(); });
    std::cout << "[DXIOCP] server listening on 127.0.0.1:" << m_port
              << " payload=" << m_payload << " payload_bytes=" << m_index.size()
              << " zstd=" << (m_zstd ? "on" : "off") << "\n";
    return true;
}

void IocpStreamServer::accept_loop() {
    while (m_running.load()) {
        SOCKET cs = accept(m_listen, nullptr, nullptr);
        if (cs == INVALID_SOCKET) {
            if (m_running.load()) continue;
            break;
        }
        int one = 1;
        setsockopt(cs, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char*>(&one), sizeof(one));
        int buf = 64 * 1024 * 1024;
        setsockopt(cs, SOL_SOCKET, SO_SNDBUF, reinterpret_cast<const char*>(&buf), sizeof(buf));
        setsockopt(cs, SOL_SOCKET, SO_RCVBUF, reinterpret_cast<const char*>(&buf), sizeof(buf));

        m_conns.push_back({cs, std::thread()});
        auto& conn = m_conns.back();
        conn.thread = std::thread([this, &conn] { serve_conn(&conn); });
    }
}

void IocpStreamServer::serve_conn(Conn* c) {
    SOCKET s = c->sock;
    std::ifstream payload(m_payload, std::ios::binary);
    while (m_running.load()) {
        ChunkRequest req;
        int total = 0;
        while (total < static_cast<int>(sizeof(ChunkRequest))) {
            int n = recv(s, reinterpret_cast<char*>(&req) + total,
                         sizeof(ChunkRequest) - total, 0);
            if (n <= 0) return;
            total += n;
        }

        if (req.op == 0) {
            ChunkHeader hdr{0, 0, static_cast<uint64_t>(m_index.size())};
            if (send(s, reinterpret_cast<const char*>(&hdr), sizeof(hdr), 0) != sizeof(hdr)) return;
            uint64_t sent = 0;
            while (sent < m_index.size()) {
                int n = send(s, reinterpret_cast<const char*>(m_index.data()) + sent,
                             static_cast<int>((std::min<uint64_t>)(m_index.size() - sent, 1 << 20)), 0);
                if (n <= 0) return;
                sent += n;
            }
        } else if (req.op == 1) {
            // Serve the raw payload file range.
            if (!payload.good()) { payload.clear(); payload.open(m_payload, std::ios::binary); }
            if (!payload.good()) return;
            payload.seekg(static_cast<std::streamoff>(req.offset));
            ChunkHeader hdr{1, m_zstd ? 1u : 0u, req.size};
            if (send(s, reinterpret_cast<const char*>(&hdr), sizeof(hdr), 0) != sizeof(hdr)) return;
            std::vector<char> buf(1 << 20);
            uint64_t remaining = req.size;
            while (remaining > 0) {
                uint64_t n = remaining > buf.size() ? buf.size() : remaining;
                payload.read(buf.data(), static_cast<std::streamsize>(n));
                std::streamsize got = payload.gcount();
                if (got <= 0) return;
                int sent_total = 0;
                while (sent_total < got) {
                    int k = send(s, buf.data() + sent_total, static_cast<int>(got - sent_total), 0);
                    if (k <= 0) return;
                    sent_total += k;
                }
                remaining -= static_cast<uint64_t>(got);
                if (got < static_cast<std::streamsize>(n)) { /* short read at EOF */ }
            }
        }
    }
}

// ---- IocpStreamClient ----

IocpStreamClient::IocpStreamClient(Device* device, const std::string& server_ip, uint16_t port,
                                   uint64_t slot_size)
    : m_device(device), m_server_ip(server_ip), m_port(port), m_slot_size(slot_size) {
    wsa_start_once();
}

IocpStreamClient::~IocpStreamClient() {
    if (m_sock != INVALID_SOCKET) {
        closesocket(m_sock);
        m_sock = INVALID_SOCKET;
    }
    if (m_iocp) {
        CloseHandle(m_iocp);
        m_iocp = nullptr;
    }
}

bool IocpStreamClient::connect(const std::string& ip, uint16_t port) {
    m_server_ip = ip;
    m_port = port;

    if (TDRGuard::is_device_removed(m_device->get())) {
        std::cerr << "[DXIOCP] connect aborted: GPU device removed\n";
        return false;
    }

    m_staging = m_device->create_buffer(m_slot_size, MemLocation::Upload);
    m_copy_queue = m_device->create_queue(QueueType::Copy);
    m_copy_fence = m_device->create_fence(0);

    warm_copy_queue(); // absorb one-time driver COPY-queue init outside the hot path

    if (!connect_socket(ip, port)) return false;

    // Create the completion port and associate the socket for overlapped WSARecv.
    m_iocp = CreateIoCompletionPort(reinterpret_cast<HANDLE>(m_sock), nullptr, 1, 0);
    if (!m_iocp) {
        std::cerr << "[DXIOCP] client CreateIoCompletionPort failed\n";
        return false;
    }
    return true;
}

bool IocpStreamClient::connect_socket(const std::string& ip, uint16_t port) {
    m_sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (m_sock == INVALID_SOCKET) return false;
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    inet_pton(AF_INET, ip.c_str(), &addr.sin_addr);
    addr.sin_port = htons(port);
    if (::connect(m_sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == SOCKET_ERROR) {
        std::cerr << "[DXIOCP] client connect() " << winsock_error() << "\n";
        return false;
    }
    int one = 1;
    setsockopt(m_sock, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char*>(&one), sizeof(one));
    int buf = 64 * 1024 * 1024;
    setsockopt(m_sock, SOL_SOCKET, SO_RCVBUF, reinterpret_cast<const char*>(&buf), sizeof(buf));
    setsockopt(m_sock, SOL_SOCKET, SO_SNDBUF, reinterpret_cast<const char*>(&buf), sizeof(buf));
    return true;
}

void IocpStreamClient::send_request(const ChunkRequest& req) {
    const char* p = reinterpret_cast<const char*>(&req);
    int total = 0;
    while (total < static_cast<int>(sizeof(ChunkRequest))) {
        int n = send(m_sock, p + total, sizeof(ChunkRequest) - total, 0);
        if (n <= 0) throw std::runtime_error("send_request failed");
        total += n;
    }
}

// Per-chunk overlapped WSARecv: submit WSARecv, wait for the completion-port
// completion, return when the full chunk has landed in the pinned staging pool.
void IocpStreamClient::recv_exact(uint8_t* dst, uint64_t n) {
    OVERLAPPED ov{};
    WSABUF wb{static_cast<ULONG>(n), reinterpret_cast<char*>(dst)};
    DWORD bytes_recv = 0;
    DWORD flags = 0;
    auto t0 = std::chrono::steady_clock::now();

    if (WSARecv(m_sock, &wb, 1, &bytes_recv, &flags, &ov, nullptr) == SOCKET_ERROR) {
        int err = WSAGetLastError();
        if (err == WSA_IO_PENDING) {
            ULONG_PTR key = 0;
            DWORD transferred = 0;
            OVERLAPPED* p_ov = nullptr;
            if (!GetQueuedCompletionStatus(m_iocp, &transferred, &key, &p_ov, INFINITE)) {
                throw std::runtime_error("client GetQueuedCompletionStatus failed");
            }
            bytes_recv = transferred;
        } else {
            throw std::runtime_error("client WSARecv failed " + winsock_error());
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    m_stats.latency_wsarecv_total_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    m_stats.chunks_received++;
    m_stats.bytes_received += bytes_recv;

    if (bytes_recv < n) {
        // TCP streamed partial data; pull the remainder blocking.
        uint8_t* p = dst + bytes_recv;
        uint64_t remaining = n - bytes_recv;
        while (remaining > 0) {
            int got = recv(m_sock, reinterpret_cast<char*>(p), static_cast<int>(remaining), 0);
            if (got <= 0) throw std::runtime_error("client recv() short read");
            p += got;
            remaining -= got;
        }
    }
}

bool IocpStreamClient::fetch_index(std::vector<uint8_t>& out) {
    ChunkRequest req{};
    req.op = 0;
    req.offset = 0;
    req.size = 0;
    send_request(req);

    ChunkHeader hdr;
    std::vector<uint8_t> hdr_buf(sizeof(hdr));
    auto hdr_start = std::chrono::steady_clock::now();
    recv_exact(hdr_buf.data(), sizeof(hdr));
    std::memcpy(&hdr, hdr_buf.data(), sizeof(hdr));
    (void)hdr_start;

    out.resize(hdr.size);
    recv_exact(out.data(), hdr.size);
    return true;
}

uint64_t IocpStreamClient::fetch_chunk(uint64_t offset, uint64_t size, ID3D12Resource* vram_slot) {
    if (!m_staging || !vram_slot) throw std::runtime_error("fetch_chunk: staging/slot null");
    ChunkHeader hdr{};

    // One staging buffer serializes all chunks: wait for the previous copy to
    // finish reading it before the next WSARecv overwrites the same memory.
    // (Async COPIES still overlap network receive of later chunks.)
    if (m_last_copy_fv > 0) wait_copy(m_last_copy_fv);

    ChunkRequest req{};
    req.op = 1;
    req.offset = offset;
    req.size = size;
    send_request(req);

    // Receive into normal system RAM: winsock cannot reliably lock a D3D12
    // upload-heap's mapped write-combine memory as an WSARecv buffer, so a
    // direct receive faults (WSAEFAULT). Copy into staging afterwards for DMA.
    std::vector<uint8_t> hdr_buf(sizeof(ChunkHeader));
    recv_exact(hdr_buf.data(), sizeof(ChunkHeader));
    std::memcpy(&hdr, hdr_buf.data(), sizeof(ChunkHeader));

    std::vector<uint8_t> payload(hdr.size);
    recv_exact(payload.data(), hdr.size);

    uint8_t* st = static_cast<uint8_t*>(m_staging->map());
    if (!st) throw std::runtime_error("fetch_chunk: staging map failed");

    uint64_t final_size = hdr.size;
    if (hdr.flags & 1u) {
        // Decompress in-place into the staging buffer (large dst capacity).
        size_t out_size = zstd_decompress(st, m_slot_size, payload.data(), hdr.size);
        if (out_size > 0) {
            m_stats.zstd_engaged++;
            final_size = out_size;
        } else {
            std::memcpy(st, payload.data(), hdr.size);
        }
    } else {
        std::memcpy(st, payload.data(), hdr.size);
    }
    m_staging->unmap();

    // Async COPY-queue DMA from pinned staging -> VRAM slot (persistent cmd objects).
    auto t0 = std::chrono::steady_clock::now();
    uint64_t fv = submit_copy(vram_slot, m_staging->get(), final_size);

    auto t1 = std::chrono::steady_clock::now();
    m_stats.copy_async_issued++;
    m_stats.copy_submit_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    return fv;
}

void IocpStreamClient::ensure_copy_cmd() {
    if (m_copy_alloc && m_copy_list) return;
    HRESULT hr = m_device->get()->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY, IID_PPV_ARGS(&m_copy_alloc));
    if (FAILED(hr)) throw std::runtime_error("copy allocator");
    hr = m_device->get()->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, m_copy_alloc.Get(), nullptr, IID_PPV_ARGS(&m_copy_list));
    if (FAILED(hr)) throw std::runtime_error("copy list");
    // Close immediately: a freshly created list is left in the recording state,
    // which makes the first m_copy_alloc->Reset() fail (E_FAIL). Idle it now.
    m_copy_list->Close();
}

uint64_t IocpStreamClient::submit_copy(ID3D12Resource* dst, ID3D12Resource* src, uint64_t size) {
    ensure_copy_cmd();
    HRESULT hr = m_copy_alloc->Reset();
    if (FAILED(hr)) {
        char buf[128];
        snprintf(buf, sizeof(buf), "copy alloc Reset hr=0x%08X removed=0x%08X",
                 static_cast<unsigned>(hr), static_cast<unsigned>(m_device->get()->GetDeviceRemovedReason()));
        throw std::runtime_error(buf);
    }
    hr = m_copy_list->Reset(m_copy_alloc.Get(), nullptr);
    if (FAILED(hr)) throw std::runtime_error("copy list Reset");

    // VRAM slot starts in COMMON; promote to COPY_DEST once per buffer.
    if (m_copied_slots.find(dst) == m_copied_slots.end()) {
        D3D12_RESOURCE_BARRIER b{};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition.pResource = dst;
        b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        b.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
        b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
        m_copy_list->ResourceBarrier(1, &b);
        m_copied_slots.insert(dst);
    }

    m_copy_list->CopyBufferRegion(dst, 0, src, 0, size);
    m_copy_list->Close();
    ID3D12CommandList* lists[] = { m_copy_list.Get() };
    m_copy_queue->execute(lists, 1);

    uint64_t fv = ++m_copy_fence_value;
    m_last_copy_fv = fv;
    m_copy_queue->signal(*m_copy_fence, fv);
    return fv;
}

// Fire one 4-byte COPY-queue operation and wait for it at connect time. AMD
// drivers lazily initialize the COPY queue / first command list; absorbing that
// one-time cost here keeps every later per-chunk wait_copy out of stall range.
void IocpStreamClient::warm_copy_queue() {
    std::vector<uint8_t> dummy(4, 0);
    auto src = m_device->create_buffer(dummy.size(), MemLocation::Upload);
    auto dst = m_device->create_buffer(dummy.size(), MemLocation::Default);
    void* mapped = src->map();
    if (mapped) std::memcpy(mapped, dummy.data(), dummy.size());
    src->unmap();

    uint64_t fv = submit_copy(dst->get(), src->get(), dummy.size());
    auto t0 = std::chrono::steady_clock::now();
    m_copy_fence->wait(fv);
    auto t1 = std::chrono::steady_clock::now();
    std::cout << "[DXIOCP] warm_copy_queue (4B) took "
              << std::chrono::duration<double, std::milli>(t1 - t0).count() << "ms\n";
}

void IocpStreamClient::warm_vram(ID3D12Resource* slot, uint32_t mb) {
    if (!slot || !m_staging) { warm_copy_queue(); return; }
    uint64_t bytes = static_cast<uint64_t>(mb) * 1024 * 1024;
    uint64_t n = bytes < m_slot_size ? bytes : m_slot_size;
    void* st = m_staging->map();
    if (st) { std::memset(st, 0xAB, static_cast<size_t>(n)); } // nonzero
    m_staging->unmap();

    uint64_t fv = submit_copy(slot, m_staging->get(), n);
    auto t0 = std::chrono::steady_clock::now();
    m_copy_fence->wait(fv);
    auto t1 = std::chrono::steady_clock::now();
    std::cout << "[DXIOCP] warm_vram wrote " << mb << "MB (nonzero) to VRAM ring in "
              << std::chrono::duration<double, std::milli>(t1 - t0).count() << "ms\n";
}

void IocpStreamClient::wait_copy(uint64_t fence_value) {
    auto t0 = std::chrono::steady_clock::now();
    m_copy_fence->wait(fence_value);
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    m_stats.copy_wait_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    m_stats.copies_that_waited++;
    std::cout << "[DXIOCP] wait_copy(fv=" << fence_value << ") blocked " << ms << "ms\n";
}

void* IocpStreamClient::staging_ptr() const {
    return m_staging ? m_staging->map() : nullptr;
}

static HMODULE g_zstd = nullptr;
static bool g_zstd_tried = false;

bool IocpStreamClient::zstd_loadable() {
    if (!g_zstd_tried) {
        g_zstd_tried = true;
        g_zstd = LoadLibraryA("zstd.dll");
        if (!g_zstd) g_zstd = LoadLibraryA("zstd_v1.dll");
    }
    return g_zstd != nullptr;
}

size_t IocpStreamClient::zstd_decompress(uint8_t* dst, uint64_t dst_cap, const uint8_t* src, uint64_t src_len) {
    if (!zstd_loadable()) return 0;
    // ZSTD_decompress(dst, dstCap, src, srcLen); returns decompressed size or 0.
    using decompress_fn = size_t(*)(void*, size_t, const void*, size_t);
    auto fn = reinterpret_cast<decompress_fn>(GetProcAddress(g_zstd, "ZSTD_decompress"));
    if (!fn) return 0;
    return fn(dst, dst_cap, src, static_cast<size_t>(src_len));
}

} // namespace dxait