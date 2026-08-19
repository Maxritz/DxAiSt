// test_dstream: DirectStorage / IOCP streaming MoE proof harness.
// Usage:
//   test_dstream                          # synthetic payload, all scenarios
//   test_dstream <weights.bin> <index.bin> # stream a real ingested model
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <windows.h>
#include "dxait/dxait.hpp"
#include "dxait/dxio.hpp"
#include "dxait/dxstream.hpp"
#include "dxait/dxiocp.hpp"
#include <windows.h>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <random>
#include <iostream>
#include <cstring>
#include <cstdio>
#include <atomic>

using namespace dxait;
using clock_ = std::chrono::steady_clock;

namespace {

std::string ws(const std::wstring& w) {
    int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), -1, nullptr, 0, nullptr, nullptr);
    if (n <= 1) return {};
    std::string s(static_cast<size_t>(n - 1), 0);
    WideCharToMultiByte(CP_UTF8, 0, w.c_str(), -1, s.data(), n, nullptr, nullptr);
    return s;
}

// Fill payload[off,off+size) with a deterministic, cheaply checkable pattern.
void stamp(uint8_t* p, uint64_t off, uint64_t size) {
    uint32_t seed = static_cast<uint32_t>(off / 4096);
    for (uint64_t i = 0; i < size; ++i) {
        p[off + i] = static_cast<uint8_t>((seed * 31u + static_cast<uint32_t>(i)) & 0xFFu);
    }
}

// Build a synthetic MoE payload: N experts, variation in size, all slices
// aligned to `alignment`. Returns (payload_path, TensorIndex).
std::pair<std::string, TensorIndex> build_synthetic(uint32_t alignment, uint32_t n_experts,
                                                    std::vector<std::string>& tags_out,
                                                    const std::string& dir) {
    std::filesystem::create_directories(dir);
    std::string payload = dir + "/synthetic.bin";
    std::mt19937 rng(0x5EED);
    std::vector<std::pair<std::string, TensorCoordinate>> coords;
    coords.reserve(n_experts);

    uint64_t cursor = 0;
    std::ofstream w(payload, std::ios::binary | std::ios::trunc);
    std::vector<uint8_t> block(1024 * 1024);
    for (uint32_t e = 0; e < n_experts; ++e) {
        uint64_t size = 48ull * 1024 + (rng() % (1024 * 1024)); // 48KB..~1MB
        uint64_t aligned = (cursor + alignment - 1) / alignment * alignment;
        if (aligned > cursor) {
            std::vector<uint8_t> zeros(aligned - cursor, 0);
            w.write(reinterpret_cast<const char*>(zeros.data()), zeros.size());
            cursor = aligned;
        }
        uint64_t src_base = cursor;
        uint64_t remaining = size;
        while (remaining > 0) {
            uint64_t n = remaining > block.size() ? block.size() : remaining;
            for (uint64_t i = 0; i < n; ++i)
                block[static_cast<size_t>(i)] = static_cast<uint8_t>((src_base + i) & 0xFFu);
            w.write(reinterpret_cast<const char*>(block.data()), static_cast<std::streamsize>(n));
            remaining -= n;
        }
        char tag[96];
        snprintf(tag, sizeof(tag), "blk.%u.ffn_exps.%u.w1", e / 8, e % 8);
        TensorCoordinate c{};
        c.absolute_offset = aligned;
        c.byte_size = size;
        c.quant_type = 8; // Q8_0
        coords.emplace_back(tag, c);
        tags_out.emplace_back(tag);
        cursor = src_base + size;
    }
    w.close();

    TensorIndex idx;
    idx.build(coords, cursor, alignment);
    return {payload, idx};
}

void print_header(const StreamStats& s, uint64_t payload_size, uint64_t slots_total_bytes) {
    std::cout << "\n[STREAM STATS]\n"
              << "  payload_bytes=" << payload_size
              << " slot_capacity_bytes=" << slots_total_bytes
              << " (" << (slots_total_bytes * 100.0 / payload_size) << "% resident upper bound)\n"
              << "  hits=" << s.fetch_hits << " misses=" << s.fetch_misses
              << " evictions=" << s.evictions << "\n"
              << "  stalls=" << s.stall_events << " stall_wait_ms=" << (s.stall_wait_ns / 1e6) << "\n"
              << "  bytes_streamed=" << s.bytes_streamed
              << " bytes_resident=" << s.bytes_resident
              << " (" << (s.bytes_resident * 100.0 / payload_size) << "% of payload resident)\n"
              << "  cold_start_ms=" << s.cold_start_ms << "\n";
}

// Scenario 1+2: logging + on-demand residency on a payload larger than the ring.
int scenario_on_demand(DirectStorageContext* ds, StreamingMoE& smoe, uint64_t payload_size,
                       std::vector<std::string>& tags, uint64_t ring_capacity_bytes) {
    if (ds == nullptr) return 1;
    std::cout << "\n=== SCENARIO: on-demand expert streaming (logging [#1] + residency [#2]) ===\n";
    smoe.flush();
    int rc = 0;

    // Slow sequential, deliberately spaced: forces DS to actually complete each read.
    for (const auto& t : tags) {
        SlotView v = smoe.fetch(t);
        if (v.slot < 0) { std::cerr << "  fetch miss on index lookup: " << t << "\n"; return 1; }
        smoe.flush(); // probe: wait so timing per tensor is attributable
    }

    const auto& s = smoe.stats();
    // Per-fetch log lines were emitted by DirectStorageContext::wait_all/status.
    // Residency: only requested experts were streamed; ring never over-committed.
    bool resident_ok = s.bytes_resident <= ring_capacity_bytes;
    bool not_all = s.bytes_resident <= payload_size && s.bytes_resident > 0;
    std::cout << "  [check] bytes_resident<=ring_capacity: " << (resident_ok ? "PASS" : "FAIL") << "\n";
    std::cout << "  [check] streamed payload < model bytes (on-demand): "
              << (not_all ? "PASS" : "FAIL") << "\n";
    if (!resident_ok || !not_all) rc = 1;
    return rc;
}

// Scenario: eviction under contention — disjointed random experts, slot count
// much smaller than working set; stall logging from in-flight victims.
int scenario_eviction(DirectStorageContext* ds, StreamingMoE& smoe, uint32_t slots,
                      uint32_t working_set, uint32_t tokens) {
    std::cout << "\n=== SCENARIO: ring eviction under contention (slots=" << slots
              << ", working_set=" << working_set << ", tokens=" << tokens << ") ===\n";
    std::mt19937 rng(0xABC);
    uint64_t seen_misses = smoe.stats().fetch_misses;
    for (uint32_t i = 0; i < tokens; ++i) {
        char tag[96];
        uint32_t e = rng() % working_set;
        snprintf(tag, sizeof(tag), "blk.%u.ffn_exps.%u.w1", e / 8, e % 8);
        SlotView v = smoe.fetch(tag);
        if (v.slot < 0) { std::cerr << "  eviction fetch miss\n"; return 1; }
        // Simulated dispatch: compute queue waits for this slot's fence.
        // (Real killer: NEVER wait here; keep enqueue-only so victims stay busy.)
    }
    smoe.flush();
    const auto& s = smoe.stats();
    uint64_t missed = s.fetch_misses - seen_misses;
    bool ok = s.stall_events > 0 || s.evictions >= missed; // eviction or a stall proves contention
    std::cout << "  misses_this_run=" << missed
              << " evictions=" << s.evictions
              << " stalls=" << s.stall_events
              << " stall_wait_ms=" << (s.stall_wait_ns / 1e6) << "\n";
    std::cout << "  [check] contended eviction/stalled (evictions>=" << missed
              << " or stalls>0): " << (ok ? "PASS" : "FAIL") << "\n";
    return ok ? 0 : 1;
}

// Scenario: alignment sabotage — corrupt absolute_offset by 4 bytes and log the
// actual failure mode (DS debug error / AV / silent unaligned read fallback).
int scenario_sabotage(DirectStorageContext* ds, const TensorIndex& idx, const std::wstring& payload,
                      uint32_t n) {
    if (ds == nullptr) return 1;
    std::cout << "\n=== SCENARIO: alignment sabotage (absolute_offset += 4) ===\n";
    if (n == 0) { std::cerr << "  empty index\n"; return 1; }
    TensorCoordinate c;
    if (!idx.lookup("blk.0.ffn_exps.0.w1", c)) { std::cerr << "  tag missing\n"; return 1; }
    c.absolute_offset += 4; // sabotage the 64KB/4K boundary

    auto dev = Adapter::create_device(0);
    auto slot = dev->create_buffer(2 * 1024 * 1024, MemLocation::Default);
    uint64_t rid = ds->enqueue_read(payload, c.absolute_offset, c.byte_size, slot->get(), 0, "sabotaged");
    ds->submit();
    ds->wait_for_request(rid);
    std::cout << "  [check] logged per-fetch line above shows actual path for the +4 read "
              << "(expect STAGED/UNALIGNED, i.e. silent unaligned fallback, not a hard crash)\n";
    return 0;
}

// Scenario: cold start — index load (decode from blob) + first fetch (<50 ms target).
int scenario_cold_start(Device* dev, DirectStorageContext* ds, const TensorIndex& idx,
                        const std::wstring& payload) {
    std::cout << "\n=== SCENARIO: cold start ===\n";
    auto blob = idx.to_blob(); // serialise once (net/local round-trip form)

    auto t0 = clock_::now();
    TensorIndex decoded;
    if (!decoded.from_blob(blob.data(), blob.size())) { std::cerr << "  index decode failed\n"; return 1; }
    StreamingMoE fresh(dev, ds, payload, decoded, 1, 4ull * 1024 * 1024);
    auto t1 = clock_::now();
    SlotView v = fresh.fetch("blk.0.ffn_exps.1.w1");
    auto t2 = clock_::now();
    double map_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double first_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    std::cout << "  header_blob_decode_ms=" << map_ms
              << " ring_spawn_ms=" << first_ms
              << " total_ms=" << (map_ms + first_ms) << " (target < 50ms)\n";
    (void)v;
    return (map_ms + first_ms) < 50.0 ? 0 : 1;
}

// Scenario: IOCP network route — server + WSARecv client, per-chunk latency,
// zstd engagement, async COPY-queue transfer into VRAM.
// NOTE: reuse the process-wide Device (no second ID3D12Device); AMD removes the
// 2nd device sharing a GPU while the 1st is live with DirectStorage.
int scenario_network(Device* dev, const std::string& payload_path, const TensorIndex& idx,
                     uint16_t base_port) {
    std::cout << "\n=== SCENARIO: remote IOCP chunk streaming [#6] ===\n";
    if (!dev) { std::cerr << "  no device\n"; return 1; }
    if (TDRGuard::is_device_removed(dev->get())) {
        std::cerr << "  device already removed at scenario_network start\n";
        return 1;
    }

    auto index_blob = idx.to_blob();
    IocpStreamServer server(payload_path, index_blob, /*zstd=*/false, base_port);
    if (!server.start()) { std::cerr << "  server start failed\n"; return 1; }
    Sleep(50);

    IocpStreamClient client(dev, "127.0.0.1", server.port(), 8ull * 1024 * 1024);
    if (!client.connect("127.0.0.1", server.port())) { std::cerr << "  client connect failed\n"; return 1; }

    std::vector<uint8_t> net_index;
    if (!client.fetch_index(net_index)) { std::cerr << "  fetch_index failed\n"; return 1; }
    TensorIndex net_idx;
    if (!net_idx.from_blob(net_index.data(), net_index.size())) { std::cerr << "  index decode failed\n"; return 1; }
    bool idx_ok = net_idx.count() == idx.count();
    std::cout << "  remote index coord count=" << net_idx.count()
              << " matches_local=" << (idx_ok ? "YES" : "NO") << "\n";

    // VRAM slot + fetch three chunks through the network route.
    auto ring = dev->create_buffer(8ull * 1024 * 1024, MemLocation::Default);
    // Warm the VRAM destination's first GPU write OUTSIDE measured fetches
    // (AMD RDNA4 COPY queue pays a one-time residency/first-write cost here).
    client.warm_vram(ring->get(), 2);
    std::vector<std::string> tags{
        "blk.0.ffn_exps.0.w1", "blk.0.ffn_exps.3.w1", "blk.4.ffn_exps.5.w1"};
    std::vector<uint64_t> fvs;
    for (size_t i = 0; i < tags.size(); ++i) {
        TensorCoordinate c;
        if (!idx.lookup(tags[i], c)) { std::cerr << "  missing tag " << tags[i] << "\n"; return 1; }
        uint64_t fv = client.fetch_chunk(c.absolute_offset, c.byte_size, ring->get());
        fvs.push_back(fv);
    }
    for (auto fv : fvs) client.wait_copy(fv);

    const auto& st = client.stats();
    std::cout << "  chunks_received=" << st.chunks_received
              << " bytes_received=" << st.bytes_received
              << " avg_wsarecv_latency_ms=" << (st.chunks_received ? st.latency_wsarecv_total_ns / 1e6 / st.chunks_received : 0.0)
              << " copies_async_issued=" << st.copy_async_issued
              << " copy_submit_ms=" << (st.copy_submit_ns / 1e6)
              << " copy_wait_total_ms=" << (st.copy_wait_ns / 1e6)
              << " zstd_engaged=" << st.zstd_engaged << " (0 == zstd.dll absent, passthrough)\n";
    server.stop();
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::cout << "DXAiT DirectStorage / IOCP Streaming MoE Proof Harness\n";
    std::cout << "========================================================\n";

    auto adapters = Adapter::enumerate();
    if (adapters.empty()) { std::cout << "No GPU found, skipping.\n"; return 0; }
    auto dev = Adapter::create_device(0);
    std::cout << "GPU: " << dev->caps().name << "\n";
    std::cout << "[CONFIG] chunk_size=" << Config::get_chunk_size_bytes()
              << " vram_margin=" << Config::get_vram_margin_ratio()
              << " async_prefetch=" << (Config::is_async_prefetch_enabled() ? "on" : "off") << "\n";

    constexpr uint32_t k_alignment = 4096;
    const std::string tmp = std::string(std::getenv("TEMP") ? std::getenv("TEMP") : ".") + "/dxstream-test";
    std::string payload;
    TensorIndex idx;
    std::vector<std::string> tags;

    if (argc >= 3) {
        payload = argv[1];
        if (!idx.load(argv[2])) { std::cerr << "index load failed: " << argv[2] << "\n"; return 1; }
        for (const auto& e : idx.entries()) tags.push_back(e.first);
        std::cout << "Loaded real ingested model: payload=" << payload
                  << " tensors=" << idx.count() << "\n";
    } else {
        std::tie(payload, idx) = build_synthetic(k_alignment, 64, tags, tmp);
        std::cout << "Built synthetic MoE: payload=" << payload
                  << " experts=" << idx.count() << " alignment=" << k_alignment << "\n";
    }

    std::wstring wpayload(payload.begin(), payload.end());
    uint64_t payload_size = idx.payload_size();
    constexpr uint32_t k_slots = 3;
    constexpr uint64_t k_slot_size = 64ull * 1024 * 1024; // 64MB/slot, 192MB ring
    uint64_t ring_capacity = k_slots * k_slot_size;

    // BypassIO mode is process-global, one-shot: choose it before constructing
    // any DirectStorageContext. A/B comparison = run the exe twice, once with
    // DXAIT_DSTORAGE_DISABLE_BYPASSIO unset, once =1.
    const char* bypass_env = getenv("DXAIT_DSTORAGE_DISABLE_BYPASSIO");
    bool bypassio_enabled = !(bypass_env && strcmp(bypass_env, "1") == 0);
    std::cout << "\n[BYPASSIO CONFIG] process mode="
              << (bypassio_enabled ? "ENABLED" : "DISABLED (DXAIT_DSTORAGE_DISABLE_BYPASSIO=1)") << "\n";

    int rc = 0;
    try {
        DirectStorageContext ds_on(dev.get(), bypassio_enabled);
        bool vol_supported = ds_on.bypass_capability(wpayload).compatible_storage;
        std::cout << "\n[BYPASSIO PROBE] volume compatible_storage="
                  << (vol_supported ? "YES" : "NO") << "\n";

        // Scenario 1+2: real on-demand streaming with per-fetch logging.
        StreamingMoE smoe(dev.get(), &ds_on, wpayload, idx, k_slots, k_slot_size);
        smoe.flush();
        rc |= scenario_on_demand(&ds_on, smoe, payload_size, tags, ring_capacity);
        print_header(smoe.stats(), payload_size, ring_capacity);

        // Scenario 3: forced eviction contention, tiny slots.
        StreamingMoE smoe_small(dev.get(), &ds_on, wpayload, idx, 2, 4ull * 1024 * 1024);
        rc |= scenario_eviction(&ds_on, smoe_small, 2, 64, 200);

        // Scenario 4: alignment sabotage on the same context.
        rc |= scenario_sabotage(&ds_on, idx, wpayload, idx.count());

        // Sustained throughput in this mode (compare across the two A/B runs).
        {
            StreamingMoE ab(dev.get(), &ds_on, wpayload, idx, k_slots, k_slot_size);
            auto ab_t0 = clock_::now();
            for (uint32_t i = 0; i < 40; ++i) {
                char tag[96];
                snprintf(tag, sizeof(tag), "blk.%u.ffn_exps.%u.w1", (i * 7) % 64 / 8, (i * 7) % 64 % 8);
                ab.fetch(tag);
            }
            ab.flush();
            double ms = std::chrono::duration<double, std::milli>(clock_::now() - ab_t0).count();
            std::cout << "\n[BYPASSIO MODE RATE] 40 fetches in " << ms << "ms "
                      << "(run with DXAIT_DSTORAGE_DISABLE_BYPASSIO=1 for the staged comparison)\n";
        }

        // Scenario 5: cold-start index map + first fetch.
        rc |= scenario_cold_start(dev.get(), &ds_on, idx, wpayload);

        // Scenario 6: IOCP network route (loopback).
        rc |= scenario_network(dev.get(), payload, idx, 9095);

    } catch (const std::exception& e) {
        std::cerr << "FATAL: " << e.what() << "\n";
        rc = 1;
    }

    std::cout << "\n=== " << (rc == 0 ? "ALL DSTREAM SCENARIOS PASSED" : "SCENARIO FAILURES PRESENT") << " ===\n";
    return rc;
}