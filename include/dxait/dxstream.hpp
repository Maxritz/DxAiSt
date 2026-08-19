#ifndef DXAIT_DXSTREAM_HPP
#define DXAIT_DXSTREAM_HPP

#include "dxait.hpp"
#include "dxio.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <deque>

namespace dxait {

#pragma pack(push, 1)
// Coordinate map entry, serialised into the network header blob verbatim.
struct TensorCoordinate {
    uint64_t absolute_offset; // offset in weights payload (weights.bin / gguf data)
    uint64_t byte_size;       // tensor payload length
    uint32_t quant_type;      // ggml quant type id
    uint32_t padding;
};
#pragma pack(pop)

struct IndexHeader {
    uint64_t magic;          // 'DXIstream'
    uint32_t count;          // number of coordinates
    uint32_t alignment;      // required byte alignment of absolute_offset
    uint64_t payload_size;   // total size of the backing payload file
    uint64_t reserved[4];
};

class TensorIndex {
public:
    TensorIndex() = default;

    // Build from a list of (tag, coordinate). Tags are the MoE tensor names.
    void build(std::vector<std::pair<std::string, TensorCoordinate>> entries,
               uint64_t payload_size, uint32_t alignment = 4096);

    // Persist/flatten (local file or network header blob front).
    bool save(const std::string& path) const;
    bool load(const std::string& path);
    std::vector<uint8_t> to_blob() const;
    bool from_blob(const uint8_t* data, size_t size);

    // Coordinate lookup by tensor name (hash). Returns false if absent.
    bool lookup(const std::string& tag, TensorCoordinate& out) const;
    uint64_t lookup_hash(const std::string& tag) const;

    const std::vector<std::pair<std::string, TensorCoordinate>>& entries() const { return m_entries; }
    const IndexHeader& header() const { return m_header; }
    uint32_t count() const { return m_header.count; }
    uint64_t payload_size() const { return m_header.payload_size; }

private:
    IndexHeader m_header{};
    std::vector<std::pair<std::string, TensorCoordinate>> m_entries;
    std::unordered_map<uint64_t, uint32_t> m_hash_to_index;
};

struct StreamStats {
    uint64_t fetch_hits{0};
    uint64_t fetch_misses{0};
    uint64_t evictions{0};
    uint64_t stall_events{0};   // count of times a slot was busy by in-flight work
    uint64_t stall_wait_ns{0};  // cumulative time spent waiting for busy slots
    uint64_t bytes_streamed{0};
    uint64_t bytes_resident{0}; // current total bytes seated in VRAM slots
    double  cold_start_ms{0.0}; // index load -> first fetch enqueued
};

struct SlotView {
    int slot{-1};
    uint64_t fence_value{0};
    ID3D12Resource* resource{nullptr};   // VRAM slot (for compute reads)
    bool is_hit{false};
};

// VRAM ring buffer backed by DirectStorage. Slots seat TensorCoordinate payloads;
// fence per slot lets a compute queue wait for data arrival before dispatch.
class StreamingMoE {
public:
    StreamingMoE(Device* device, DirectStorageContext* dstorage,
                 const std::wstring& payload_path, const TensorIndex& index,
                 uint32_t slot_count = 3, uint64_t slot_size = 128ull * 1024 * 1024);
    ~StreamingMoE() = default;

    // Request a tensor by tag. If not resident, evicts LRU victim (waiting for it
    // if it still has in-flight work) and enqueues a DirectStorage read into the
    // freed slot. Returns the slot + fence value the compute queue must wait on.
    SlotView fetch(const std::string& tag);

    // Utility for tests: force the slot's in-flight DS read to complete.
    void flush(); // waits all outstanding slot fences

    const StreamStats& stats() const { return m_stats; }
    uint32_t slot_count() const { return m_slot_count; }
    uint64_t slot_size() const { return m_slot_size; }
    uint32_t alignment() const { return m_index.header().alignment; }

private:
    Device* m_device;
    DirectStorageContext* m_dstorage;
    std::wstring m_payload_path;
    TensorIndex m_index;
    uint32_t m_slot_count;
    uint64_t m_slot_size;

    std::vector<std::unique_ptr<Buffer>> m_slots; // VRAM slot buffers
    std::vector<std::unique_ptr<Fence>> m_slot_fences;
    std::vector<uint64_t> m_slot_versions; // last fence value per slot
    std::vector<bool> m_slot_has_payload;
    std::vector<uint64_t> m_slot_resident_bytes;

    // resident_tag -> slot index; LRU ordering
    std::unordered_map<uint64_t, int> m_resident;
    std::deque<int> m_lru;
    std::vector<uint64_t> m_lru_last_tick; // monotonic access tick for LRU age

    StreamStats m_stats;
    uint64_t m_tick{0};
};

// Offline ingestor helper: repackage GGUF tensor table into an aligned (64KB)
// flat payload + TensorCoordinate index. Returns coordinate entries in order.
// Reads the payload bytes directly from the source GGUF and writes weights.bin
// with 64KB-strided slices; coordinates point into the repacked file.
struct IngestResult {
    std::string payload_path;                 // weights.bin (aligned repack)
    std::vector<std::pair<std::string, TensorCoordinate>> coords;
    uint64_t payload_size{0};
};
bool ingest_gguf(const std::string& gguf_path, const std::string& out_dir,
                 uint32_t alignment, IngestResult& out);

} // namespace dxait

#endif // DXAIT_DXSTREAM_HPP