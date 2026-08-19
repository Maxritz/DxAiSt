#include "dxait/dxstream.hpp"
#include <windows.h>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <algorithm>
#include <cstring>
#include <iostream>

namespace dxait {

static constexpr uint64_t k_index_magic = 0x7375697374495844ull; // 'DXIstream'

static uint64_t fnv1a64(const std::string& s) {
    uint64_t h = 14695981039346656037ULL;
    for (char c : s) {
        h ^= static_cast<uint8_t>(c);
        h *= 1099511628211ULL;
    }
    return h;
}

// ---- TensorIndex ----

void TensorIndex::build(std::vector<std::pair<std::string, TensorCoordinate>> entries,
                        uint64_t payload_size, uint32_t alignment) {
    m_entries = std::move(entries);
    m_header.magic = k_index_magic;
    m_header.count = static_cast<uint32_t>(m_entries.size());
    m_header.alignment = alignment;
    m_header.payload_size = payload_size;
    std::fill(std::begin(m_header.reserved), std::end(m_header.reserved), 0);
    m_hash_to_index.clear();
    for (uint32_t i = 0; i < m_entries.size(); ++i) {
        m_hash_to_index[fnv1a64(m_entries[i].first)] = i;
    }
}

bool TensorIndex::save(const std::string& path) const {
    auto blob = to_blob();
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f.write(reinterpret_cast<const char*>(blob.data()), static_cast<std::streamsize>(blob.size()));
    return f.good();
}

bool TensorIndex::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    std::streamsize size = f.tellg();
    if (size <= static_cast<std::streamsize>(sizeof(IndexHeader))) return false;
    f.seekg(0);
    std::vector<uint8_t> blob(static_cast<size_t>(size));
    if (!f.read(reinterpret_cast<char*>(blob.data()), size)) return false;
    return from_blob(blob.data(), blob.size());
}

std::vector<uint8_t> TensorIndex::to_blob() const {
    std::vector<uint8_t> blob(sizeof(IndexHeader) + m_entries.size() * sizeof(TensorCoordinate), 0);
    std::memcpy(blob.data(), &m_header, sizeof(IndexHeader));
    for (size_t i = 0; i < m_entries.size(); ++i) {
        std::memcpy(blob.data() + sizeof(IndexHeader) + i * sizeof(TensorCoordinate),
                    &m_entries[i].second, sizeof(TensorCoordinate));
    }
    return blob;
}

bool TensorIndex::from_blob(const uint8_t* data, size_t size) {
    if (size < sizeof(IndexHeader)) return false;
    std::memcpy(&m_header, data, sizeof(IndexHeader));
    if (m_header.magic != k_index_magic) return false;
    size_t coord_bytes = static_cast<size_t>(m_header.count) * sizeof(TensorCoordinate);
    if (size < sizeof(IndexHeader) + coord_bytes) return false;

    m_entries.clear();
    m_entries.reserve(m_header.count);
    m_hash_to_index.clear();
    for (uint32_t i = 0; i < m_header.count; ++i) {
        TensorCoordinate c;
        std::memcpy(&c, data + sizeof(IndexHeader) + i * sizeof(TensorCoordinate), sizeof(TensorCoordinate));
        m_entries.emplace_back(std::to_string(i), c); // remote entries are numeric ids
        m_hash_to_index[fnv1a64(m_entries.back().first)] = i;
    }
    return true;
}

bool TensorIndex::lookup(const std::string& tag, TensorCoordinate& out) const {
    auto it = m_hash_to_index.find(fnv1a64(tag));
    if (it == m_hash_to_index.end()) return false;
    out = m_entries[it->second].second;
    return true;
}

uint64_t TensorIndex::lookup_hash(const std::string& tag) const {
    return fnv1a64(tag);
}

// ---- StreamingMoE ----

StreamingMoE::StreamingMoE(Device* device, DirectStorageContext* dstorage,
                           const std::wstring& payload_path, const TensorIndex& index,
                           uint32_t slot_count, uint64_t slot_size)
    : m_device(device), m_dstorage(dstorage), m_payload_path(payload_path),
      m_index(index), m_slot_count(slot_count), m_slot_size(slot_size) {

    m_slots.reserve(slot_count);
    m_slot_fences.reserve(slot_count);
    m_slot_versions.assign(slot_count, 0);
    m_slot_has_payload.assign(slot_count, false);
    m_slot_resident_bytes.assign(slot_count, 0);
    m_lru_last_tick.assign(slot_count, 0);

    for (uint32_t i = 0; i < slot_count; ++i) {
        m_slots.push_back(device->create_buffer(slot_size, MemLocation::Default));
        m_slot_fences.push_back(device->create_fence(0));
    }
    m_dstorage->open_file(payload_path); // prime file cache + bypassio probe
}

SlotView StreamingMoE::fetch(const std::string& tag) {
    TensorCoordinate coord;
    if (!m_index.lookup(tag, coord)) {
        SlotView bad;
        bad.slot = -1;
        return bad;
    }

    ++m_tick;
    uint64_t hash = m_index.lookup_hash(tag);
    auto hit = m_resident.find(hash);
    if (hit != m_resident.end()) {
        int s = hit->second;
        m_lru_last_tick[s] = m_tick;
        ++m_stats.fetch_hits;
        return SlotView{s, m_slot_versions[s], m_slots[s]->get(), true};
    }

    // MISS: evict the least-recently-used slot.
    ++m_stats.fetch_misses;
    int victim = -1;
    uint64_t oldest = UINT64_MAX;
    for (uint32_t i = 0; i < m_slot_count; ++i) {
        if (m_lru_last_tick[i] < oldest) {
            oldest = m_lru_last_tick[i];
            victim = static_cast<int>(i);
        }
    }
    if (victim < 0) {
        SlotView bad;
        bad.slot = -1;
        return bad;
    }

    // If the victim still has in-flight (un-signaled) work, that is a stall.
    if (m_slot_has_payload[victim]) {
        auto t0 = std::chrono::steady_clock::now();
        if (!m_slot_fences[victim]->is_completed(m_slot_versions[victim])) {
            ++m_stats.stall_events;
            m_slot_fences[victim]->wait(m_slot_versions[victim]);
        }
        auto t1 = std::chrono::steady_clock::now();
        m_stats.stall_wait_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        ++m_stats.evictions;
    }

    m_stats.bytes_streamed += coord.byte_size;
    m_stats.bytes_resident -= m_slot_resident_bytes[victim];
    m_resident.erase(hash);

    // Enqueue DS read into the victim slot; signal its fence when the read drains.
    uint64_t next_val = m_slot_versions[victim] + 1;
    m_dstorage->enqueue_read(m_payload_path, coord.absolute_offset, coord.byte_size,
                             m_slots[victim]->get(), 0, tag);
    m_dstorage->enqueue_signal_and_submit(m_slot_fences[victim]->get(), next_val);

    m_slot_versions[victim] = next_val;
    m_slot_has_payload[victim] = true;
    m_slot_resident_bytes[victim] = coord.byte_size;
    m_stats.bytes_resident += coord.byte_size;
    m_lru_last_tick[victim] = m_tick;

    m_resident[hash] = victim;
    return SlotView{victim, next_val, m_slots[victim]->get(), false};
}

void StreamingMoE::flush() {
    for (uint32_t i = 0; i < m_slot_count; ++i) {
        if (m_slot_has_payload[i]) {
            m_slot_fences[i]->wait(m_slot_versions[i]);
        }
    }
}

// ---- Offline ingestor: GGUF -> aligned weights.bin + coordinate index ----

namespace {

struct GgufTensor {
    std::string name;
    uint64_t offset; // relative to data section start
    uint64_t size;   // payload bytes
    uint32_t type;
};

bool read_gguf_string(std::ifstream& f, uint64_t& off, std::string& out) {
    uint64_t len = 0;
    if (!f.seekg(static_cast<std::streamoff>(off))) return false;
    if (!f.read(reinterpret_cast<char*>(&len), 8)) return false;
    if (len > (1u << 26)) return false; // sanity: 64MB name cap
    out.resize(static_cast<size_t>(len));
    if (!f.read(out.data(), static_cast<std::streamsize>(len))) return false;
    off += 8 + len;
    return true;
}

void skip_gguf_val(std::ifstream& f, uint64_t& off, uint32_t type) {
    switch (type) {
        case 0: case 1: case 7: off += 1; break;
        case 2: case 3: case 4: case 5: case 6: off += 4; break;
        case 10: case 11: case 12: off += 8; break;
        case 8: {
            uint64_t len = 0;
            if (f.seekg(static_cast<std::streamoff>(off)) &&
                f.read(reinterpret_cast<char*>(&len), 8)) {
                off += 8 + len;
            }
            break;
        }
        case 9: {
            uint32_t elem = 0;
            uint64_t n = 0;
            if (f.seekg(static_cast<std::streamoff>(off)) &&
                f.read(reinterpret_cast<char*>(&elem), 4) &&
                f.read(reinterpret_cast<char*>(&n), 8)) {
                off += 12;
                for (uint64_t i = 0; i < n; ++i) skip_gguf_val(f, off, elem);
            }
            break;
        }
        default: break;
    }
}

} // namespace

bool ingest_gguf(const std::string& gguf_path, const std::string& out_dir,
                 uint32_t alignment, IngestResult& out) {
    std::ifstream f(gguf_path, std::ios::binary);
    if (!f) return false;

    uint32_t magic = 0;
    if (!f.read(reinterpret_cast<char*>(&magic), 4) || magic != 0x46554747) return false;
    uint32_t version = 0;
    uint64_t tensor_count = 0, kv_count = 0;
    if (!f.read(reinterpret_cast<char*>(&version), 4)) return false;
    if (!f.read(reinterpret_cast<char*>(&tensor_count), 8)) return false;
    if (!f.read(reinterpret_cast<char*>(&kv_count), 8)) return false;
    uint64_t off = 24;

    for (uint64_t i = 0; i < kv_count; ++i) {
        std::string k;
        if (!read_gguf_string(f, off, k)) return false;
        uint32_t vt = 0;
        if (!f.seekg(static_cast<std::streamoff>(off))) return false;
        if (!f.read(reinterpret_cast<char*>(&vt), 4)) return false;
        off += 4;
        skip_gguf_val(f, off, vt);
    }

    std::vector<GgufTensor> tensors;
    tensors.reserve(static_cast<size_t>(tensor_count));
    for (uint64_t i = 0; i < tensor_count; ++i) {
        GgufTensor t;
        if (!read_gguf_string(f, off, t.name)) return false;
        uint32_t n_dims = 0;
        if (!f.seekg(static_cast<std::streamoff>(off))) return false;
        if (!f.read(reinterpret_cast<char*>(&n_dims), 4)) return false;
        off += 4;
        f.seekg(static_cast<std::streamoff>(off + 8ull * n_dims));
        off += 8ull * n_dims;
        uint32_t type = 0;
        uint64_t toff = 0;
        if (!f.read(reinterpret_cast<char*>(&type), 4)) return false;
        if (!f.read(reinterpret_cast<char*>(&toff), 8)) return false;
        off += 12;
        t.type = type;
        t.offset = toff;
        t.size = 0;
        tensors.push_back(std::move(t));
    }

    uint64_t data_start = off; // payload section begins here

    f.clear();
    f.seekg(0, std::ios::end);
    uint64_t file_size = static_cast<uint64_t>(f.tellg());

    // Byte length = gap to the next tensor's relative offset; last ends at EOF.
    std::sort(tensors.begin(), tensors.end(),
              [](const GgufTensor& a, const GgufTensor& b) { return a.offset < b.offset; });
    for (size_t i = 0; i + 1 < tensors.size(); ++i) {
        tensors[i].size = tensors[i + 1].offset - tensors[i].offset;
    }
    if (!tensors.empty()) {
        uint64_t data_end = file_size;
        uint64_t last_end = data_start + tensors.back().offset;
        tensors.back().size = last_end <= data_end ? (data_end - last_end) : 0;
    }

    // Repack into alignment-strided slices inside weights.bin.
    std::filesystem::path out_path = std::filesystem::path(out_dir) / "weights.bin";
    std::ofstream w(out_path, std::ios::binary | std::ios::trunc);
    if (!w) return false;

    std::vector<char> chunk(16 * 1024 * 1024);
    std::vector<char> zeros(chunk.size(), 0);
    uint64_t cursor = 0;
    out.coords.reserve(tensors.size());
    for (const auto& t : tensors) {
        uint64_t aligned = (cursor + alignment - 1) / alignment * alignment;
        w.seekp(static_cast<std::streamoff>(aligned));
        cursor = aligned;

        uint64_t src = data_start + t.offset;
        f.clear();
        f.seekg(static_cast<std::streamoff>(src));
        uint64_t remaining = t.size;
        while (remaining > 0) {
            size_t n = remaining > chunk.size() ? chunk.size() : static_cast<size_t>(remaining);
            if (!f.read(chunk.data(), static_cast<std::streamsize>(n))) break;
            w.write(chunk.data(), static_cast<std::streamsize>(n));
            remaining -= n;
        }
        if (remaining > 0) {
            // Short read (file truncated vs header); pad deterministically with zeros.
            while (remaining > 0) {
                size_t n = remaining > zeros.size() ? zeros.size() : static_cast<size_t>(remaining);
                w.write(zeros.data(), static_cast<std::streamsize>(n));
                remaining -= n;
            }
        }
        cursor += t.size;

        TensorCoordinate c{};
        c.absolute_offset = aligned;
        c.byte_size = t.size;
        c.quant_type = t.type;
        out.coords.emplace_back(t.name, c);
    }
    out.payload_size = cursor;
    out.payload_path = out_path.string();
    w.close();
    return out.coords.size() > 0 && out.payload_size > 0;
}

} // namespace dxait