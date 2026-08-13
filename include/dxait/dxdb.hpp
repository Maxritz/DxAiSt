#ifndef DXAIT_DXDB_HPP
#define DXAIT_DXDB_HPP

#include "dxait.hpp"
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>

namespace dxait {

struct VectorDocument {
    std::string id;
    std::string text;
    std::vector<float> embedding;
    bool deleted{false};
};

struct RAGQueryResult {
    std::string id;
    std::string text;
    float similarity{0.0f};
};

class FastRetrieveDB {
public:
    explicit FastRetrieveDB(uint32_t embedding_dim = 1536);
    ~FastRetrieveDB();

    // 1. In-Memory Vector Indexing & Ingestion
    void insert(const std::string& id, const std::string& text, const std::vector<float>& embedding);

    // 2. RAG Similarity Query (Cosine / Inner Product)
    std::vector<RAGQueryResult> search_rag(const std::vector<float>& query_embedding, uint32_t top_k = 5);

    // 3. Defragmentation & Compacting (Vacuum tombstones)
    void compact();

    // 4. Index Stats
    size_t size() const;
    size_t active_count() const;
    void remove(const std::string& id);

private:
    uint32_t m_embedding_dim;
    std::vector<VectorDocument> m_docs;
    std::unordered_map<std::string, size_t> m_id_to_index;

    static float cosine_similarity(const float* a, const float* b, uint32_t dim);
    // ponytail: flat vector scan with SIMD/AVX2; upgrade path is HNSW graph index for 10M+ scaling
};

} // namespace dxait

#endif // DXAIT_DXDB_HPP
