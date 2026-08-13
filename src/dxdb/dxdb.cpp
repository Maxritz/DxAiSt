#include "dxait/dxdb.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace dxait {

FastRetrieveDB::FastRetrieveDB(uint32_t embedding_dim) : m_embedding_dim(embedding_dim) {}

FastRetrieveDB::~FastRetrieveDB() = default;

float FastRetrieveDB::cosine_similarity(const float* a, const float* b, uint32_t dim) {
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (uint32_t i = 0; i < dim; ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    float denom = std::sqrt(norm_a) * std::sqrt(norm_b);
    return (denom > 1e-8f) ? (dot / denom) : 0.0f;
}

void FastRetrieveDB::insert(const std::string& id, const std::string& text, const std::vector<float>& embedding) {
    if (m_id_to_index.find(id) != m_id_to_index.end()) {
        size_t idx = m_id_to_index[id];
        m_docs[idx] = {id, text, embedding, false};
        return;
    }
    size_t new_idx = m_docs.size();
    m_docs.push_back({id, text, embedding, false});
    m_id_to_index[id] = new_idx;
}

void FastRetrieveDB::remove(const std::string& id) {
    auto it = m_id_to_index.find(id);
    if (it != m_id_to_index.end()) {
        m_docs[it->second].deleted = true;
    }
}

std::vector<RAGQueryResult> FastRetrieveDB::search_rag(const std::vector<float>& query_embedding, uint32_t top_k) {
    std::vector<RAGQueryResult> results;
    if (query_embedding.size() != m_embedding_dim || m_docs.empty()) return results;

    for (const auto& doc : m_docs) {
        if (doc.deleted || doc.embedding.size() != m_embedding_dim) continue;
        float sim = cosine_similarity(query_embedding.data(), doc.embedding.data(), m_embedding_dim);
        results.push_back({doc.id, doc.text, sim});
    }

    std::partial_sort(results.begin(),
                      results.begin() + std::min<size_t>(top_k, results.size()),
                      results.end(),
                      [](const RAGQueryResult& a, const RAGQueryResult& b) {
                          return a.similarity > b.similarity;
                      });

    if (results.size() > top_k) {
        results.resize(top_k);
    }
    return results;
}

void FastRetrieveDB::compact() {
    std::vector<VectorDocument> active_docs;
    m_id_to_index.clear();

    for (const auto& doc : m_docs) {
        if (!doc.deleted) {
            size_t new_idx = active_docs.size();
            m_id_to_index[doc.id] = new_idx;
            active_docs.push_back(doc);
        }
    }
    m_docs = std::move(active_docs);
}

size_t FastRetrieveDB::size() const {
    return m_docs.size();
}

size_t FastRetrieveDB::active_count() const {
    size_t count = 0;
    for (const auto& doc : m_docs) {
        if (!doc.deleted) count++;
    }
    return count;
}

} // namespace dxait
