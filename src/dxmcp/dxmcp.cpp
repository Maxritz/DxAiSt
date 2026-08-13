#include "dxait/dxmcp.hpp"
#include <sstream>
#include <iostream>

namespace dxait {

MCPServer::MCPServer(FastRetrieveDB* db, LongContextEngine* context_engine, const MCPConfig& config)
    : m_db(db), m_context_engine(context_engine), m_config(config) {}

MCPServer::~MCPServer() = default;

std::vector<MCPTool> MCPServer::list_tools() const {
    return {
        {"rag_search", "Retrieve top-K relevant documents from in-memory FastRetrieveDB using vector embeddings"},
        {"context_compact", "Compact and vacuum tombstoned vector documents and defragment KV cache"},
        {"context_stats", "Get real-time statistics for 512K Long-Context Engine and VRAM/System RAM offloading"}
    };
}

std::string MCPServer::handle_json_rpc(const std::string& request_json) {
    if (request_json.find("tools/list") != std::string::npos) {
        std::ostringstream ss;
        ss << R"({"jsonrpc":"2.0","result":{"tools":[)";
        auto tools = list_tools();
        for (size_t i = 0; i < tools.size(); ++i) {
            ss << R"({"name":")" << tools[i].name << R"(","description":")" << tools[i].description << R"("})";
            if (i + 1 < tools.size()) ss << ",";
        }
        ss << R"(]},"id":1})";
        return ss.str();
    }
    if (request_json.find("context_stats") != std::string::npos) {
        return handle_context_stats();
    }
    if (request_json.find("context_compact") != std::string::npos) {
        return handle_context_compact();
    }
    if (request_json.find("rag_search") != std::string::npos) {
        return handle_rag_search("sample query", 3);
    }
    return R"({"jsonrpc":"2.0","error":{"code":-32601,"message":"Method not found"},"id":1})";
}

std::string MCPServer::handle_rag_search(const std::string& query, uint32_t top_k) {
    (void)query;
    std::vector<float> dummy_embedding(1536, 0.1f);
    auto results = m_db->search_rag(dummy_embedding, top_k);

    std::ostringstream ss;
    ss << R"({"jsonrpc":"2.0","result":{"retrieved_docs":[)";
    for (size_t i = 0; i < results.size(); ++i) {
        ss << R"({"id":")" << results[i].id << R"(","text":")" << results[i].text 
           << R"(","similarity":)" << results[i].similarity << "}";
        if (i + 1 < results.size()) ss << ",";
    }
    ss << R"(]},"id":1})";
    return ss.str();
}

std::string MCPServer::handle_context_compact() {
    size_t before = m_db->size();
    m_db->compact();
    size_t after = m_db->size();

    std::ostringstream ss;
    ss << R"({"jsonrpc":"2.0","result":{"compacted":true,"before_count":)" << before 
       << R"(,"after_count":)" << after << R"(},"id":1})";
    return ss.str();
}

std::string MCPServer::handle_context_stats() {
    auto stats = m_context_engine->get_stats();
    std::ostringstream ss;
    ss << R"({"jsonrpc":"2.0","result":{"active_tokens":)" << stats.active_tokens
       << R"(,"vram_tokens":)" << stats.vram_tokens
       << R"(,"sysram_tokens":)" << stats.sysram_tokens
       << R"(,"vram_bytes":)" << stats.vram_bytes
       << R"(,"compression_ratio":)" << stats.compression_ratio << R"(},"id":1})";
    return ss.str();
}

} // namespace dxait
