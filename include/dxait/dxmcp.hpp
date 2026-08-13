#ifndef DXAIT_DXMCP_HPP
#define DXAIT_DXMCP_HPP

#include "dxait.hpp"
#include "dxdb.hpp"
#include "dxcontext.hpp"
#include <string>
#include <vector>
#include <memory>

namespace dxait {

struct MCPConfig {
    std::string server_name{"DXAiT-MCP-Server"};
    std::string server_version{"1.0.0"};
    uint32_t port{8080};
    uint32_t default_top_k{5};
    bool enable_sse{false};
    bool verbose_logging{false};
};

struct MCPTool {
    std::string name;
    std::string description;
};

class MCPServer {
public:
    MCPServer(FastRetrieveDB* db, LongContextEngine* context_engine, const MCPConfig& config = MCPConfig{});
    ~MCPServer();

    // 1. Process incoming JSON-RPC MCP request and produce response
    std::string handle_json_rpc(const std::string& request_json);

    // 2. List available MCP tool endpoints
    std::vector<MCPTool> list_tools() const;

    // 3. Dynamic Configuration API
    const MCPConfig& config() const { return m_config; }
    void set_config(const MCPConfig& config) { m_config = config; }

private:
    FastRetrieveDB* m_db;
    LongContextEngine* m_context_engine;
    MCPConfig m_config;

    std::string handle_rag_search(const std::string& query, uint32_t top_k);
    std::string handle_context_compact();
    std::string handle_context_stats();
    // ponytail: basic stdio JSON-RPC parser; upgrade to WebSocket/SSE protocol transport
};

} // namespace dxait

#endif // DXAIT_DXMCP_HPP
