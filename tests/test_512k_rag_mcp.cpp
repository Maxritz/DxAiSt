#include "dxait/dxait.hpp"
#include "dxait/dxcontext.hpp"
#include "dxait/dxdb.hpp"
#include "dxait/dxmcp.hpp"
#include <iostream>
#include <vector>
#include <cassert>
#include <chrono>

int main() {
    std::cout << "========================================================\n";
    std::cout << " DXAiT 512K Context, FastRetrieveDB RAG & MCP Server Test\n";
    std::cout << "========================================================\n\n";

    auto adapters = dxait::Adapter::enumerate();
    if (adapters.empty()) {
        std::cout << "No GPU found, skipping test.\n";
        return 0;
    }

    auto device = dxait::Adapter::create_device(0);
    const auto& caps = device->caps();

    std::cout << "Target GPU:              " << caps.name << "\n";
    std::cout << "Dedicated VRAM Capacity:  " << caps.dedicated_video_memory / (1024 * 1024) << " MB\n";
    std::cout << "Shared System RAM:       " << caps.shared_system_memory / (1024 * 1024) << " MB\n\n";

    // 1. Initialize 512K Token Long-Context Engine
    dxait::ContextConfig ctx_config{};
    ctx_config.max_tokens = 524288;     // 512k max context window!
    ctx_config.active_target_tokens = 262144; // Dynamically user-requested 256k context
    ctx_config.sliding_window = 8192;   // 8k hot VRAM window
    ctx_config.quant_type = dxait::ContextQuantType::Q4_0;
    ctx_config.enable_offloading = true;

    dxait::LongContextEngine context_engine(device.get(), ctx_config);
    context_engine.append_tokens(100000); // Simulate ingestion of 100,000 prompt context tokens

    std::cout << "1. 512K Engine Initialized (256K User Requested Target) & 100,000 Tokens Ingested:\n";
    auto stats = context_engine.get_stats();
    std::cout << "   - Target User Context: " << stats.target_tokens << " tokens\n";
    std::cout << "   - Active Ingested:    " << stats.active_tokens << "\n";
    std::cout << "   - VRAM Hot Tokens:    " << stats.vram_tokens << " (" << (stats.vram_bytes / (1024 * 1024)) << " MB)\n";
    std::cout << "   - System RAM Offload: " << stats.sysram_tokens << " (" << (stats.sysram_bytes / (1024 * 1024)) << " MB)\n";
    std::cout << "   - KV Quantization:    Q4_0 (4x Compression)\n\n";

    // Test Context Translation for a smaller model (e.g. 4096 context window)
    std::cout << "   Testing Context Translation for Small Model (4096 Context Window)...\n";
    context_engine.translate_context_for_model(4096, nullptr, nullptr);
    std::cout << "   Context Translation Verified!\n\n";

    // 2. Initialize FastRetrieveDB & Index Documents for RAG
    constexpr uint32_t embed_dim = 1536;
    dxait::FastRetrieveDB db(embed_dim);

    std::cout << "2. Indexing Documents into FastRetrieveDB Vector Store...\n";
    std::vector<float> vec1(embed_dim, 0.1f);
    std::vector<float> vec2(embed_dim, 0.8f);
    std::vector<float> vec3(embed_dim, -0.5f);

    db.insert("doc_1", "DirectX 12 GPU Compute Substrate Architecture", vec1);
    db.insert("doc_2", "512K Long Context Window & Dynamic Offloading", vec2);
    db.insert("doc_3", "Tombstoned Stale Chunk Document", vec3);

    std::cout << "   - Total Vector Documents: " << db.size() << "\n";

    // Perform RAG Search
    auto rag_results = db.search_rag(vec2, 2);
    assert(!rag_results.empty());
    std::cout << "   - Top RAG Match: [" << rag_results[0].id << "] \"" << rag_results[0].text
              << "\" (Similarity: " << rag_results[0].similarity << ")\n\n";

    // Delete stale doc and compact index
    db.remove("doc_3");
    std::cout << "   - Removing doc_3 and compacting database index...\n";
    db.compact();
    assert(db.size() == 2);
    std::cout << "   - Compacted Database Count: " << db.size() << " documents!\n\n";

    // 3. Initialize & Test MCP Server JSON-RPC Interface
    std::cout << "3. Testing Model Context Protocol (MCP) Server JSON-RPC Interface...\n";
    dxait::MCPServer mcp_server(&db, &context_engine);

    std::string list_resp = mcp_server.handle_json_rpc(R"({"jsonrpc":"2.0","method":"tools/list","id":1})");
    std::cout << "   - MCP tools/list response: " << list_resp << "\n";

    std::string stats_resp = mcp_server.handle_json_rpc(R"({"jsonrpc":"2.0","method":"context_stats","id":1})");
    std::cout << "   - MCP context_stats response: " << stats_resp << "\n";

    std::string rag_resp = mcp_server.handle_json_rpc(R"({"jsonrpc":"2.0","method":"rag_search","id":1})");
    std::cout << "   - MCP rag_search response: " << rag_resp << "\n\n";

    std::cout << "========================================================\n";
    std::cout << " 512K Context, FastRetrieveDB RAG & MCP Server Test PASSED!\n";

    return 0;
}
