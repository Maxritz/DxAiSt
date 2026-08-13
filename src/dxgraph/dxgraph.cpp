#include "dxait/dxgraph.hpp"
#include <stdexcept>
#include <queue>

namespace dxait {

CommandGraph::CommandGraph(Device* device) : m_device(device) {
    if (FAILED(device->get()->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&m_cmd_alloc)))) {
        throw std::runtime_error("Failed to create graph command allocator");
    }
    if (FAILED(device->get()->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, m_cmd_alloc.Get(), nullptr, IID_PPV_ARGS(&m_cmd_list)))) {
        throw std::runtime_error("Failed to create graph command list");
    }
    m_cmd_list->Close();
}

uint32_t CommandGraph::add_node(const std::string& name, NodeFunc func, const std::vector<uint32_t>& deps) {
    uint32_t id = static_cast<uint32_t>(m_nodes.size());
    m_nodes.push_back({id, name, func, deps});
    m_compiled = false;
    return id;
}

void CommandGraph::compile() {
    // Kahn's algorithm: real topological order, throws on cycles or bad deps.
    std::vector<uint32_t> in_degree(m_nodes.size(), 0);
    std::vector<std::vector<uint32_t>> adj(m_nodes.size());

    for (const auto& node : m_nodes) {
        for (uint32_t dep : node.dependencies) {
            if (dep >= m_nodes.size()) {
                throw std::runtime_error("Invalid dependency node ID");
            }
            adj[dep].push_back(node.id);
            in_degree[node.id]++;
        }
    }

    std::queue<uint32_t> ready;
    for (uint32_t i = 0; i < m_nodes.size(); ++i) {
        if (in_degree[i] == 0) ready.push(i);
    }

    m_order.clear();
    while (!ready.empty()) {
        uint32_t id = ready.front();
        ready.pop();
        m_order.push_back(id);
        for (uint32_t next : adj[id]) {
            if (--in_degree[next] == 0) ready.push(next);
        }
    }

    if (m_order.size() != m_nodes.size()) {
        throw std::runtime_error("CommandGraph contains a cycle");
    }
    m_compiled = true;
}

void CommandGraph::execute(Queue* queue) {
    if (!m_compiled) compile();

    m_cmd_alloc->Reset();
    m_cmd_list->Reset(m_cmd_alloc.Get(), nullptr);

    // Execute nodes in topological order (dependency first).
    for (uint32_t id : m_order) {
        m_nodes[id].func(m_cmd_list.Get());
    }

    m_cmd_list->Close();

    ID3D12CommandList* lists[] = { m_cmd_list.Get() };
    queue->execute(lists, 1);
}

} // namespace dxait
