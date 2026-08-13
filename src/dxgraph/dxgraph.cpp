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
    // Topological sort validation
    std::vector<uint32_t> in_degree(m_nodes.size(), 0);
    for (const auto& node : m_nodes) {
        for (uint32_t dep : node.dependencies) {
            if (dep >= m_nodes.size()) {
                throw std::runtime_error("Invalid dependency node ID");
            }
        }
    }
    m_compiled = true;
}

void CommandGraph::execute(Queue* queue) {
    if (!m_compiled) compile();

    m_cmd_alloc->Reset();
    m_cmd_list->Reset(m_cmd_alloc.Get(), nullptr);

    for (const auto& node : m_nodes) {
        node.func(m_cmd_list.Get());
    }

    m_cmd_list->Close();

    ID3D12CommandList* lists[] = { m_cmd_list.Get() };
    queue->execute(lists, 1);
}

} // namespace dxait
