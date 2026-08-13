#ifndef DXAIT_DXGRAPH_HPP
#define DXAIT_DXGRAPH_HPP

#include "dxait.hpp"
#include <functional>
#include <vector>
#include <memory>

namespace dxait {

using NodeFunc = std::function<void(ID3D12GraphicsCommandList*)>;

struct GraphNode {
    uint32_t id;
    std::string name;
    NodeFunc func;
    std::vector<uint32_t> dependencies;
};

class CommandGraph {
public:
    explicit CommandGraph(Device* device);
    ~CommandGraph() = default;

    uint32_t add_node(const std::string& name, NodeFunc func, const std::vector<uint32_t>& deps = {});
    void compile();
    void execute(Queue* queue);

private:
    Device* m_device;
    std::vector<GraphNode> m_nodes;
    ComPtr<ID3D12CommandAllocator> m_cmd_alloc;
    ComPtr<ID3D12GraphicsCommandList> m_cmd_list;
    bool m_compiled{false};
};

} // namespace dxait

#endif // DXAIT_DXGRAPH_HPP
