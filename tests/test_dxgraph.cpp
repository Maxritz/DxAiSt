#include "dxait/dxait.hpp"
#include "dxait/dxgraph.hpp"
#include <cstdio>
#include <vector>
#include <string>

int main() {
    printf("DXAiT CommandGraph Topological Sort Test\n");
    printf("==========================================\n\n");

    auto adapters = dxait::Adapter::enumerate();
    if (adapters.empty()) { printf("No GPU found, skipping test.\n"); return 0; }
    auto device = dxait::Adapter::create_device(0);

    dxait::CommandGraph graph(device.get());

    // DAG: 2 depends on 0,1; 3 depends on 2.
    uint32_t n0 = graph.add_node("node0", [](ID3D12GraphicsCommandList*) {});
    uint32_t n1 = graph.add_node("node1", [](ID3D12GraphicsCommandList*) {});
    uint32_t n2 = graph.add_node("node2", [](ID3D12GraphicsCommandList*) {}, {n0, n1});
    uint32_t n3 = graph.add_node("node3", [](ID3D12GraphicsCommandList*) {}, {n2});
    (void)n3;

    graph.compile();
    auto order = graph.execution_order();

    printf("Execution order:");
    for (uint32_t id : order) printf(" %u", id);
    printf("\n");

    // Validity: each node appears after its dependencies.
    auto pos_of = [&](uint32_t id) -> size_t {
        for (size_t i = 0; i < order.size(); ++i) if (order[i] == id) return i;
        return order.size();
    };
    bool ok = graph.is_compiled() &&
              pos_of(n0) < pos_of(n2) &&
              pos_of(n1) < pos_of(n2) &&
              pos_of(n2) < pos_of(n3);

    // Dependency validation: referencing a non-existent node must throw.
    dxait::CommandGraph bad(device.get());
    uint32_t b0 = bad.add_node("b0", [](ID3D12GraphicsCommandList*) {});
    bad.add_node("b1", [](ID3D12GraphicsCommandList*) {}, {b0});
    bad.add_node("b2", [](ID3D12GraphicsCommandList*) {}, {99u}); // invalid dep
    bool dep_threw = false;
    try { bad.compile(); } catch (const std::exception&) { dep_threw = true; }
    printf("Invalid dependency threw: %s\n", dep_threw ? "yes" : "no");

    ok = ok && dep_threw;

    printf("\nResult: %s\n", ok ? "CommandGraph PASSED" : "FAILED");
    return ok ? 0 : 1;
}
