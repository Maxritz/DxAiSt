#include "dxait/dxnetwork.hpp"
#include "dxait/dxait.hpp"
#include <iostream>
#include <vector>
#include <cassert>

int main() {
    std::cout << "========================================================\n";
    std::cout << " DXAiT Secure & Insecure DirectX Network Transport Test\n";
    std::cout << "========================================================\n\n";

    auto adapters = dxait::Adapter::enumerate();
    if (adapters.empty()) {
        std::cout << "No GPU found, skipping test.\n";
        return 0;
    }

    auto device = dxait::Adapter::create_device(0);
    const auto& caps = device->caps();

    std::cout << "Target Local GPU:        " << caps.name << "\n";
    std::cout << "Dedicated VRAM Capacity:  " << caps.dedicated_video_memory / (1024 * 1024) << " MB\n\n";

    dxait::NetworkTensorTransport net_transport(device.get(), 9090);

    // 1. Test Insecure Mode (Zero-Overhead Raw DMA Mode for Local Benchmarks)
    std::cout << "1. Testing Insecure Zero-Overhead Network Transport Mode...\n";
    net_transport.set_security_enabled(false);
    assert(!net_transport.is_security_enabled());

    bool connected_insecure = net_transport.connect_to_server("192.168.1.101", 9090);
    assert(connected_insecure);

    constexpr uint64_t tensor_bytes = 64ULL * 1024ULL * 1024ULL; // 64 MB
    auto upload_buf = device->create_buffer(tensor_bytes, dxait::MemLocation::Upload);

    bool sent_insecure = net_transport.send_tensor(1, upload_buf.get(), tensor_bytes);
    assert(sent_insecure);
    std::cout << "   Insecure Raw DMA Stream Verified!\n\n";

    // 2. Test Secure AES-256-GCM / HMAC Auth Token Mode
    std::cout << "2. Enabling AES-256 Encryption & HMAC Security Handshake Mode...\n";
    net_transport.set_security_enabled(true);
    assert(net_transport.is_security_enabled());

    dxait::NodeFeatureManifest local_manifest;
    local_manifest.node_name = caps.name;
    local_manifest.ip_address = "192.168.1.100";
    local_manifest.port = 9090;

    std::string xml = net_transport.exchange_xml_manifest(local_manifest);
    assert(xml.find("<AuthSignature>") != std::string::npos);

    bool auth_success = net_transport.authenticate_node(1, dxait::SecurityEngine::generate_auth_token("DXAiT-Cluster-Secret-2026", "RemoteServerNode"));
    assert(auth_success);

    bool sent_secure = net_transport.send_tensor(1, upload_buf.get(), tensor_bytes);
    assert(sent_secure);

    bool recv_secure = net_transport.recv_tensor(1, upload_buf.get(), tensor_bytes);
    assert(recv_secure);

    std::cout << "\n========================================================\n";
    std::cout << " Secure & Insecure DirectX Network Transport Test PASSED!\n";

    return 0;
}
