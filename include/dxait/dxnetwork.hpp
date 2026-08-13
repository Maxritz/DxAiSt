#ifndef DXAIT_DXNETWORK_HPP
#define DXAIT_DXNETWORK_HPP

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>

#include "dxait.hpp"
#include <string>
#include <vector>
#include <memory>

namespace dxait {

struct NodeFeatureManifest {
    std::string node_name;
    std::string ip_address;
    uint16_t port{9090};
    uint32_t vendor_id{0};
    uint32_t device_id{0};
    uint64_t vram_capacity_mb{0};
    std::string shader_model{"6.6"};
    std::string hlsl_shader_hash{"0x9F82A4B1"};
    bool supports_wmma{false};
    bool supports_fp16{true};
    std::string auth_signature;

    std::string to_xml() const;
    static NodeFeatureManifest parse_xml(const std::string& xml_str);
};

struct NodeEndpoint {
    std::string ip_address;
    uint16_t port{9090};
    uint32_t node_id{0};
    bool is_server{false};
    bool is_authenticated{false};
    NodeFeatureManifest manifest;
};

class SecurityEngine {
public:
    static std::string generate_auth_token(const std::string& secret_key, const std::string& payload);
    static bool verify_auth_token(const std::string& secret_key, const std::string& payload, const std::string& token);
    
    // AES-256-GCM payload encryption for network tensor transfers
    static void encrypt_payload(const uint8_t* in, uint8_t* out, uint64_t bytes, const std::string& key);
    static void decrypt_payload(const uint8_t* in, uint8_t* out, uint64_t bytes, const std::string& key);
};

class NetworkTensorTransport {
public:
    NetworkTensorTransport(Device* device, uint16_t listen_port = 9090, const std::string& cluster_key = "DXAiT-Cluster-Secret-2026");
    ~NetworkTensorTransport();

    // 1. Dynamic Security Toggle (Option to run insecure for maximum zero-overhead LAN throughput)
    void set_security_enabled(bool enable_security) { m_security_enabled = enable_security; }
    bool is_security_enabled() const { return m_security_enabled; }

    // 2. Authenticated Client Connection & XML Feature Exchange
    bool connect_to_server(const std::string& server_ip, uint16_t server_port);
    std::string exchange_xml_manifest(const NodeFeatureManifest& local_manifest);
    bool authenticate_node(uint32_t node_id, const std::string& auth_token);

    // 3. Encrypted / Plaintext UDP Autodiscovery Beacon
    void broadcast_autodiscovery_beacon();
    std::vector<NodeFeatureManifest> discover_lan_peers(uint32_t timeout_ms = 1000);

    // 4. Server Listening & Connection Accept
    bool start_server_listener();

    // 5. Maximum Throughput Tensor DMA Stream (AES-256 Encrypted or Plaintext Raw DMA)
    bool send_tensor(uint32_t target_node_id, Buffer* tensor_buf, uint64_t size_bytes);
    bool recv_tensor(uint32_t source_node_id, Buffer* dest_tensor_buf, uint64_t size_bytes);
    void broadcast_tensor(Buffer* tensor_buf, uint64_t size_bytes);

    const std::vector<NodeEndpoint>& connected_nodes() const { return m_nodes; }
    bool is_server_active() const { return m_is_listening; }

private:
    Device* m_device;
    uint16_t m_port;
    std::string m_cluster_key;
    bool m_security_enabled{true};
    bool m_is_listening{false};
    std::vector<NodeEndpoint> m_nodes;

    void optimize_socket_throughput(SOCKET sock);
    // ponytail: Winsock TCP with optional AES-256 GCM encryption & HMAC auth tokens; upgrade path is IPsec / TLS 1.3 socket wrapper
};

} // namespace dxait

#endif // DXAIT_DXNETWORK_HPP
