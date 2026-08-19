#ifndef DXAIT_DXLA_HPP
#define DXAIT_DXLA_HPP

#include "dxait.hpp"
#include "dxjit.hpp"

// Guarded fallback: DXAIT_API is defined in dxait.hpp when building/using the
// shared dxait.dll. Do not redefine it here.
#if !defined(DXAIT_API)
#define DXAIT_API
#endif

namespace dxait {

enum class LAOp : int { Add = 0, Sub = 1, Mul = 2, Div = 3 };
enum class LAActivation : int { Relu = 0, Gelu = 1, Silu = 2, Tanh = 3, Sigmoid = 4, LeakyRelu = 5 };
enum class LAReduce : int { Sum = 0, Max = 1, Min = 2, Mean = 3 };

class DXAIT_API LA {
public:
    explicit LA(Device* device);
    ~LA() = default;

    void elementwise(Queue* q, Buffer* out, Buffer* in0, Buffer* in1, uint32_t count, LAOp op, float alpha = 1.0f, float beta = 0.0f);
    void activation(Queue* q, Buffer* out, Buffer* in, uint32_t count, LAActivation act, float alpha = 0.01f);
    void rmsnorm(Queue* q, Buffer* out, Buffer* in, Buffer* gamma, uint32_t rows, uint32_t dim, float eps);
    void softmax(Queue* q, Buffer* out, Buffer* in, uint32_t rows, uint32_t dim);
    void reduce(Queue* q, Buffer* out, Buffer* in, uint32_t rows, uint32_t dim, LAReduce op);
    void gemm_f16_dot2(Queue* q, Buffer* out_f32, Buffer* a_f16, Buffer* b_f16, uint32_t M, uint32_t N, uint32_t K); // dims padded to /16
    void gemm_f16_wmma(Queue* q, Buffer* out_f32, Buffer* a_f16, Buffer* b_f16, uint32_t M, uint32_t N, uint32_t K); // dispatches dot2 (WaveMatrix HLSL shelved; SM6.9/6.10 LinAlg excluded)

    const char* wmma_status() const { return m_wmma_status.c_str(); }

private:
    Device* m_device;
    PipelineCache m_pso_cache;
    ComPtr<ID3D12RootSignature> m_root_sig;
    ComPtr<ID3D12CommandAllocator> m_cmd_alloc;
    ComPtr<ID3D12GraphicsCommandList> m_cmd_list;
    std::unique_ptr<Fence> m_fence;
    uint64_t m_fence_val{0};
    std::string m_wmma_status;

    void init_root_signature();
    void ensure_idle();
};

} // namespace dxait

#endif // DXAIT_DXLA_HPP