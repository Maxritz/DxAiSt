# DXAiT: DirectX 12 High Performance GPU Compute Fabric and SDK

## Project Purpose and Vision

DXAiT is a native Windows-first Direct3D 12 (D3D12) GPU compute fabric and SDK crafted for zero-overhead deep learning inference, extreme long-context processing, and hardware-accelerated memory sharding. The primary goal of this toolkit is to eliminate heavy cross-platform abstraction layers and execute high-throughput AI workloads directly on modern graphics hardware such as AMD RDNA4 (RX 9070 XT), AMD RDNA2 (RX 6700 XT), NVIDIA RTX, and Intel Arc architectures.

By leveraging direct DirectX 12 memory management, PCIe ReBAR (Resizable BAR), zero-copy memory-mapped file access, and low-level HLSL compute shaders, DXAiT provides a ultra-fast foundation for running massive open-weights Large Language Models (LLMs) on consumer and workstation hardware.

---

## Toolkit Overview

The DXAiT architecture operates as a low-overhead GPU compute substrate. It decouples high-level model execution from driver overhead by introducing dedicated copy queue DMA pipelines, custom memory heaps, dynamic quantization, and automated GPU Timeout Detection and Recovery (TDR) blackout prevention.

```
                      +------------------------------------------+
                      |         DXAiT High-Level Client          |
                      |   (Chat Harness / RAG / MCP Server)      |
                      +--------------------+---------------------+
                                           |
                                           v
                      +--------------------+---------------------+
                      |         LongContextEngine (512K)         |
                      |   (Q4_0 Quantisation / Offloading)       |
                      +--------------------+---------------------+
                                           |
                                           v
                      +--------------------+---------------------+
                      |    FastRetrieveDB  |    MCPServer        |
                      |  (Vector RAG Store)|  (JSON-RPC API)     |
                      +--------------------+---------------------+
                                           |
                                           v
                      +--------------------+---------------------+
                      |    ChunkStreamer / ShardPlanner Engine   |
                      |  (Dynamic VRAM & System RAM Offload)     |
                      +--------------------+---------------------+
                                           |
                                           v
                      +--------------------+---------------------+
                      |   DirectX 12 Low-Level Substrate Layer   |
                      | (Buffers, Queues, Fences, HLSL Shaders)  |
                      +--------------------+---------------------+
                                           |
                                           v
                      +--------------------+---------------------+
                      |      Physical GPU Hardware (AMD / NV)    |
                      +------------------------------------------+
```

---

## Key Features

1. **512K Long-Context Management Engine**
   - Supports active context windows up to 524,288 tokens.
   - Dynamic user-configurable target context length adjustment.
   - Integrated Q4_0 4-bit and INT8 8-bit KV cache quantization for up to 4x memory savings.
   - Context window translation and downsampling for smaller model context limits (e.g. down to 4,096 tokens).

2. **In-Memory FastRetrieveDB Vector Store**
   - High-speed vector similarity indexing (Cosine Similarity) for RAG pipelines.
   - Live document tombstoning and memory defragmentation via compacting.

3. **Model Context Protocol (MCP) Server Module**
   - Standard JSON-RPC interface for AI agent integration.
   - Exposes `rag_search`, `context_compact`, and `context_stats` tools directly over stdio/SSE transports.

4. **Dynamic Sharding and 50/50 VRAM Spillover**
   - Automated allocation strategy across Dedicated GPU VRAM and Shared System RAM.
   - Zero-copy DMA page swapping over PCIe Gen4 copy queues.

5. **GPU TDR Blackout Safeguard Engine**
   - Heartbeat fence synchronization to prevent Windows WDDM driver resets during heavy GEMM dispatches.

---

## Directory Architecture

```
F:\DXAiSt\
├── CMakeLists.txt              # CMake build manifest and configuration
├── AGENTS.md                   # Development guidelines and standards
├── README.md                   # Toolkit documentation
├── include\
│   └── dxait\
│       ├── dxait.hpp           # Core D3D12 device, buffer, queue, and TDRGuard declarations
│       ├── dxadapter.hpp       # GPU architecture enumeration and feature detection
│       ├── dxattention.hpp     # FlashAttention and scaled dot-product attention compute shaders
│       ├── dxblas.hpp          # Basic Linear Algebra Subprograms (GEMM, Vector Add)
│       ├── dxcache.hpp         # Pipeline State Object (PSO) and shader compilation cache
│       ├── dxchunk.hpp         # Memory-mapped file streaming and double-buffered DMA chunking
│       ├── dxcollective.hpp    # Multi-GPU tensor reduction and memory synchronization
│       ├── dxcontext.hpp       # 512K Long-Context Engine and KV cache quantization
│       ├── dxdb.hpp            # FastRetrieveDB in-memory vector store and RAG engine
│       ├── dxfft.hpp           # Fast Fourier Transform GPU compute shaders
│       ├── dxio.hpp            # Asynchronous file I/O and DirectStorage 1.4 integration
│       ├── dxjit.hpp           # Just-In-Time HLSL shader compilation pipeline
│       ├── dxkv.hpp            # Paged Attention KV cache manager
│       ├── dxmath.hpp          # Element-wise operations (RMSNorm, Softmax, GELU, RoPE)
│       ├── dxmcp.hpp           # Model Context Protocol (MCP) JSON-RPC server module
│       ├── dxmem.hpp           # Memory allocator for Default, Upload, and ReBAR heaps
│       ├── dxmodel.hpp         # GGUF and SafeTensors model file parser
│       ├── dxquant.hpp         # Quantization and dequantization kernels (Q4_0, Q8_0)
│       ├── dxsched.hpp         # Async execution graph scheduler
│       ├── dxshard.hpp         # Multi-GPU and System RAM offloading sharder
│       └── dxspeculative.hpp   # Speculative decoding and draft model verification
├── src\                        # Implementation source files matching header modules
└── tests\                      # Unit tests and performance benchmarks (16 validation suites)
```

---

## Test Suite and Verification

Building and executing all 16 test validation suites:

```cmd
cmake -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release -j 16
ctest --test-dir build -C Release --output-on-failure
```

### Included Verification Harnesses

| Test Harness | Coverage |
|--------------|----------|
| `test_device_init` | Direct3D 12 device creation and GPU architecture classification |
| `test_mem_alloc` | VRAM, Upload, Readback, and ReBAR heap allocation |
| `test_dxjit_blas` | JIT HLSL compilation and GEMM vector compute execution |
| `test_large_gguf_spillover` | Zero-copy 18.68 GB GGUF model memory-mapping and 14 GB streaming |
| `test_50_50_offloading` | Split-load memory allocation (50% Dedicated VRAM / 50% System RAM) |
| `test_sharding_perf` | Chunk size sweep benchmark identifying optimal 256 MB DMA granularity |
| `test_split_load_inference` | 16-layer transformer forward pass across split VRAM and System RAM |
| `test_chat_inference_harness` | Multi-turn chat token decode loop with interactive latency tracking |
| `test_512k_rag_mcp` | 512K context ingestion, FastRetrieveDB RAG search, and MCP JSON-RPC server |

---

## What DXAiT Can Help With

- **Running Oversized Models on Single GPUs**: Offloads layers seamlessly into System RAM with minimal latency overhead.
- **Extreme Long-Context Processing**: Handles up to 512,000 token KV caches using Q4_0 quantization and sliding window VRAM paging.
- **Embedded RAG Applications**: Provides sub-millisecond vector similarity search directly inside C++ application memory without external database dependencies.
- **Agentic Integration**: Offers native Model Context Protocol (MCP) endpoints for seamless connection to agentic software ecosystems.
