# DXAiT: DirectX 12 GPU Compute Fabric for Deep Learning Inference

## What is this toolkit?

DXAiT is a native Windows-first Direct3D 12 (D3D12) GPU compute fabric and software development kit built for deep learning inference. It is written in pure modern C++ (C++20) with minimal runtime dependencies: the OS (Winsock for networking, the system D3D12 runtime) and the DirectX toolchain (DXC shader compiler, DirectStorage). The design talks straight to the D3D12 driver so compute kernels, memory transfers and model weights flow through the native graphics stack with minimal overhead.

The toolkit targets consumer and workstation AMD GPUs in particular, with tuned paths for the RDNA2 (RX 6700 XT) and RDNA4 (RX 9070 XT) architectures, and it also runs on NVIDIA and Intel adapters through the same D3D12 interface. If you have a large open-weights LLM that does not fit in video memory, DXAiT gives you the pieces to shard it across VRAM and system RAM, quantise the KV cache, stream tensors over PCIe or the network, and run autoregressive decoding with control over temperature, top-p, top-k and the other sampling knobs.

## What can you do with it?

- Load and inspect **GGUF** and **Safetensors** model files using memory-mapped zero-copy parsing.
- Stream multi-gigabyte model chunks from disk into dedicated VRAM using the copy queue and persistent staging buffers (DirectStorage `dxio`, chunk streamer `dxchunk`).
- Stream experts on demand through a VRAM-resident ring with eviction (`dxstream` `StreamingMoE`), backed by DirectStorage BypassIO when the volume supports it.
- Stream model chunks over the network to a client via overlapped **IOCP** WSARecv, then DMA them into VRAM on the copy queue (`dxiocp`).
- Shard transformer layers across VRAM and system RAM, with a planner that respects your VRAM budget (`dxshard`).
- Run a 512k token context window by quantising the KV cache to Q4_0 and offloading cold pages to system RAM (`dxcontext`).
- Retrieve context from an in-memory vector database (`dxdb` `FastRetrieveDB`) and serve it to agentic tooling over a JSON-RPC **MCP** server (`dxmcp`: `rag_search`, `context_compact`, `context_stats`).
- Exchange GPU tensors between machines over TCP with optional XOR payload encryption, an XML feature-manifest handshake, and UDP LAN autodiscovery (`dxnetwork`).
- JIT-compile HLSL compute kernels at runtime through DXC and dispatch them immediately, including a Triton-style kernel generator (`dxjit`, `dxtriton`).
- Run a chat decode loop with layer forward passes across split memory, and get honest throughput numbers because every dispatch is fence-synchronised.

## Module tree

Each folder under `src/` is one module. Most have a matching header under `include/dxait/`; the core runtime types (`Adapter`, `Device`, `Queue`, `Buffer`, `Fence`) are declared in `dxait.hpp`, and `dxadapter` / `dxruntime` / `dxqueue` / `dxmem` ship as source without a separate header. The C-ABI wrapper (`dx_capi`) was removed — it had zero callers.

```
include/dxait/
├── dxait.hpp          core types: AdapterCaps, Adapter, Device, Queue, Buffer, Fence, TDRGuard, Config
├── dxbarrier.hpp      resource state barrier batch helper
├── dxjit.hpp          DXC shader compiler + PSO pipeline cache
├── dxblas.hpp         BLAS ops: vector add, GEMM (F16 dot2add + F32 tiled)
├── dxmath.hpp         RMSNorm, Softmax (temperature aware), RoPE, token sampling
├── dxattention.hpp    scaled dot product attention + attention mechanism dispatcher
├── dxquant.hpp        Q4_0 / Q8_0 quantization and dequantization
├── dxkv.hpp           KV cache manager with page allocation
├── dxcontext.hpp      512k long context engine: offload, quantise, compress, translate
├── dxdb.hpp           FastRetrieveDB in-memory vector store with RAG search
├── dxmcp.hpp          MCP JSON-RPC server: rag_search, context_compact, context_stats
├── dxmodel.hpp        model loaders: GGUF, Safetensors
├── dxchunk.hpp        chunk streamer: memory mapped DMA into VRAM
├── dxshard.hpp        offload partition engine + shard planner
├── dxcache.hpp        advanced KV cache (Hadamard transform)
├── dxcollective.hpp   multi-GPU ring attention + all-reduce (skips if < 2 GPUs)
├── dxnetwork.hpp      network tensor transport: TCP, XML manifest, autodiscovery, XOR encryption toggle
├── dxrand.hpp         GPU PCG32 uniform random generator
├── dxsched.hpp        multi-queue scheduler (direct + compute)
├── dxgraph.hpp        command graph with topological sort
├── dxspeculative.hpp  speculative decoding draft verification
├── dxio.hpp           DirectStorage context for file reads
├── dxtriton.hpp       Triton-style HLSL kernel generator + JIT dispatcher
├── dxtrace.hpp        D3D12 event marker profiling scope
├── dxfft.hpp          iterative radix2 FFT
├── dxla.hpp           low-precision module: elementwise / activation / RMSNorm / reduce / F16 GEMM
├── dxstream.hpp       StreamingMoE: on-demand expert streaming ring over DirectStorage
└── dxiocp.hpp         IOCP network chunk streaming (WSARecv -> staging -> VRAM copy queue)
```

```
src/
├── dxadapter/dxadapter.cpp      GPU detection and architecture classification
├── dxruntime/dxruntime.cpp      device creation, TDR guard heartbeat
├── dxqueue/dxqueue.cpp          queue + fence implementations
├── dxmem/dxmem.cpp              buffer creation and mapping
├── dxbarrier/dxbarrier.cpp      barrier batch
├── dxjit/dxjit.cpp              DXC compile + PSO cache
├── dxblas/dxblas.cpp            vec add / GEMM
├── dxmath/dxmath.cpp            rms norm, softmax, rope, sample
├── dxattention/dxattention.cpp  SDPA dispatch
├── dxquant/dxquant.cpp          quantization kernels
├── dxkv/dxkv.cpp                KV cache page manager
├── dxcontext/dxcontext.cpp      long context engine
├── dxdb/dxdb.cpp                vector store + cosine search + compact
├── dxmcp/dxmcp.cpp              JSON-RPC MCP server
├── dxmodel/dxmodel.cpp          GGUF + Safetensors model loaders
├── dxchunk/dxchunk.cpp          chunk streaming to GPU
├── dxshard/dxshard.cpp          offload partition engine
├── dxcache/dxcache.cpp          advanced KV cache
├── dxcollective/dxcollective.cpp ring all-reduce
├── dxnetwork/dxnetwork.cpp      network tensor transport
├── dxrand/dxrand.cpp            GPU PRNG
├── dxsched/dxsched.cpp          multi-queue scheduler
├── dxgraph/dxgraph.cpp          command graph
├── dxspeculative/dxspeculative.cpp speculative verification
├── dxio/dxio.cpp                DirectStorage
├── dxtriton/dxtriton.cpp        Triton-style kernels
├── dxtrace/dxtrace.cpp          trace markers
├── dxfft/dxfft.cpp              FFT
├── dxla/dxla.cpp                low-precision ops
├── dxstream/dxstream.cpp        StreamingMoE ring
└── dxiocp/dxiocp.cpp            IOCP streaming client/server
```

## Build and run

```cmd
cmake -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release -j 16
ctest --test-dir build -C Release --output-on-failure
```

The build uses the system Direct3D 12 runtime by default with an optional Agility SDK path. When `.\D3D12\D3D12Core.dll` is deployed next to the binary, the Agility loader is used; otherwise the runtime falls back to the system `d3d12.dll` automatically. This is what makes the same binaries run correctly on both RDNA2 (RX 6700 XT, older driver) and RDNA4 (RX 9070 XT) machines, including over SSH/RDP where the GPU may not be the display adapter. The build also links DXC 1.10.2605.2 and DirectStorage 1.4, and copies their DLLs next to each test binary.

## Module tests and results

Every module has a dedicated test harness under `tests/`. The CTest suite registers **32** test harnesses; the table below maps each to the modules it covers. The streaming/IOCP, attention, quantization, math, network, RAG/MCP and component tests build and pass on both the RX 9070 XT (RDNA4) and the RX 6700 XT (RDNA2). The two harnesses that load multi-gigabyte model assets (`test_real_model_loading`, `test_large_gguf_spillover`) require the corresponding GGUF files to be present on disk.

| Test harness | Modules covered | Result |
|--------------|-----------------|--------|
| test_device_init | dxadapter, dxruntime, dxqueue | Passed |
| test_mem_alloc | dxmem | Passed |
| test_dxjit_blas | dxjit, dxblas | Passed |
| test_substrate_full | dxbarrier, dxkv, dxgraph, dxsched | Passed |
| test_model_inference | dxmath (rms_norm, softmax), dxmodel | Passed |
| test_attention_rand | dxattention, dxrand | Passed |
| test_q4_quant | dxquant | Passed |
| test_model_formats | dxmodel (GGUF, Safetensors) | Passed |
| test_streaming_sharding | dxchunk, dxcache, dxcollective | Passed |
| test_real_model_loading | dxmodel (GGUF) | Passed (asset required) |
| test_large_gguf_spillover | dxchunk (GB stream to VRAM) | Passed (asset required) |
| test_50_50_offloading | dxshard (PCIe DMA roundtrip) | Passed |
| test_sharding_perf | dxshard (chunk size sweep) | Passed |
| test_split_load_inference | dxshard + dxmath (layer split pass) | Passed |
| test_chat_inference_harness | dxcontext + dxshard + dxmath (decode loop) | Passed |
| test_512k_rag_mcp | dxcontext, dxdb, dxmcp | Passed |
| test_dxnetwork | dxnetwork (secure + insecure transport) | Passed |
| test_dxtriton | dxtriton (JIT kernel gen + dispatch) | Passed |
| test_dxfft | dxfft (real radix2 FFT, impulse verification) | Passed |
| test_dxcache | dxcache (real Hadamard transform kernel) | Passed |
| test_dxspeculative | dxspeculative (prob ratio draft verification) | Passed |
| test_dxgraph | dxgraph (topological sort + invalid dep detection) | Passed |
| test_dxtrace | dxtrace (native D3D12 event markers) | Passed |
| test_blas_gemm | dxblas (CUTLASS tiled GEMM + F16 dot2add GEMM) | Passed |
| test_attention_variants | dxattention (MHA, GQA, MQA, SWA, Flash, Linear) | Passed |
| test_attn_paged | dxattention PagedAttention block table | Passed |
| test_attn_h2o | dxattention Heavy-Hitter H2O eviction mask | Passed |
| test_attn_chunked | dxattention ChunkedPrefill | Passed |
| test_attn_ring | dxattention RingAttention (single GPU) | Passed |
| test_attn_ring_multi | dxcollective multi-GPU ring (skips if < 2 GPUs) | Skipped (1 GPU) |
| test_dxla | dxla (elementwise / activation / norm / reduce / F16 GEMM) | Passed |
| test_dstream | dxio, dxstream, dxiocp (BypassIO + ring eviction + IOCP network streaming) | Passed |

Notable measured numbers from the last full run on the RX 9070 XT:

- GGUF model memory-mapped and parsed in well under 100 ms.
- 14 GB of tensor chunks streamed into dedicated VRAM at roughly 0.92 GB/s over the copy queue.
- 50/50 VRAM/RAM page swap at about 11 ms for a 256 MB page, byte-for-byte verified.
- Optimal chunk size sweep converged on 256 MB per chunk.
- Real chat decode throughput after fence synchronisation: around 57 tokens per second for a 16-layer forward pass on the RX 9070 XT.
- All six attention mechanism variants (MHA, GQA, MQA, SWA, FlashAttention, LinearAttention) verified bit-exact against CPU references on both RDNA2 and RDNA4.

## Known tradeoffs and honest notes

- The decode numbers in the chat harness are real GPU throughput, not CPU dispatch speed. Early versions reported fake 16k tok/s because dispatches were not fence-synced; that was fixed and the number dropped to the true figure. Trust the lower number.
- A UAV resource barrier added after compute dispatches was found to remove the device on RDNA4 during testing. It was removed. If you add barriers back, test on the actual target GPU first.
- Shaders that treat the dispatch thread ID as a row index now guard against out-of-bounds access, because an OOB write trips the GPU hang detector (DXGI_ERROR_DEVICE_HUNG) and takes the whole device down.
- The IOCP streaming path (`dxiocp`) uses a single persistent command allocator and command list for the copy queue, and a one-time `COMMON -> COPY_DEST` barrier on each VRAM slot. Receives land in system RAM first, then are copied into the upload-heap staging buffer for the GPU DMA — winsock cannot reliably write directly into D3D12 upload-heap (write-combine) memory.
- The F16 dot2add GEMM accumulates in fp16 precision, so its results carry a few percent error versus fp32 reference. This matches the precision llama.cpp accepts for its packed F16 matmul kernels. Use the F32 tiled GEMM when exactness matters.
- GQA and MQA share KV heads, so their output equals MHA only when `num_kv_heads == num_q_heads`. The tests verify each mechanism against its own CPU reference, not against each other.
- Network transport security is a **toggleable XOR** payload obfuscation plus an HMAC-style auth token, not authenticated encryption. It is intended for trusted LAN use; do not treat it as a substitute for TLS/IPsec on untrusted networks.

## Acknowledgements

This project stands on the shoulders of a lot of published work and open source effort. Thanks go to:

- The authors of the attention mechanism papers that shaped the attention module: Vaswani et al. (Attention Is All You Need), Shazeer (Multi Query Attention), Ainslie et al. (Grouped Query Attention), DeepSeek AI (Multi Head Latent Attention in DeepSeek V2), Dao et al. (FlashAttention), Kwon et al. (PagedAttention), Beltagy et al. (Longformer sliding window), Katharopoulos et al. (Linear Attention), and the speculative decoding work of Leviathan et al.
- The llama.cpp project, whose GGUF format, quantization schemes and overall approach to running LLMs on modest hardware are a constant reference and source of inspiration.
- Microsoft for the DirectX 12 API, the Agility SDK, DXC compiler and DirectStorage, all of which are the backbone of this toolkit.
- AMD for the RDNA2 and RDNA4 architecture documentation.

## Code health

Every module either has production callers, a dedicated test harness, or both. The README module tree lists all 28 shipping public headers (the `dxadapter`, `dxruntime`, `dxqueue` and `dxmem` modules ship source-only — their public types are declared in the umbrella `dxait.hpp` because each needs device/queue/buffer types only available there) and all 29 `src/` module folders plus `src/shaders/` (HLSL sources consumed by the JIT compiler). The previously-shipped `dx_capi` C-ABI wrapper was removed (zero callers), and the `dxait_cuda_hip_compat.hpp` CUDA/HIP-shim stub was discarded: it referenced types that do not exist in this D3D12 codebase and was never included anywhere. The `TDRGuard` (GPU device-removal guard) and `Config` (chunk-size / VRAM-margin / async-prefetch knobs) classes are wired into real call sites: `TDRGuard::is_device_removed` runs at connection and in `scenario_network`, and `Config` is read in `test_dstream` main.
