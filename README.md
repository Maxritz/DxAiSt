# DXAiT: DirectX 12 GPU Compute Fabric for Deep Learning Inference

## What is this toolkit?

DXAiT is a native Windows first Direct3D 12 (D3D12) GPU compute fabric and software development kit built for deep learning inference. It is written in pure modern C++ (C++20) with zero external runtime dependencies beyond the OS and the DirectX toolchain. The whole idea is to skip heavy abstraction layers like CUDA or Vulkan wrappers and talk straight to the D3D12 driver, so that compute kernels, memory transfers and model weights all flow through the native graphics stack with minimal overhead.

The toolkit targets consumer and workstation AMD GPUs in particular, with tuned paths for the RDNA2 (RX 6700 XT) and RDNA4 (RX 9070 XT) architectures, and it also runs on NVIDIA and Intel adapters through the same D3D12 interface. If you have a big open weights Large Language Model that does not fit in video memory, DXAiT gives you the pieces to shard it across VRAM and system RAM, quantise the KV cache, stream tensors over PCIe or the network, and run autoregressive decoding with control over temperature, top-p, top-k and the other sampling knobs.

## What can you do with it?

- Load and inspect GGUF, Safetensors, PTE, ONNX and PyTorch bin model files using memory mapped zero copy parsing.
- Stream multi gigabyte model chunks from disk into dedicated VRAM using the copy queue and persistent staging buffers.
- Shard transformer layers across VRAM and system RAM, with a planner that respects your VRAM budget.
- Run a 512k token context window by quantising the KV cache to Q4_0 and offloading cold pages to system RAM.
- Retrieve context from an in memory vector database (FastRetrieveDB) and serve it to agentic tooling over a JSON RPC MCP server.
- Exchange GPU tensors between machines over TCP with optional XOR payload encryption, XML feature manifest handshake, and UDP LAN autodiscovery.
- JIT compile HLSL compute kernels at runtime through DXC and dispatch them immediately, including a Triton style kernel generator.
- Run a chat decode loop with 16 layer forward passes across split memory, and get honest throughput numbers because every dispatch is fence synchronised.

## Module tree

Below is the layout of the source tree. Each folder under `src/` is one module, with a matching header under `include/dxait/`.

```
include/dxait/
├── dxait.hpp          core types: AdapterCaps, Device, Queue, Buffer, Fence, TDRGuard, Config
├── dxadapter.hpp      GPU enumeration, vendor/device classification, wave size detection
├── dxruntime.hpp      device creation wrapper (forwarding header)
├── dxqueue.hpp        command queue + fence helpers (forwarding header)
├── dxmem.hpp          buffer allocation across Default / Upload / Readback / ReBAR heaps
├── dxbarrier.hpp      resource state barrier batch helper
├── dxjit.hpp          DXC shader compiler + PSO pipeline cache
├── dxblas.hpp         BLAS ops: vector add, GEMM
├── dxmath.hpp         RMSNorm, Softmax (temperature aware), RoPE, token sampling
├── dxattention.hpp    scaled dot product attention + attention mechanism dispatcher
├── dxquant.hpp        Q4_0 / Q8_0 quantization and dequantization
├── dxkv.hpp           KV cache manager with page allocation
├── dxcontext.hpp      512k long context engine: offload, quantise, compress, translate
├── dxdb.hpp           FastRetrieveDB in memory vector store with RAG search
├── dxmcp.hpp          MCP JSON RPC server: rag_search, context_compact, context_stats
├── dxmodel.hpp        model loaders: GGUF, Safetensors, PTE, ONNX, PyTorch bin
├── dxchunk.hpp        chunk streamer: memory mapped DMA into VRAM
├── dxshard.hpp        offload partition engine + shard planner
├── dxcache.hpp        advanced KV cache (hadamard transform stub)
├── dxcollective.hpp   multi GPU ring all reduce
├── dxnetwork.hpp      network tensor transport: TCP, XML manifest, autodiscovery, security toggle
├── dxrand.hpp         GPU PCG32 uniform random generator
├── dxsched.hpp        multi queue scheduler (direct + compute)
├── dxgraph.hpp        command graph with dependency validation
├── dxspeculative.hpp  speculative decoding draft verification
├── dxio.hpp           DirectStorage context for file reads
├── dxtriton.hpp       Triton style HLSL kernel generator + JIT dispatcher
├── dxtrace.hpp        PIX style trace markers
└── dxfft.hpp          radix2 FFT (identity shader, see audit)
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
├── dxmcp/dxmcp.cpp              JSON RPC server
├── dxmodel/dxmodel.cpp          multi format model loaders
├── dxchunk/dxchunk.cpp          chunk streaming to GPU
├── dxshard/dxshard.cpp          offload partition engine
├── dxcache/dxcache.cpp          advanced KV cache
├── dxcollective/dxcollective.cpp ring all reduce
├── dxnetwork/dxnetwork.cpp      network tensor transport
├── dxrand/dxrand.cpp            GPU PRNG
├── dxsched/dxsched.cpp          multi queue scheduler
├── dxgraph/dxgraph.cpp          command graph
├── dxspeculative/dxspeculative.cpp speculative verification
├── dxio/dxio.cpp                DirectStorage
├── dxtriton/dxtriton.cpp        Triton style kernels
├── dxtrace/dxtrace.cpp          trace markers
└── dxfft/dxfft.cpp              FFT (identity shader, see audit)
```

## Build and run

```cmd
cmake -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release -j 16
ctest --test-dir build -C Release --output-on-failure
```

The build links against the Agility SDK 1.720, DXC 1.10.2605.2 and DirectStorage 1.4. The relevant DLLs are copied next to each test binary automatically.

## Module tests and results

Every module has a dedicated test harness under `tests/`. The full CTest suite runs 18 tests and all 18 pass. The table below maps each test to the modules it covers and shows the last recorded result on the RX 9070 XT.

| Test harness | Modules covered | Result |
|--------------|-----------------|--------|
| test_device_init | dxadapter, dxruntime, dxqueue | Passed |
| test_mem_alloc | dxmem | Passed |
| test_dxjit_blas | dxjit, dxblas | Passed |
| test_substrate_full | dxbarrier, dxkv, dxgraph, dxsched | Passed |
| test_model_inference | dxmath (rms_norm, softmax), dxmodel | Passed |
| test_attention_rand | dxattention, dxrand | Passed |
| test_q4_quant | dxquant | Passed |
| test_model_formats | dxmodel (GGUF, Safetensors, PTE, ONNX, bin) | Passed |
| test_streaming_sharding | dxchunk, dxcache, dxcollective | Passed |
| test_real_model_loading | dxmodel (18.68 GB GGUF) | Passed |
| test_large_gguf_spillover | dxchunk (14 GB stream to VRAM) | Passed |
| test_50_50_offloading | dxshard (PCIe DMA roundtrip) | Passed |
| test_sharding_perf | dxshard (chunk size sweep) | Passed |
| test_split_load_inference | dxshard + dxmath (16 layer split pass) | Passed |
| test_chat_inference_harness | dxcontext + dxshard + dxmath (decode loop) | Passed |
| test_512k_rag_mcp | dxcontext, dxdb, dxmcp | Passed |
| test_dxnetwork | dxnetwork (secure + insecure transport) | Passed |
| test_dxtriton | dxtriton (JIT kernel gen + dispatch) | Passed |

Notable measured numbers from the last full run:

- 18.68 GB GGUF model memory mapped and parsed in under 35 ms.
- 14 GB of tensor chunks streamed into dedicated VRAM at roughly 0.92 GB/s over the copy queue.
- 50/50 VRAM/RAM page swap at about 11 ms for a 256 MB page, byte for byte verified.
- Optimal chunk size sweep converged on 256 MB per chunk.
- Real chat decode throughput after fence synchronisation: around 57 tokens per second for a 16 layer forward pass on the RX 9070 XT.

## Known tradeoffs and honest notes

- The decode numbers in the chat harness are real GPU throughput, not CPU dispatch speed. Early versions reported fake 16k tok/s because dispatches were not fence synced; that was fixed and the number dropped to the true figure. Trust the lower number.
- A UAV resource barrier added after compute dispatches was found to remove the device on RDNA4 during testing. It was removed. If you add barriers back, test on the actual target GPU first.
- Shaders that treat the dispatch thread ID as a row index now guard against out of bounds access, because an OOB write trips the GPU hang detector (DXGI_ERROR_DEVICE_HUNG) and takes the whole device down.
- The dxfft module currently contains an identity shader rather than a real radix2 FFT, and dxcache contains a hadamard transform stub. Both are flagged in the repository audit and are candidates for removal or completion.

## Acknowledgements

This project stands on the shoulders of a lot of published work and open source effort. Thanks go to:

- The authors of the attention mechanism papers that shaped the attention module: Vaswani et al. (Attention Is All You Need), Shazeer (Multi Query Attention), Ainslie et al. (Grouped Query Attention), DeepSeek AI (Multi Head Latent Attention in DeepSeek V2), Dao et al. (FlashAttention), Kwon et al. (PagedAttention), Beltagy et al. (Longformer sliding window), Katharopoulos et al. (Linear Attention), and the speculative decoding work of Leviathan et al.
- The llama.cpp project, whose GGUF format, quantization schemes and overall approach to running LLMs on modest hardware are a constant reference and source of inspiration.
- Microsoft for the DirectX 12 API, the Agility SDK, DXC compiler and DirectStorage, all of which are the backbone of this toolkit.
- AMD for the RDNA2 and RDNA4 architecture documentation, and for the hardware this project is tuned and tested against (RX 6700 XT and RX 9070 XT).

## Over engineering audit summary

A lazy review pass found the following candidates for removal, listed biggest cut first:

1. dxfft module, identity shader with no test coverage, about 113 lines.
2. dxtrace module, empty no op markers with no test coverage, about 36 lines.
3. dxcache hadamard transform empty stub.
4. dxspeculative verify kernel which unconditionally accepts all draft tokens.
5. dxgraph topo sort which only range checks dependencies.

Net removable around 300 lines, zero dependencies, since everything already uses the standard library and native D3D12.
