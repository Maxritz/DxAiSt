# DXAiT — DirectX AI Compute & Execution Fabric
## Master Architecture & Functional Specification

**Status:** Architecture specification  
**Target:** Windows-first DirectX 12 / D3D12 compute platform  
**Primary purpose:** A modular GPU compute SDK for AI, scientific compute, inference, media, and heterogeneous GPU workloads  
**Design inspiration:** CUDA Runtime/Driver APIs, ROCm/HIP/rocBLAS/rocSPARSE/rocFFT/rocRAND, Vulkan compute, DirectStorage, PIX/GPU tracing  
**Implementation language:** Pure Modern C++ (C++20/C++23 API and implementation; `namespace dxait`)  
**GPU scope:** RDNA-era AMD GPUs and any D3D12-capable hardware implementing the required feature set  
**Operating model:** Local GPU, multi-GPU, multi-server, heterogeneous memory, asynchronous execution  
**Core principle:** Modules describe work; the runtime, graph, scheduler, memory manager, and synchronization system decide where and when that work executes.
**Vulkan 1.4 API Mapping:** See [`docs/API_Mapping_Vulkan_D3D12.md`](docs/API_Mapping_Vulkan_D3D12.md) for Vulkan 1.4 ↔ D3D12 extension equivalences and hardware optimizations.

---

# 1. Executive Summary

DXAiT is a DirectX-native GPU compute toolkit designed to provide a modular execution substrate comparable in scope to CUDA/ROCm while remaining native to Windows and D3D12.

The SDK is deliberately divided into two layers:

1. **Execution substrate**
   - device discovery
   - queues
   - command recording
   - fences
   - barriers
   - memory allocation
   - residency
   - DirectStorage
   - graph scheduling
   - tracing
   - multi-GPU execution
   - multi-server execution

2. **Compute modules**
   - BLAS
   - quantization
   - math
   - FFT
   - RNG
   - convolution
   - sparse operations
   - attention
   - KV cache
   - model loading
   - JIT/runtime shader compilation
   - future specialized kernels

The substrate must not depend on any particular AI model.

A model runtime should be able to use:

```text
dxruntime
dxmem
dxqueue
dxfence
dxgraph
dxtrace
dxblas
dxmath
dxquant
dxmodel
```

without knowing whether execution occurs on:

```text
GPU 0
GPU 1
a different local GPU
a remote GPU
multiple GPUs
multiple servers
VRAM
system RAM
NVMe
```

The runtime hides those decisions behind explicit handles and execution graphs.

---

# 1.1 Implementation Status (read this first)

This document is a **forward-looking architecture specification**. The following describes the
complete intended fabric. As of the current tree, only a subset is actually implemented and
shipped in `src/`. To avoid confusion between spec and reality:

**Implemented modules (present in `src/`, each with a header under `include/dxait/` and a test harness):**

```
dxadapter  dxruntime  dxqueue    dxmem     dxbarrier  dxio      dxgraph
dxsched    dxtrace    dxjit      dxblas    dxmath     dxattention dxquant
dxkv       dxcontext  dxdb       dxmcp     dxmodel    dxchunk   dxshard
dxcache    dxcollective dxnetwork dxrand   dxfft      dxspeculative dxtriton
dxla       dxstream   dxiocp
```

**Model loaders implemented:** GGUF and Safetensors only (memory-mapped). The earlier
PTE / ONNX / PyTorch-bin parsers were removed (they were stubs with no torch dependency).

**NOT yet implemented (described below as forward scope, not present in the tree):**

- `dxpeer`, `dxcluster` — local GPU-to-GPU and multi-server fabric (Sections 29, 32–39, 42–44).
- `dxconv` — convolution module (Sections 51, 654+).
- `dxsparse` — sparse module (Sections 52, 672+).
- Distributed collectives beyond single-GPU ring all-reduce (`dxcollective` exists but
  multi-GPU paths skip when only one GPU is present).
- Tracing (`dxtrace`) is implemented as native D3D12 event markers; the full telemetry
  timeline / stall-classifier described in Sections 25–28 is not built out.

Everything else (execution substrate, BLAS/math/quant/attention/KV/model loading/JIT,
DirectStorage streaming, IOCP network streaming, sharding, long-context, RAG+MCP) is real.

---

# 2. Design Goals

## 2.1 Primary Goals

DXAiT MUST:

- be native to Windows;
- use D3D12 as the primary GPU execution API;
- expose a stable C99-compatible public ABI;
- support asynchronous GPU execution;
- support DIRECT, COMPUTE, and COPY queues;
- support GPU timeline synchronization;
- avoid CPU-side waits in hot paths;
- support GPU/CPU memory hierarchy management;
- support VRAM ↔ system RAM movement;
- support lazy model/tensor loading;
- integrate DirectStorage for NVMe-backed data;
- support multi-GPU systems;
- support heterogeneous GPUs;
- support multi-server execution;
- expose GPU and CPU trace information;
- provide explicit queue and synchronization diagnostics;
- allow modules to be used independently;
- allow modules to be assembled into execution graphs;
- allow a single command graph to span multiple queues and devices;
- permit remote execution without exposing native D3D12 handles over the network.

## 2.2 Secondary Goals

The SDK SHOULD:

- minimize CPU submission overhead;
- minimize descriptor allocation in hot paths;
- minimize redundant barriers;
- minimize synchronization;
- reuse pipelines and command allocators;
- exploit wave/subgroup functionality where available;
- support FP16/BF16/INT8/FP8/MX formats where hardware permits;
- support indirect execution;
- support asynchronous copies;
- support shader specialization;
- support runtime kernel selection;
- expose hardware capability information;
- permit architecture-specific optimized kernels without making them mandatory;
- provide deterministic execution options where practical.

---

# 3. Non-Goals

DXAiT is not intended to:

- replace D3D12;
- replace DirectStorage;
- replace PIX;
- require CUDA;
- require ROCm;
- require a specific AI framework;
- force a particular model architecture;
- require Linux;
- expose D3D12 objects across network boundaries;
- make every operation automatically distributed.

External libraries MAY be used by optional bridge modules, but the core execution path must not depend on CUDA or ROCm.

---

# 4. Architectural Model

```text
                         DXAiT
                           |
                +----------+----------+
                |                     |
          EXECUTION FABRIC       COMPUTE MODULES
                |                     |
       +--------+--------+      +-----+-----+
       |        |        |      |     |     |
     Queue    Memory   Graph   BLAS Math Quant
       |        |        |      FFT  RNG  Conv
       |        |        |      Attn KV   Sparse
       |        |        |      Model JIT ...
       |        |
       +--------+----------------+
                |
          Trace / Telemetry
                |
       +--------+---------+
       |                  |
     LOCAL             REMOTE
     GPUs              SERVERS
```

The execution fabric is the foundation.

Compute modules must not independently reinvent:

- queue creation;
- fence management;
- descriptor management;
- memory allocation;
- residency;
- tracing;
- device discovery.

---

# 5. Repository Layout

Recommended repository (current state — only these modules exist; see §1.1 for the
implemented list and the forward-scope modules that are not yet present):

```text
DXAiT/
├── include/dxait/        one header per module (dxait.hpp is the core umbrella)
│
├── src/                  one folder per module (matches include/dxait headers)
│   ├── dxadapter/ dxruntime/ dxqueue/ dxmem/ dxbarrier/ dxio/
│   ├── dxgraph/ dxsched/ dxtrace/ dxjit/ dxblas/ dxmath/
│   ├── dxattention/ dxquant/ dxkv/ dxcontext/ dxdb/ dxmcp/
│   ├── dxmodel/ dxchunk/ dxshard/ dxcache/ dxcollective/
│   ├── dxnetwork/ dxrand/ dxfft/ dxspeculative/ dxtriton/
│   ├── dxla/ dxstream/ dxiocp/
│   └── shaders/          HLSL sources for the compute kernels
│
├── tools/                dxinspect, dxbench, dxai_ingest
└── tests/                one harness per module (see README test matrix)
```

Note: there is no `dxfence` module (fences live in `dxait.hpp` / `dxqueue`), and the
aspirational `dxpeer`, `dxcluster`, `dxconv`, `dxsparse`, `dxshaderc`, `dxmodelinfo`,
`specs/`, `examples/`, `third_party/` entries from earlier drafts are not present.

---

# 6. Handle Model

Public APIs must use opaque handles.

Examples:

```c
typedef struct DXRuntime DXRuntime;
typedef struct DXDevice DXDevice;
typedef struct DXQueue DXQueue;
typedef struct DXBuffer DXBuffer;
typedef struct DXFence DXFence;
typedef struct DXGraph DXGraph;
typedef struct DXGraphNode DXGraphNode;
typedef struct DXTrace DXTrace;
typedef struct DXCluster DXCluster;
```

Applications MUST NOT rely on internal structure layouts.

Native D3D12 objects may be retrieved through explicit interop functions:

```c
HRESULT dxdevice_get_native(
    DXDevice *device,
    ID3D12Device **out_device);
```

Interop MUST be explicit.

---

# 7. Device Discovery

`dxadapter` is responsible for enumerating physical adapters.

Required information:

```text
adapter name
vendor ID
device ID
subsystem ID
revision
VRAM
shared system memory
BAR/resource visibility
feature level
shader model
wave support
FP16
INT8
BF16
FP8 where available
ray tracing capability
mesh shader capability
indirect execution capability
timestamp capability
queue capabilities
memory budget
UMA/non-UMA classification
```

Example:

```c
typedef struct DXAdapterCaps {
    char name[256];

    uint32_t vendor_id;
    uint32_t device_id;

    uint64_t dedicated_video_memory;
    uint64_t shared_system_memory;

    uint32_t feature_level;
    uint32_t shader_model;

    uint32_t wave_min;
    uint32_t wave_max;

    bool wave_ops;
    bool fp16;
    bool bf16;
    bool int8;
    bool fp8;

    bool execute_indirect;
    bool enhanced_barriers;
    bool direct_storage;
} DXAdapterCaps;
```

---

# 8. Queue Architecture

DXAiT MUST support three logical queue classes:

```text
DX_QUEUE_DIRECT
DX_QUEUE_COMPUTE
DX_QUEUE_COPY
```

The implementation may expose multiple physical queues of each class.

## 8.1 Queue Selection

Modules may request:

```c
DX_QUEUE_AUTO
DX_QUEUE_DIRECT
DX_QUEUE_COMPUTE
DX_QUEUE_COPY
```

`DX_QUEUE_AUTO` delegates placement to `dxsched`.

## 8.2 Queue Policy

```c
typedef struct DXQueuePolicy {
    DXQueueType preference;

    bool allow_migration;
    bool allow_direct_fallback;
    bool allow_compute_fallback;

    uint32_t priority;
} DXQueuePolicy;
```

## 8.3 Queue Cost Model

The scheduler should consider:

```text
queue availability
dependency cost
estimated execution time
synchronization cost
resource state transitions
queue switch cost
memory locality
device locality
network transfer cost
current utilization
```

The scheduler MUST avoid excessive queue migration.

---

# 9. GPU Synchronization

`dxfence` is a foundational subsystem.

The runtime MUST favor GPU-side synchronization over CPU waits.

## 9.1 Timeline Fence

Each queue SHOULD maintain a monotonically increasing timeline value.

```text
COMPUTE
signal 100
signal 101
signal 102

COPY
signal 200
signal 201
```

A dependent queue can wait directly:

```text
COPY 201
     |
     +------> COMPUTE waits >= 201
```

## 9.2 CPU Waits

CPU waits are permitted for:

- shutdown;
- explicit synchronization requests;
- debugging;
- readback APIs that explicitly request synchronous behavior.

They MUST NOT be inserted implicitly into normal dispatch paths.

---

# 10. GPU Fencing and Stall Prevention

The SDK must actively prevent synchronization patterns that create hidden stalls.

Every resource access should be associated with:

```text
last writer queue
last writer fence
read dependencies
pending writes
pending copies
residency state
```

Before a resource is reused:

```text
resource.last_gpu_fence <= completed_fence
```

must be established.

The runtime MUST detect:

- CPU wait after dispatch;
- queue wait on an unnecessarily late fence;
- repeated queue ownership transitions;
- resource reuse before GPU completion;
- copy/compute dependency chains;
- unnecessary global barriers;
- serialization caused by shared resources.

---

# 11. Barrier System

`dxbarrier` owns resource state transitions.

The preferred model is graph-derived barriers.

The user should be able to express:

```text
A writes Buffer X
B reads Buffer X
```

and the graph compiler produces the required synchronization.

The runtime should support enhanced barrier paths where available and fall back to compatible D3D12 mechanisms where required.

Barriers must be traceable through `dxtrace`.

---

# 12. Command Recording

Each graph execution should ideally record work into reusable command structures.

Hot-path design:

```text
application
   |
   v
DXGraph
   |
   v
command allocator
   |
   v
command list
   |
   v
queue
```

The runtime SHOULD cache:

- command allocators;
- pipeline state objects;
- root signatures;
- descriptor heaps;
- shader blobs;
- graph compilation results.

---

# 13. Descriptor Management

Descriptor allocation MUST NOT require a new CPU-side allocation per dispatch.

Preferred hierarchy:

```text
persistent descriptor heap
        |
descriptor cache
        |
bindless/indexed resources where supported
        |
transient descriptors
```

Compute modules should use stable descriptor layouts.

---

# 14. Memory Architecture

`dxmem` is responsible for a tiered memory system.

Logical tiers:

```text
TIER 0 — GPU VRAM
TIER 1 — BAR-visible / mapped GPU memory
TIER 2 — system RAM
TIER 3 — remote GPU memory
TIER 4 — remote RAM
TIER 5 — NVMe / DirectStorage
```

The application sees a logical `DXBuffer`.

The runtime tracks its physical residency.

---

# 15. VRAM ↔ RAM Sharing

DXAiT must support buffers whose backing storage can move between VRAM and system memory.

Resource states:

```text
RESIDENT_VRAM
RESIDENT_SYSTEM
MIGRATING_TO_VRAM
MIGRATING_TO_SYSTEM
REMOTE_RESIDENT
DISK_RESIDENT
PREFETCHING
EVICTING
```

The memory manager tracks:

```text
size
preferred location
current location
last GPU use
last CPU use
read/write frequency
priority
evictability
prefetch status
```

---

# 16. Lazy Tensor Loading

Large models MUST NOT require all weights to reside in VRAM.

Example:

```text
Model
 |
 +-- layer 0     VRAM
 +-- layer 1     VRAM
 +-- layer 2     VRAM
 +-- layer 3     RAM
 +-- layer 4     NVMe
 +-- layer 5     NVMe
 +-- layer 6     NVMe
```

When execution reaches layer 4:

```text
NVMe
  |
DirectStorage
  |
system memory / GPU destination
  |
VRAM
  |
compute
```

The graph scheduler should know about this dependency.

---

# 17. DirectStorage Integration

`dxio` provides asynchronous storage operations.

Required operations:

```c
dxio_prefetch(...)
dxio_read(...)
dxio_read_async(...)
dxio_cancel(...)
dxio_query(...)
```

The preferred path is:

```text
NVMe
  |
DirectStorage
  |
GPU-accessible destination
```

where the platform and resource configuration permit it.

The system should support asynchronous prefetch distance:

```text
prefetch_depth = 1
prefetch_depth = 2
prefetch_depth = N
```

The scheduler may adapt this according to measured latency.

---

# 18. ReBAR / BAR-Aware Memory

The runtime must detect BAR/resource visibility.

BAR-visible allocations can be classified separately:

```text
VRAM_LOCAL
VRAM_BAR_VISIBLE
SYSTEM_VISIBLE
```

The scheduler should understand that BAR access is not equivalent to local VRAM access.

BAR memory is therefore a performance tier, not simply "free VRAM."

---

# 19. Memory Prefetch Scheduler

For sequential inference:

```text
current layer = N
next layer    = N+1
future layer  = N+2
```

The memory system can perform:

```text
compute N
   |
   +---- prefetch N+1
   |
   +---- prefetch N+2
```

The transfer must overlap computation whenever dependencies permit.

---

# 20. Memory Eviction

Eviction priority should consider:

```text
last use
reuse distance
size
reload cost
priority
pinned status
graph dependencies
```

Pinned resources MUST NOT be evicted.

Resources with a near-future graph dependency SHOULD NOT be evicted.

---

# 21. Execution Graph

`dxgraph` is the central execution abstraction.

A graph is a DAG:

```text
A ───> B ───> D
      |
      └──> C ───> D
```

Nodes can represent:

```text
dispatch
copy
clear
barrier
prefetch
eviction
remote transfer
collective operation
host callback
```

The graph compiler derives:

- dependencies;
- queue assignment;
- barriers;
- fence signals;
- fence waits;
- residency requirements;
- transfer scheduling.

---

# 22. Graph Execution

Example:

```c
DXGraph *graph;

dxgraph_create(device, &graph);

dxgraph_dispatch(graph, gemm);
dxgraph_dispatch(graph, rmsnorm);
dxgraph_dispatch(graph, attention);
dxgraph_dispatch(graph, ffn);

dxgraph_compile(graph);
dxgraph_execute(graph);
```

Compilation should be reusable.

The graph may be parameterized by:

```text
tensor addresses
sequence length
batch size
token position
workspace
```

without recompiling the complete dependency structure.

---

# 23. Automatic Queue Placement

A graph node declares capabilities:

```text
requires compute
requires graphics
requires copy
```

The scheduler selects:

```text
DIRECT
COMPUTE
COPY
```

according to:

```text
capability
availability
dependency
estimated cost
locality
```

Example:

```text
GEMM
  -> COMPUTE

COPY weights
  -> COPY

graphics-compatible operation
  -> DIRECT
```

A compute-capable workload may be placed on DIRECT if policy allows and the cost model determines that this is beneficial.

---

# 24. Queue Migration

Queue migration MUST be dependency-aware.

Bad:

```text
compute → direct → compute → direct
```

Good:

```text
compute ------------------------>
          |
          +---- direct ---------->

copy --------------------------->
```

The scheduler should batch compatible work.

---

# 25. `dxtrace`

`dxtrace` is a first-class SDK component.

It must provide:

```text
CPU timeline
GPU timeline
queue timeline
fence timeline
resource timeline
memory timeline
network timeline
storage timeline
shader/dispatch metadata
barrier metadata
```

---

# 26. Trace Events

Minimum event types:

```text
FRAME_BEGIN
FRAME_END

GRAPH_BEGIN
GRAPH_END

DISPATCH_BEGIN
DISPATCH_END

COPY_BEGIN
COPY_END

BARRIER_BEGIN
BARRIER_END

FENCE_SIGNAL
FENCE_WAIT

RESOURCE_CREATE
RESOURCE_DESTROY

RESIDENT
EVICT
PREFETCH

NETWORK_SEND
NETWORK_RECEIVE

NODE_BEGIN
NODE_END

QUEUE_IDLE
QUEUE_STALL
GPU_IDLE
CPU_WAIT
```

---

# 27. GPU Timestamping

Each supported queue should expose timestamp frequency.

Trace entries should include:

```text
CPU start
CPU end
GPU start
GPU end
queue
device
server
graph node
resource IDs
```

This permits direct CPU/GPU correlation.

---

# 28. Stall Detector

`dxtrace` SHOULD automatically classify stalls.

Categories:

```text
CPU_SUBMISSION_STALL
CPU_WAIT_STALL
GPU_IDLE
QUEUE_STARVATION
FENCE_WAIT
BARRIER_SERIALIZATION
MEMORY_RESIDENCY_STALL
VRAM_PRESSURE
COPY_STARVATION
NETWORK_STALL
STORAGE_STALL
QUEUE_SWITCH_OVERHEAD
```

Example:

```text
GPU STALL
Duration: 73 us

Queue: COMPUTE
Waiting for: COPY fence 1821

Resource:
layer_18_weights

Likely cause:
weight prefetch completed too late
```

---

# 29. Multi-GPU Architecture

`dxpeer` handles local GPU-to-GPU communication.

A machine may contain:

```text
GPU 0
GPU 1
GPU 2
...
GPU N
```

Each GPU has independent:

```text
device
queues
memory
fences
graphs
pipelines
```

The cluster layer presents them as a device fabric.

---

# 30. Heterogeneous GPUs

Different GPUs may have different:

```text
VRAM
compute throughput
wave size
shader model
FP16
BF16
INT8
FP8
matrix capabilities
memory bandwidth
queue availability
```

The scheduler MUST NOT assume identical devices.

Example:

```text
GPU A — RDNA4
GPU B — RDNA3
GPU C — RDNA2
GPU D — legacy D3D12 GPU
```

Each module exposes a capability ladder.

---

# 31. Capability Tiers

Each shader module SHOULD implement:

```text
TIER 0
portable D3D12 shader

TIER 1
wave/subgroup optimized

TIER 2
architecture/hardware optimized

TIER 3
specialized matrix/instruction path
```

The runtime selects the highest compatible tier.

The fallback path MUST remain functional.

---

# 32. Multi-Server Fabric

`dxcluster` provides network execution.

A server is an execution node containing:

```text
CPU
RAM
one or more GPUs
VRAM
NVMe
network interface
DXAiT runtime
```

The client sees a logical cluster.

---

# 33. Server Discovery

Discovery may use:

```text
explicit configuration
DNS/service discovery
manual IP
authenticated discovery service
```

Each node advertises:

```text
hostname
node ID
protocol version
GPU list
VRAM
system RAM
network bandwidth
network latency
supported modules
supported shader tiers
```

---

# 34. Remote Handles

Native D3D12 handles MUST NOT cross the network.

Instead:

```c
typedef uint64_t DXRemoteBuffer;
typedef uint64_t DXRemoteFence;
typedef uint64_t DXRemoteGraph;
```

These are server-local identifiers.

---

# 35. Remote Execution

Example:

```text
CLIENT
 |
 | dispatch GEMM(buffer 10, buffer 11, buffer 12)
 |
SERVER
 |
 +-- resolve remote handles
 +-- record D3D12 command list
 +-- execute on GPU
 +-- signal server fence
 |
CLIENT
```

Only required data crosses the network.

---

# 36. Persistent Remote Residency

Remote resources should remain resident between operations.

Example:

```text
SERVER A

Model:
layer 0
layer 1
layer 2
layer 3
layer 4

KV:
session 21

Workspace:
GEMM_17
```

A subsequent operation references existing handles rather than retransmitting data.

---

# 37. Network Transfers

`dxcluster` should support:

```text
send
receive
send_async
receive_async
copy
broadcast
scatter
gather
```

Transfers should integrate with graph dependencies.

Example:

```text
GPU A GEMM
   |
   | signal
   v
NETWORK SEND
   |
   | transfer
   v
NETWORK RECEIVE
   |
   v
GPU B ATTENTION
```

---

# 38. Multi-Server Graph

A graph may span nodes:

```text
                 GRAPH
                   |
        +----------+----------+
        |                     |
     SERVER A             SERVER B
        |                     |
      GEMM                  ATTENTION
        |                     |
        +------ NETWORK ------+
```

The graph compiler must create distributed dependencies.

---

# 39. Network-Aware Scheduling

The scheduler must estimate:

```text
network latency
network bandwidth
payload size
GPU execution time
queue availability
serialization cost
```

A remote operation should only be selected when:

```text
remote_compute_gain > transfer_cost
```

unless explicitly forced.

---

# 40. Distributed Tensor Sharding

`dxshard` defines:

```text
ROW_SHARD
COLUMN_SHARD
BLOCK_SHARD
HEAD_SHARD
LAYER_SHARD
EXPERT_SHARD
```

Example:

```text
Tensor
+-------------+-------------+-------------+
| GPU 0       | GPU 1       | GPU 2       |
| columns     | columns     | columns     |
+-------------+-------------+-------------+
```

---

# 41. Collectives

`dxcollective` must expose:

```text
broadcast
scatter
gather
allgather
reduce
allreduce
alltoall
send
receive
```

The implementation may choose:

```text
GPU P2P
shared memory
host staging
network
```

according to topology.

---

# 42. Topology Model

The runtime should construct a topology graph:

```text
GPU0
 |
PCIe
 |
NIC
 |
Network
 |
NIC
 |
PCIe
 |
GPU1
```

Local paths may be:

```text
GPU0 ↔ GPU1
GPU0 ↔ RAM
GPU1 ↔ RAM
```

Each edge gets measured or estimated:

```text
bandwidth
latency
directionality
```

---

# 43. Remote Failure Handling

Nodes can enter:

```text
ONLINE
DEGRADED
UNREACHABLE
FAILED
RECOVERING
```

If a node fails:

1. stop scheduling new work;
2. identify dependent graph nodes;
3. identify lost resources;
4. locate replicas/checkpoints if available;
5. reschedule where possible;
6. report exact failure origin.

No silent recovery.

---

# 44. Security

Remote execution MUST support authenticated sessions.

Recommended model:

```text
node identity
    |
authentication
    |
encrypted transport
    |
capability exchange
    |
session authorization
    |
RPC
```

Remote buffers are scoped to authenticated sessions.

> **Current implementation note:** the shipped `dxnetwork` `NetworkTensorTransport`
> provides a *toggleable XOR* payload obfuscation plus an HMAC-style auth token
> (`SecurityEngine`). This is obfuscation for trusted LAN use, **not** authenticated
> encryption — it is not a substitute for TLS/IPsec on untrusted networks. Real
> encrypted transport (and the multi-server fabric above) remains forward scope.

---

# 45. Compute Module API

Every compute module should use the common runtime.

Example:

```c
DXBLASContext *ctx;

dxblas_create(
    device,
    &ctx
);

dxblas_sgemm(
    ctx,
    command_context,
    ...
);
```

The module must not create its own:

```text
device
queue
allocator
global fence
memory manager
```

unless explicitly required.

---

# 46. BLAS Module

Initial operations:

```text
SGEMM
DGEMM
HGEMM
BFGEMM
GEMM_EX

strided batched GEMM
batched GEMM

GEMV
AXPY
SCAL
DOT
NRM2
ASUM
AMAX

TRSV
TRSM
SYMV
HEMV
SYMM
HEMM
SYRK
HERK
```

Quantized GEMM:

```text
Q4
Q5
Q6
Q8
IQ
K-quants
T-quants
```

---

# 47. Math Module

Required classes:

```text
add
subtract
multiply
divide
scale

relu
silu
gelu
tanh
sigmoid

exp
log
sqrt
rsqrt
pow
abs
sign
clip

softmax
rmsnorm
layernorm

reduce
argmax
argmin
scan
cumsum
```

---

# 48. Quantization Module

Required:

```text
dequantization
quantization
block formats
scale calculation
zero point handling
mixed precision conversion
```

Target formats may include:

```text
Q4
Q5
Q6
Q8
IQ
FP8
MXFP4
BF16
FP16
INT8
```

---

# 49. FFT Module

Support:

```text
1D
2D
3D

forward
inverse

FP32
FP16
```

Future:

```text
batched FFT
real FFT
complex FFT
```

---

# 50. RNG Module

Required:

```text
uniform
normal
uint32
uint64 where practical
counter-based generators
```

Generators should be stateless or explicitly stateful.

---

# 51. Convolution Module

Support:

```text
Conv1D
Conv2D
Conv3D
depthwise
grouped
strided
dilated
```

The module should expose multiple kernel strategies.

---

# 52. Sparse Module

Initial:

```text
CSR
COO
CSC where practical

SpMM
SpMV
SpSV
```

Future:

```text
sparse factorization
block sparse
structured sparsity
```

---

# 53. Attention Module

Architecture-neutral primitives:

```text
QKV projection
scaled dot-product attention
causal attention
paged attention
GQA
MQA
MHA
sliding-window attention
sparse attention
```

Advanced implementations may add:

```text
fused attention
fused RoPE
flash-style attention
```

---

# 54. KV Module

The KV subsystem should manage:

```text
allocation
block tables
paging
append
read
eviction
prefetch
compaction
multi-GPU placement
remote placement
```

KV pages must be regular `DXBuffer` resources underneath the abstraction.

---

# 55. Model Loader

`dxmodel` supports (implemented):

```text
GGUF          (memory-mapped, full metadata + tensor parser)
safetensors   (memory-mapped)
```

Earlier PTE / ONNX / PyTorch-bin parsers existed as stubs and were removed (no torch
dependency). A model does not need to load all tensor bytes into VRAM.

---

# 56. JIT / Shader Compilation

`dxjit` provides:

```text
HLSL source
DXIL
shader library
pipeline creation
specialization
kernel cache
```

Pipeline caching is mandatory for production workloads.

A shader source should not need architecture-specific source files when specialization constants or capability selection are sufficient.

---

# 57. Shader Tiering

Example:

```text
shader:
    gemm.hlsl

tier 0:
    portable implementation

tier 1:
    wave optimized

tier 2:
    matrix optimized

tier 3:
    architecture-specific
```

Selection:

```text
hardware capabilities
        |
        v
tier detector
        |
        v
highest valid kernel
```

---

# 58. Module Independence

Every module must be usable independently.

For example:

```text
dxblas
```

should be usable without:

```text
dxmodel
dxattention
dxcluster
```

Likewise:

```text
dxmem
dxtrace
```

should be usable by non-AI applications.

---

# 59. Portability Model

The SDK should make porting from CUDA/ROCm-style code straightforward.

Examples:

```text
hipblasSgemm
    ↓
dxblas_sgemm

hipMalloc
    ↓
dxmem_alloc

hipMemcpy
    ↓
dxmem_copy

hipStream
    ↓
DXQueue / DXCommandContext

hipEvent
    ↓
DXFence / DXTimeline

hipModuleLaunchKernel
    ↓
dxkernel_dispatch
```

The mapping should be conceptual, not a promise of binary compatibility.

---

# 60. Asynchronous Programming Contract

All core APIs should provide asynchronous variants.

Bad:

```c
dxblas_sgemm(...);
dxwait();
dxmath_silu(...);
```

Preferred:

```c
dxblas_sgemm(...);
dxmath_silu(...);
dxgraph_execute(...);
```

The graph/runtime handles ordering.

---

# 61. Error Model

Every API must return an explicit status.

Example:

```c
typedef enum DXStatus {
    DX_SUCCESS = 0,
    DX_ERROR_INVALID_ARGUMENT,
    DX_ERROR_OUT_OF_MEMORY,
    DX_ERROR_DEVICE_LOST,
    DX_ERROR_UNSUPPORTED,
    DX_ERROR_TIMEOUT,
    DX_ERROR_NETWORK,
    DX_ERROR_STORAGE,
    DX_ERROR_SHADER,
    DX_ERROR_PIPELINE,
    DX_ERROR_SYNCHRONIZATION
} DXStatus;
```

Errors should include diagnostic context where possible.

---

# 62. Device-Lost Handling

The runtime must capture:

```text
device
queue
last submitted fence
last completed fence
graph node
pipeline
resource
trace context
```

before returning `DX_ERROR_DEVICE_LOST`.

`dxtrace` should preserve the final execution timeline.

---

# 63. Resource Lifetime

Resources should use reference-counted or explicit ownership semantics.

A resource may not be destroyed until all dependent GPU work has completed.

The runtime should use deferred destruction:

```text
destroy requested
       |
       v
retire queue
       |
       v
fence completion
       |
       v
actual release
```

---

# 64. Thread Safety

Contexts should define explicit threading rules.

Recommended:

```text
device             thread-safe
allocator          thread-safe
pipeline cache     thread-safe
queue               externally synchronized or per-thread contexts
command recorder    thread-confined
graph compiler      thread-safe
trace collector    thread-safe
```

Hot-path locks should be minimized.

---

# 65. Performance Requirements

Production hot paths SHOULD avoid:

```text
malloc/free
CPU waits
per-dispatch descriptor allocation
pipeline recompilation
unnecessary resource transitions
unnecessary queue switches
synchronous readback
global device locks
```

---

# 66. Command Batching

The runtime should support:

```text
one command list per inference step
```

where practical.

A decode step should ideally look like:

```text
record
  |
  +-- QKV
  +-- RoPE
  +-- KV
  +-- Attention
  +-- FFN
  +-- sampling
  |
submit
```

rather than:

```text
submit
wait
submit
wait
submit
wait
```

---

# 67. Resource Aliasing

The memory system may reuse physical allocations when lifetimes do not overlap.

Graph lifetime analysis should determine:

```text
A live [0, 10]
B live [11, 20]
```

Therefore A's physical allocation can be reused for B.

All aliasing must be synchronized and traceable.

---

# 68. Workspace Management

Compute modules should request workspace:

```c
dxblas_get_workspace_size(...)
```

The runtime may provide:

```text
persistent workspace
graph workspace
per-command workspace
per-device workspace
```

---

# 69. Auto-Tuning

Modules may benchmark multiple kernels:

```text
kernel A
kernel B
kernel C
```

and cache the best result by:

```text
GPU
driver
shader
shape
precision
batch
```

The cache should be persistent.

---

# 70. Benchmark System

`dxbench` should measure:

```text
kernel time
GPU occupancy proxy metrics where available
memory bandwidth
submission overhead
queue utilization
transfer bandwidth
network bandwidth
DirectStorage throughput
```

Results should be exportable.

---

# 71. Validation System

Every module requires:

```text
CPU reference
GPU result
error metric
randomized tests
edge cases
shape tests
precision tests
```

For floating point:

```text
absolute error
relative error
ULP where meaningful
```

---

# 72. Cross-Architecture Test Matrix

The test system should distinguish:

```text
portable correctness
subgroup correctness
optimized correctness
```

Example:

```text
legacy D3D12 GPU
RDNA2
RDNA3
RDNA4
future GPU
```

The same functional test should run through the appropriate capability tier.

---

# 73. Trace-Based Regression Testing

A trace may become a performance regression artifact.

Example:

```text
baseline:
GEMM 0.41 ms

new build:
GEMM 0.57 ms

regression:
+39%
```

The CI system should optionally reject large regressions.

---

# 74. Configuration

Environment variables may override defaults, but the public configuration API should be primary.

Example:

```c
DXRuntimeConfig cfg = {
    .enable_tracing = true,
    .enable_async_copy = true,
    .enable_auto_queue = true,
    .enable_prefetch = true,
    .enable_multi_gpu = true,
    .enable_remote = false
};
```

---

# 75. Debug Modes

Required modes:

```text
NORMAL
VALIDATION
TRACE
STALL_ANALYSIS
DETERMINISTIC
PERFORMANCE
```

Debug mode may enable:

```text
extra barriers
resource validation
synchronization validation
GPU markers
timestamping
```

---

# 76. GPU Markers

Every module should emit meaningful debug labels:

```text
DXBLAS::SGEMM
DXMATH::RMSNORM
DXATTENTION::SDPA
DXKV::APPEND
DXMEM::PREFETCH
DXIO::DIRECTSTORAGE_READ
DXCLUSTER::SEND
```

These labels should be visible to external GPU debugging/profiling tools where supported.

---

# 77. PIX Integration

DXAiT should be designed to coexist with PIX.

Requirements:

- D3D12 debug names;
- GPU markers;
- event ranges;
- timestamp queries;
- meaningful pipeline names;
- resource names;
- command-list labels;
- queue labels.

`dxtrace` is complementary to PIX.

PIX provides external inspection; `dxtrace` provides application/runtime semantics.

---

# 78. Distributed Trace Correlation

Every event gets:

```text
trace_id
node_id
device_id
queue_id
graph_id
node_id
timestamp
```

A distributed execution can therefore be reconstructed:

```text
Server A GPU 0
       |
       | trace_id 9182
       v
Network
       |
       v
Server B GPU 1
```

---

# 79. Multi-Server Security Boundary

Remote nodes must never receive raw pointers.

They receive:

```text
session
resource IDs
command IDs
validated descriptors
```

The server validates:

```text
resource ownership
resource size
operation type
buffer ranges
shader/pipeline identity
```

---

# 80. Remote Command Protocol

Protocol phases:

```text
HELLO
CAPS
AUTH
REGISTER_RESOURCE
UPLOAD
DISPATCH
COPY
SIGNAL
WAIT
READBACK
RELEASE
BYE
```

All messages should include:

```text
protocol version
request ID
session ID
payload size
checksum/authentication metadata
```

---

# 81. Distributed Memory Semantics

A remote buffer can be:

```text
REGISTERED
RESIDENT
IN_FLIGHT
REMOTE
REPLICATED
RETIRED
```

The client does not need to know the physical D3D12 allocation.

---

# 82. Distributed Scheduling Policies

Supported policies:

```text
LOCAL_ONLY
REMOTE_ONLY
BEST
LOWEST_LATENCY
MAX_THROUGHPUT
MIN_MEMORY
BALANCED
USER_PINNED
```

---

# 83. Replication

Optional resource replication:

```text
primary
replica 1
replica 2
```

Useful for:

```text
model weights
read-only tensors
static lookup tables
```

Mutable state such as KV cache should be replicated only when explicitly requested.

---

# 84. Remote Storage

Future extension:

```text
Server A GPU
      |
      v
Server A NVMe

or

Server B GPU
      |
      v
Server B NVMe
```

The scheduler may choose the node where the data already exists.

---

# 85. Inference Execution Example

A hypothetical decode step:

```text
1. Prefetch next layer from NVMe
2. Transfer weights to VRAM
3. QKV GEMM
4. RoPE
5. KV append
6. Attention
7. FFN gate/up GEMM
8. activation
9. FFN down GEMM
10. logits
11. sampling
```

The graph can overlap:

```text
COPY:
prefetch layer N+1

COMPUTE:
execute layer N
```

while another GPU may process another partition.

---

# 86. Full Distributed Example

```text
SERVER A
GPU 0
    QKV
    |
    +----------------------+
                           |
                           v
SERVER B                NETWORK
GPU 1                     |
    ATTENTION <------------+
       |
       v
SERVER C
GPU 0
    FFN
       |
       v
SERVER A
GPU 0
    LOGITS
```

The CPU does not wait between every operation.

---

# 87. Scheduler Ratchet Levels

The scheduler should progressively enable optimizations:

```text
LEVEL 0
single queue

LEVEL 1
multiple queues

LEVEL 2
async copy overlap

LEVEL 3
automatic queue placement

LEVEL 4
multi-GPU

LEVEL 5
VRAM/RAM residency

LEVEL 6
DirectStorage prefetch

LEVEL 7
local GPU fabric

LEVEL 8
multi-server

LEVEL 9
distributed sharding

LEVEL 10
auto-tuned heterogeneous execution
```

Each level must retain correctness if higher-level optimization is disabled.

---

# 88. "Ratchet" Design Principle

Every optimization is optional but monotonic.

Example:

```text
portable shader
      ↓
wave shader
      ↓
matrix shader
      ↓
fused shader
      ↓
multi-GPU
      ↓
multi-server
```

If an optimization is unavailable:

```text
fallback
```

must preserve functionality.

---

# 89. C4 / Heavyweight Resource Operations

The SDK should expose aggressive resource management primitives, but they must remain explicit.

Examples:

```text
force residency
force eviction
prefetch
discard
alias
replicate
migrate
remote pin
disk pin
```

These are the "heavy tools" of the memory subsystem.

They must not be silently invoked without policy permission.

---

# 90. Jackhammer-Class Operations

For benchmarking/debugging, provide explicit stress APIs:

```text
memory stress
queue stress
fence stress
barrier stress
multi-GPU transfer stress
network stress
DirectStorage stress
pipeline compilation stress
```

The purpose is to expose:

```text
driver bugs
synchronization bugs
resource lifetime bugs
residency bugs
queue starvation
performance cliffs
```

These tools belong under `tools/`, not the normal inference path.

---

# 91. Module Contract

Every module must provide:

```text
create
destroy
query capabilities
workspace query
record operation
optional async operation
optional graph node
```

Example:

```c
DXStatus dxblas_create(...);
DXStatus dxblas_destroy(...);

DXStatus dxblas_query_caps(...);

DXStatus dxblas_sgemm(...);

DXStatus dxblas_sgemm_graph(...);
```

---

# 92. Public ABI Rules

Public headers:

- C99-compatible;
- no STL;
- no C++ classes;
- no compiler-specific ABI types;
- opaque handles;
- fixed-width integers;
- explicit alignment;
- explicit ownership;
- explicit lifetime.

C++ applications can link directly.

---

# 93. Build System

Recommended:

```text
CMake
Ninja
Visual Studio 2022
clang-cl
MSVC
Windows SDK
DirectX Shader Compiler
```

Shader pipeline:

```text
HLSL
 |
DXC
 |
DXIL
 |
embedded/cacheable binary
```

---

# 94. Shader Build

A script should discover:

```text
shaders/**/*.hlsl
```

and generate:

```text
DXIL blobs
shader metadata
reflection metadata
```

Adding a shader should require:

```text
add HLSL
build
test
```

rather than manually editing build lists.

---

# 95. Shader Metadata

Each shader should declare:

```text
name
module
version
precision
required features
thread-group size
resource layout
wave requirements
supported tiers
```

Example:

```text
dxblas_gemm_f16
requires: SM6.x
wave: optional
fp16: required
```

---

# 96. Pipeline Cache

Pipeline identity:

```text
GPU
driver
shader hash
root signature
precision
specialization
thread group
module version
```

The runtime caches the resulting pipeline.

---

# 97. Specialization

Prefer runtime specialization over source duplication.

Parameters may include:

```text
tile M
tile N
tile K
wave width
unroll
vector width
workgroup size
```

---

# 98. Memory Safety

All GPU buffer accesses must be range validated in debug builds.

Remote execution MUST always validate ranges.

Out-of-bounds descriptors must produce explicit errors where validation is possible.

---

# 99. Determinism

Where practical, modules should provide deterministic modes.

This may require:

```text
fixed reduction order
fixed RNG seed
fixed kernel selection
fixed scheduling
```

Distributed floating-point reductions may not be bit-identical unless explicitly implemented that way.

---

# 100. Performance Telemetry

Runtime telemetry should expose:

```text
GPU utilization proxy
queue busy time
queue idle time
fence waits
barrier count
dispatch count
copy bandwidth
VRAM usage
RAM usage
NVMe bandwidth
network bandwidth
graph execution time
CPU submission time
```

---

# 101. Diagnostics API

Example:

```c
dxruntime_get_stats(...)
dxqueue_get_stats(...)
dxmem_get_stats(...)
dxgraph_get_stats(...)
dxcluster_get_stats(...)
dxtrace_export(...)
```

---

# 102. Trace Export

Recommended formats:

```text
JSON
CSV
binary trace
Chrome trace-compatible format
```

A future adapter may export directly into external profiling systems.

---

# 103. Testing Requirements

Each subsystem requires:

```text
unit tests
GPU correctness tests
stress tests
failure tests
performance tests
```

The complete suite should be executable with:

```text
ctest --test-dir build
```

---

# 104. Required Initial Test Matrix

At minimum:

```text
device discovery
queue creation
fence signal/wait
barrier correctness
VRAM allocation
RAM allocation
VRAM ↔ RAM copy
descriptor allocation
shader compilation
pipeline creation
SGEMM
HGEMM
BF16 GEMM
quantized GEMM
RMSNorm
softmax
RNG
FFT
model loading
DirectStorage read
graph execution
multi-queue execution
multi-GPU execution
remote execution
distributed tensor transfer
trace generation
stall detection
```

---

# 105. Correctness Rule

Every optimization MUST have a fallback.

Example:

```text
coop/matrix path unavailable
        ↓
wave path

wave unavailable
        ↓
portable path
```

Likewise:

```text
VRAM unavailable
        ↓
RAM

remote unavailable
        ↓
local

DirectStorage unavailable
        ↓
normal asynchronous file I/O
```

---

# 106. Versioning

The SDK should expose:

```c
#define DXAIT_VERSION_MAJOR
#define DXAIT_VERSION_MINOR
#define DXAIT_VERSION_PATCH
```

Protocol versions must be independent:

```c
#define DXCLUSTER_PROTOCOL_VERSION
```

Shader ABI versions must also be tracked independently.

---

# 107. Compatibility

The runtime should distinguish:

```text
API compatibility
shader compatibility
driver compatibility
network protocol compatibility
model format compatibility
```

---

# 108. Example Minimal Application

```c
DXRuntime *runtime;
DXDevice *device;
DXQueue *queue;

dxruntime_create(NULL, &runtime);
dxruntime_create_device(runtime, 0, &device);
dxdevice_get_queue(device, DX_QUEUE_AUTO, &queue);

DXBLASContext *blas;
dxblas_create(device, &blas);

dxblas_sgemm(
    blas,
    queue,
    ...);

dxqueue_submit(queue);
```

The application never needs to manually manage every fence.

---

# 109. Example Graph Application

```c
DXGraph *graph;

dxgraph_create(device, &graph);

dxgraph_add_dispatch(graph, qkv);
dxgraph_add_dispatch(graph, rope);
dxgraph_add_dispatch(graph, attention);
dxgraph_add_dispatch(graph, ffn);

dxgraph_compile(graph);
dxgraph_execute(graph);
```

The runtime determines:

```text
queues
barriers
fences
resource transitions
```

---

# 110. Example Multi-GPU Application

```c
DXCluster *cluster;

dxcluster_create(runtime, &cluster);

dxcluster_add_device(cluster, gpu0);
dxcluster_add_device(cluster, gpu1);

dxcluster_execute_graph(
    cluster,
    graph,
    DX_PLACEMENT_BEST);
```

---

# 111. Example Multi-Server Application

```c
DXCluster *cluster;

dxcluster_create(runtime, &cluster);

dxcluster_connect(
    cluster,
    "server-a",
    ...);

dxcluster_connect(
    cluster,
    "server-b",
    ...);

dxcluster_execute_graph(
    cluster,
    graph,
    DX_PLACEMENT_BEST);
```

The same graph API is retained.

---

# 112. Architecture Invariants

The following are mandatory invariants:

1. No hidden CPU synchronization in hot paths.
2. No raw D3D12 handles cross the network.
3. No resource is destroyed before GPU completion.
4. No remote operation executes without authenticated ownership.
5. No optimized shader path may remove the portable fallback.
6. Queue migration must be dependency-aware.
7. Residency changes must be graph-visible.
8. Trace events must preserve causal relationships.
9. Compute modules must reuse the common runtime.
10. Multi-server execution must remain optional.
11. Core runtime must not require CUDA or ROCm.
12. CPU-visible APIs must remain C ABI compatible.
13. Every asynchronous operation must have an explicit completion mechanism.
14. Every module must expose capability information.
15. Performance optimizations must be measurable through `dxtrace`.

---

# 113. Implementation Order

Recommended development sequence:

## Phase 1 — Runtime

```text
dxruntime
dxadapter
dxqueue
dxfence
dxbarrier
```

## Phase 2 — Memory

```text
dxmem
VRAM
RAM
BAR
residency
deferred destruction
```

## Phase 3 — Shader Infrastructure

```text
dxjit
shader compiler
pipeline cache
reflection
```

## Phase 4 — Trace

```text
dxtrace
timestamps
markers
fence timeline
stall detector
```

## Phase 5 — Graph

```text
dxgraph
barrier compiler
queue scheduler
async copy
```

## Phase 6 — Storage

```text
dxio
DirectStorage
lazy loading
prefetch
```

## Phase 7 — Compute Modules

```text
dxmath
dxblas
dxquant
dxfft
dxrand
dxconv
dxsparse
```

## Phase 8 — AI Modules

```text
dxattention
dxkv
dxmodel
```

## Phase 9 — Local Multi-GPU

```text
dxpeer
dxshard
dxcollective
```

## Phase 10 — Network Fabric

```text
dxcluster
authentication
remote buffers
remote graphs
distributed trace
```

## Phase 11 — Optimization

```text
auto-tuning
adaptive prefetch
heterogeneous scheduling
replication
failure recovery
```

---

# 114. Definition of "Fully Functional"

DXAiT is considered functionally complete when it can:

1. discover multiple D3D12 GPUs;
2. create DIRECT/COMPUTE/COPY queues;
3. execute asynchronous compute;
4. synchronize through GPU fences;
5. perform correct resource barriers;
6. allocate VRAM and system memory;
7. asynchronously migrate resources;
8. load large tensors lazily from storage;
9. overlap DirectStorage/copy with compute;
10. select queues automatically;
11. execute a graph without CPU synchronization between operations;
12. trace the entire execution;
13. identify GPU/queue/fence stalls;
14. execute across multiple local GPUs;
15. shard tensors;
16. execute collectives;
17. connect multiple Windows servers;
18. maintain remote buffers;
19. execute distributed graphs;
20. trace distributed execution;
21. survive node/device failures according to policy;
22. execute BLAS/math/quantization/model primitives;
23. provide architecture-specific optimized shaders with fallback;
24. remain usable without CUDA or ROCm.

---

# 115. Final Architecture

The intended final system is:

```text
                                      DXAiT
                                        |
        +-------------------------------+-------------------------------+
        |                               |                               |
     COMPUTE                       EXECUTION                         FABRIC
     MODULES                       ENGINE                            LAYER
        |                               |                               |
 +------+------+              +---------+---------+             +-------+-------+
 |      |      |              |         |         |             |       |       |
BLAS  MATH   QUANT          GRAPH    SCHED     TRACE          LOCAL   REMOTE  STORAGE
 |      |      |              |         |         |             |       |       |
FFT   RNG   CONV             QUEUE    FENCE     PIX           GPUs   SERVERS NVMe
 |      |      |              |         |         |             |       |       |
SPARSE ATTENTION KV         BARRIER   MEMORY   TIMELINE       P2P   NETWORK DirectStorage
        |                       |
      MODEL                  RESIDENCY
        |
      JIT/HLSL
```

The fundamental execution path becomes:

```text
                 APPLICATION / MODEL RUNTIME
                            |
                            v
                         DXGRAPH
                            |
                            v
                         DXSCHED
                            |
             +--------------+--------------+
             |              |              |
          COMPUTE        DIRECT          COPY
             |              |              |
             +--------------+--------------+
                            |
                          DXFENCE
                            |
                         DXMEM
             +--------------+--------------+
             |              |              |
            VRAM           RAM            NVMe
             |                             |
             +---------- DXIO -------------+
                            |
                       DXCLUSTER
                            |
                 +----------+----------+
                 |                     |
              SERVER A              SERVER B
                 |                     |
                GPU                   GPU
                 |                     |
                 +------ NETWORK ------+
                            |
                         DXTRACE
```

**This is the central design principle of DXAiT:**

> **A compute module supplies an operation. The execution fabric decides where it lives, how it is synchronized, how its memory becomes resident, which queue executes it, whether another GPU/server should execute it, and how the entire operation is measured.**

That separation is what allows every future module to become a portable building block rather than another isolated runtime.
