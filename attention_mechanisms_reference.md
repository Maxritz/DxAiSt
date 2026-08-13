# Attention mechanisms in LLMs: a working reference

Compiled August 2026. Scope is broad but not infinite, this covers every mechanism that either shipped in a production model or is cited repeatedly as a load-bearing technique in current literature. Organised by what problem each one solves, not chronologically, since that is how you will actually reach for this when deciding what to port to LLAMA-DX or VAiSt.

Notation used throughout: Q, K, V are query/key/value matrices, shape roughly [seq_len, d_head]. n = sequence length, d = head dim, h = number of heads.

---

## 1. Foundational

### 1.1 Scaled dot-product (softmax) attention
What it does: for each query, computes a weighted sum over all values, where the weight is the softmax-normalised similarity between that query and every key.

Implementation:
```
Attention(Q, K, V) = softmax(QK^T / sqrt(d)) V
```
The 1/sqrt(d) scaling stops dot products from growing with d and pushing softmax into a saturated regime. This is the operation every variant below is either accelerating, sparsifying, or approximating.

Cost: O(n^2 d) time and O(n^2) memory for the score matrix if materialised naively.

### 1.2 Multi-head attention (MHA)
What it does: runs several independent attention operations ("heads") in parallel, each with its own learned Q/K/V projections, then concatenates and projects back down. Lets different heads specialise (syntax, coreference, position, whatever the gradient finds).

Implementation:
```
head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
MultiHead(Q,K,V) = Concat(head_1 ... head_h) W^O
```
Each head typically has d_head = d_model / h. This is the baseline every efficiency variant is measured against, and the one your KV cache is full-size for: every head gets its own K and V.

### 1.3 Causal attention
What it does: masks out future positions so token t can only attend to positions <= t. Required for autoregressive decoding.

Implementation: add -inf to the score matrix at positions j > i before the softmax, or equivalently only compute the lower-triangular block. In fused kernels this is done by skipping upper-triangular tiles entirely rather than masking after the fact, which is where a chunk of FlashAttention's speedup over naive causal masking comes from.

### 1.4 Cross-attention
What it does: queries come from one sequence, keys and values from another. Used to let a decoder attend to an encoder's output, or to fuse a different modality (image tokens, audio) into a text stream.

Implementation: identical math to self-attention, Q = X_a W^Q, K = X_b W^K, V = X_b W^V where X_a != X_b. Standard in encoder-decoder transformers (original 2017 architecture, T5) and in most vision-language fusion layers.

---

## 2. Hardware-aligned exact-attention kernels

These do not change what attention computes, only how the computation is scheduled on the GPU. Directly relevant to your DXLA/Cooperative Vector work.

### 2.1 FlashAttention (v1/v2/v3)
What it does: computes the exact same softmax attention output as the naive formula, but never materialises the full n x n score matrix in HBM.

Implementation: tiles Q, K, V into blocks sized to fit SRAM. For each Q tile, streams over K/V tiles, maintaining a running (unnormalised) output accumulator and a running max/sum for a numerically stable online softmax, then rescales at the end. This is the "flash" trick: online softmax lets you fuse QK^T, mask, softmax, and PV into one kernel without ever writing the intermediate n x n matrix to global memory.

- v1: introduced IO-aware tiling and recomputation in the backward pass instead of storing the full score matrix.
- v2: better work partitioning and parallelism across thread blocks, reduced non-matmul FLOPs, roughly 2x over v1.
- v3 (Hopper-targeted): exploits async execution, warp specialisation, and FP8 with an incoherent-processing trick to correct for quantisation error; hits far higher hardware utilisation than v2 did on H100 by overlapping GEMM and softmax across warps instead of serialising them.

This is squarely the shape of the fused flash-attention compute shader you agreed on for LLAMA-DX. RDNA has no direct analogue of Hopper's async barrier/TMA hardware, so your gfx1201 path realistically gets the tiling and online-softmax parts of v1/v2, not the warp-specialised async pipeline of v3, unless Cooperative Vector exposes something equivalent.

### 2.2 PagedAttention
What it does: not an attention algorithm change, a KV-cache memory manager. Solves fragmentation when serving many concurrent sequences of different lengths.

Implementation: KV cache is stored in fixed-size, non-contiguous "pages" (blocks), analogous to OS virtual memory paging, with a per-sequence block table mapping logical positions to physical pages. Attention kernels are written to gather across these page tables instead of assuming one contiguous KV buffer per sequence. Introduced by vLLM specifically to cut memory waste from over-allocating worst-case-length contiguous buffers.

Relevant to AMD-AI-COMPASS/LLAMA-ALL-INCLUSIVE if you ever move from single-stream inference to a batched server; not something you need for a single-user DirectX inference backend.

### 2.3 Ring attention
What it does: distributes attention computation for very long sequences across multiple devices, without ever holding the full sequence's K/V on one device.

Implementation: sequence is split into chunks, one per device, arranged in a ring. Each device holds its local Q, K, V. Attention proceeds in rounds, each device computes partial attention against its local K/V, then K/V blocks are passed to the next device in the ring (overlapped with compute), accumulating the online-softmax running statistics as in FlashAttention. A causal-aware variant ("striped attention") rebalances the workload since later chunks in a causal mask have less to do.

Multi-device only, not applicable to your single-card RDNA4/RDNA2 setup.

---

## 3. KV-cache head-sharing family (MHA -> MQA -> GQA -> MLA)

This is the lineage that matters most for your decode-bound bottleneck, since KV cache size directly gates how many tokens you can keep resident and how much bandwidth per token you spend.

### 3.1 Multi-Query Attention (MQA)
What it does: all query heads share a single key head and a single value head. Cuts KV cache size by a factor of h (number of heads).

Implementation: same projections as MHA except W^K and W^V are shared, i.e. only one K, one V computed per token instead of h of each. Query heads still get independent W^Q. Big win for memory-bandwidth-bound decode, since decode is one query token at a time reading the entire KV cache from memory, but it noticeably degrades quality versus full MHA because all heads are now forced to read from the same K/V representation.

### 3.2 Grouped-Query Attention (GQA)
What it does: middle ground. Query heads are split into g groups, each group shares one K/V head. g=1 recovers MQA, g=h recovers MHA.

Implementation: mapping function group(head_i) = floor(i / (h/g)) determines which shared K/V head a given query head reads, realised in practice as an interleaved repeat_kv that broadcasts each K/V head across its group before the QK^T matmul. Existing MHA checkpoints can be "uptrained" into GQA cheaply, by mean-pooling the K/V head weights within each group and fine-tuning briefly, rather than retraining from scratch. This is the current default baseline across most open-weight models (Llama family, Mistral, Qwen3, Gemma, gpt-oss), sitting under both llama.cpp's ggml scheduler and whatever LLAMA-ALL-INCLUSIVE is doing for RDNA4-specific kernels.

### 3.3 Multi-Head Latent Attention (MLA)
What it does: instead of caching full K and V per head, compresses them into a shared low-rank latent vector per token, and reconstructs per-head K/V from that latent on the fly. Achieves a much larger KV cache reduction than GQA while staying closer to full-MHA quality, since it's a learned compression rather than a coarse sharing scheme.

Implementation (DeepSeek-V2/V3 formulation): a down-projection matrix compresses the hidden state into a low-dimensional latent c_KV (order of a few hundred dims, versus h * d_head for full MHA). Only this latent is cached, not per-head K/V. At attention time, per-head K and V are reconstructed via up-projection matrices from the cached latent. Positional information (RoPE) is handled by splitting each head into a "content" part (compressed) and a small separate "rope" part (kept uncompressed, since RoPE's rotation doesn't commute cleanly with the low-rank compression), then concatenating the two before the QK^T dot product. The up-projection matrices can be algebraically absorbed into the W^Q and W^O matrices at inference time so you never materialise the full per-head K/V, which is where the actual FLOP savings come from beyond the cache-size savings.

### 3.4 Grouped-Head Latent Attention (GTA)
What it does: a 2025 variant combining ideas from GQA and MLA. Shares attention *maps* (not just K/V) across groups of heads, and compresses only the value cache into a latent space via a learned nonlinear decoder.

Implementation: (1) shared attention-score mechanism, one attention map is computed and reused across a head group rather than recomputed per head, cutting key-cache size; (2) a nonlinear value decoder projects a compressed latent value cache back to full per-head V. Reported gains: up to 62.5% fewer attention FLOPs than GQA and up to 70% smaller KV cache, while avoiding some of MLA's extra reconstruction overhead. Recent enough that you should treat it as promising-but-unproven rather than something to build a dependency on.

### 3.5 Cross-Layer Attention (CLA) and Depth-Attention
What it does: shares K/V not just across heads within a layer but across *layers*, since adjacent transformer layers often produce fairly similar K/V representations. Cuts memory further orthogonally to GQA/MLA.

Depth-Attention (2026) is a more recent refinement: rather than sharing K/V wholesale across layers, it lets a layer's query attend over *earlier layers' keys at the same token position* inside the attention module itself, mixing information across depth without adding state beyond the existing KV cache. Framed explicitly as complementary to GQA/MLA rather than competing with them.

---

## 4. Sparse attention: fixed, hand-designed patterns

Pre-learned-sparsity generation. All reduce O(n^2) to something closer to O(n) or O(n log n) by restricting which token pairs are ever compared, chosen by a fixed rule rather than learned per-input.

### 4.1 Sliding window / local attention
What it does: each token only attends to a fixed-size window of nearby tokens. O(n * w) instead of O(n^2) for window size w.

Implementation: mask everything outside [i-w, i] (causal) or [i-w, i+w] (bidirectional) before or during the score computation; in fused kernels the out-of-window K/V tiles are simply never loaded. Used in most long-context models as one component of a hybrid scheme (Mistral's original sliding-window layers, DuoAttention's sliding-window heads).

### 4.2 Global + local (Longformer, ETC, BigBird)
What it does: combines a local sliding window with a small set of "global" tokens that attend to, and are attended to by, everything. Captures long-range dependency at low cost via those global anchor tokens.

Implementation, Longformer: dilated sliding window (some heads skip positions to widen effective receptive field without more compute) plus task-specific global tokens (e.g. [CLS], or question tokens in QA) that get full bidirectional attention. BigBird adds a third component, a small number of random attention edges per token, and proves the combination is a universal sequence-function approximator despite the sparsity. Both need custom block-sparse matmul kernels to actually realise the theoretical speedup; naive masking on top of dense compute gets you nothing.

### 4.3 Sparse Transformer (strided / fixed factorized)
What it does: earliest of this family (2019). Factorises full attention into two sparse patterns whose product approximates full coverage, e.g. one head attends to a strided pattern (every k-th token), another to a local block, and stacking layers with different patterns lets information propagate across the full sequence over depth.

### 4.4 Dilated attention (LongNet)
What it does: window size grows exponentially with distance, similar in spirit to dilated convolutions. Lets a model reach extremely long context (LongNet's original claim was up to a billion tokens) while keeping attention linear in n.

### 4.5 Attention sinks / StreamingLLM
What it does: a narrow but load-bearing observation. If you run pure sliding-window attention at inference and just evict old KV entries once the window fills, quality collapses once you exceed the training window, because models learn to dump a disproportionate amount of softmax mass onto the first few tokens regardless of their semantic content ("attention sink" tokens). Keep those few initial tokens in the KV cache permanently alongside the sliding window, and the model generalises to effectively unbounded sequence length without any fine-tuning.

Implementation: KV cache = [first k tokens] + [most recent w tokens], everything in between evicted. Trivial to implement, large practical effect, now standard practice in most production long-context serving stacks.

---

## 5. Sparse attention: learned / dynamic patterns

The current frontier as of mid-2026. Instead of a hand-designed fixed pattern, a lightweight auxiliary mechanism decides per-query, per-input, which tokens are worth attending to, and that decision is trained jointly with the model rather than bolted on afterward.

### 5.1 Native Sparse Attention (NSA) — DeepSeek, Feb 2025
What it does: the first design proven natively trainable end-to-end, hardware-aligned, and competitive with full attention on quality, at meaningful sequence lengths. Won a Best Paper award at ACL 2025.

Implementation: three parallel branches per query, whose outputs are gated and combined —
1. **Compressed attention**: keys/values are pooled into coarse block-level summaries (block size l, e.g. 32), giving cheap coverage of the whole sequence.
2. **Selected attention**: reuses/lightly reprocesses the compression-branch scores to pick the top-n most relevant fine-grained blocks (block size l', e.g. 64), and attends to those at full resolution, always force-including the first block and a couple of local blocks.
3. **Sliding window**: a standard local window (e.g. 512) for guaranteed local coherence.

Kernel design loads queries by GQA groups, fetches the corresponding sparse KV blocks, and does the actual attention math on-chip in SRAM, i.e. it's designed from the start as a FlashAttention-style fused kernel, not a masking layer on top of dense attention.

### 5.2 DeepSeek Sparse Attention (DSA) — DeepSeek-V3.2
What it does: a token-level (not block-level) refinement, deployed via continued pretraining from V3.1 rather than trained from scratch. Sits on top of MLA.

Implementation: a "lightning indexer", a handful of low-dimensional indexer heads (typically 4), computes a cheap relevance score between the current query and every previous token using a ReLU-based low-rank projection: `I_t,s = sum_j w_t,j * ReLU(q_t,j · k_s)`. Top-k of that index selects which tokens the real (full-dimensional) attention computation runs over. Since d_indexer << d_model, the O(n^2) indexer pass is cheap, and the expensive part drops to O(n*k). Reported at roughly 1.6% of full-attention operations at 128K context with quality reported as indistinguishable from dense. Distinct from NSA despite the naming confusion in some serving-engine codebases (SGLang's backend is literally named "nsa" but implements the DSA pipeline; started before V3.2's terminology settled).

### 5.3 Mixture of Block Attention (MoBA)
What it does: partitions context into large blocks, uses a gating mechanism (block-averaged keys, trained only through the ordinary language-modelling loss, no auxiliary indexer loss) to route each query to its most relevant blocks. Operates directly on top of a GQA backbone.

### 5.4 SeerAttention
What it does: MoE-inspired. A trainable small MLP gate learns to extract a relevance signal per block from the model's own hidden states, used to decide which blocks to keep dense attention over.

### 5.5 InfLLM-V2
What it does: aims for zero-shot dense-to-sparse switching, i.e. a model trained dense can be run sparse at inference without retraining, by unifying a parameter-free block-selection rule with a local sliding window.

### 5.6 MInference / FlexPrefill / XAttention / SpargeAttn
What they do: prefill-focused block-sparse attention that computes a dynamic sparse mask per input rather than a fixed pattern, targeting the long-prompt prefill phase specifically (as opposed to decode, which is a different bottleneck, bandwidth not compute). XAttention adds a finer anti-diagonal scoring pass to better capture the empirically-observed "vertical and slash" sparsity structure that real attention maps tend to have.

### 5.7 Quest / DoubleSparsity / MInference-for-decode class
What they do: decode-time equivalents of the above, deciding at each decode step which cached KV blocks are worth reading, since decode is memory-bandwidth-bound and skipping irrelevant cache reads is a direct latency win, distinct from the compute-bound prefill sparsity problem.

### 5.8 MiniMax Sparse Attention (MSA)
What it does: keeps a GQA backbone and adds learned sparse block selection on top, deployed in MiniMax-M3. Positioned by MiniMax's own related-work section as sitting alongside NSA/DSA/MoBA/InfLLM-V2 as one of several "natively trained sparse-attention" designs whose indexer is trained jointly during pretraining rather than retrofitted.

---

## 6. Linear / kernel attention (avoiding softmax's O(n^2) entirely)

Different strategy from sparsity: instead of skipping token pairs, reformulate the similarity function so the whole computation becomes linear in n.

### 6.1 Linear attention
What it does: replaces the softmax kernel with a feature map phi(x) such that similarity is a dot product of features, which lets you use the associativity of matrix multiplication to compute `phi(Q) (phi(K)^T V)` instead of `(phi(Q) phi(K)^T) V`, avoiding ever forming the n x n matrix. Complexity drops to O(n) in sequence length.

Implementation: `Attention(Q,K,V) ≈ phi(Q) (phi(K)^T V) / (phi(Q) (phi(K)^T 1))`, where phi is a positive feature map (e.g. elu(x)+1 in the original Katharopoulos formulation). This also gives you an equivalent recurrent formulation for decoding: maintain a running state matrix `S = sum_t phi(k_t) v_t^T` and update it token by token, which is O(1) per decode step instead of scanning the whole KV cache. Tends to underperform softmax attention on tasks needing sharp, high-precision retrieval, which is the core trade-off of the whole linear-attention family.

### 6.2 Performer
What it does: approximates softmax attention specifically (not just "a kernel"), using random feature maps (FAVOR+, positive orthogonal random features) chosen so the linear-attention approximation converges to the true softmax result in expectation. Unbiased approximation rather than a different similarity function.

### 6.3 Linformer
What it does: projects the n-length K and V sequences down to a fixed, much smaller length k via learned low-rank projection matrices before computing attention, on the empirical claim that attention matrices are approximately low-rank. Reduces complexity to O(n*k).

### 6.4 Reformer (LSH attention)
What it does: uses locality-sensitive hashing to bucket similar queries and keys together, then only computes attention within each hash bucket, avoiding the need to compare all pairs. Complexity roughly O(n log n).

### 6.5 RWKV / Lightning Attention / gated linear attention family
What they do: modern production-grade linear-attention variants, generally add a learned data-dependent decay/gate on the recurrent state to fix linear attention's classic weakness (unbounded state accumulation, poor forgetting). RWKV in particular is explicitly framed as "an RNN that trains like a transformer", parallel-scan formulation for training, pure recurrence for inference. Lightning Attention (MiniMax) is the linear-attention component used in MiniMax's hybrid stacks (interleaved with periodic full-attention layers to recover quality linear attention alone tends to lose on retrieval-heavy tasks).

### 6.6 Multi-matrix factorization attention (MFA) and other 2025/2026 factorisation variants
What they do: general family exploring different low-rank factorisations of the Q/K/V or attention-score matrices to hit different points on the quality/compute/memory trade-off surface than the specific choices MLA and Linformer made. Treat any specific name here as an incremental variation on the low-rank theme above rather than a fundamentally new mechanism.

---

## 7. State-space models (adjacent to, not strictly, attention)

Not attention mechanisms in the QKV sense, but they're the mechanism most directly competing with attention for the "how do we model long sequences" slot, and modern architectures increasingly hybridise the two, so they belong in this list.

### 7.1 Mamba / Mamba-2 (selective state-space models, SSMs)
What it does: processes the sequence via a linear recurrent state, similar in spirit to an RNN, but with the recurrence made input-dependent ("selective"), so the model can choose what to keep versus discard at each timestep, and the recurrence is expressed as a structured matrix that has a parallel-scan formulation for efficient training. O(n) time and O(1) memory per generated token at inference, since there's no growing KV cache at all, just a fixed-size state.

### 7.2 Hybrid stacks (Jamba, MiniMax's hybrid line, and similar)
What they do: interleave a majority of cheap linear/SSM layers with a minority of full (or GQA) attention layers, on the observation that pure linear/SSM models lose noticeably on tasks needing precise long-range retrieval, while pure attention is expensive; a periodic full-attention layer recovers most of the lost retrieval quality at a fraction of the cost of making every layer dense. This is the dominant direction for genuinely long-context, cost-sensitive serving as of 2026 (cited in MiniMax's own related work as their design lineage, and matches what Qwen3.5's gated-linear-attention-hybrid approach is doing per the GQA-series source above).

---

## 8. KV-cache eviction, quantisation, and retrieval (adjacent optimisations)

Not attention mechanisms per se, but they operate on the same KV cache and are usually discussed in the same breath, worth knowing since you're already deep in KV cache dimension work for LLAMA-DX.

- **H2O (Heavy-Hitter Oracle)**: tracks a running importance score per cached token (roughly, cumulative attention received) and evicts low-importance entries once the cache fills, keeping a mix of "heavy hitters" plus recent tokens.
- **KIVI**: tuning-free asymmetric low-bit (2-bit) quantisation of the KV cache, quantising keys per-channel and values per-token (asymmetric because keys and values have different statistical structure) rather than one scheme for both.
- **RetrievalAttention / ClusterKV / MagicPIG**: treat the KV cache as a retrieval problem, cluster or index cached K/V offline and do approximate nearest-neighbour lookup per query instead of scanning linearly, closer to a vector-database approach than a masking scheme.

---

## 9. Other named mechanisms worth knowing (2025/2026 crop)

- **Interleaved Head Attention (IHA)**: builds "pseudo-heads" as combinations of all real heads, so cross-head interaction scales roughly quadratically with head count, aimed at improving multi-step reasoning by letting heads combine information rather than staying fully independent as in standard MHA.
- **Causal Attention with Lookahead Keys (CASTLE)**: a variant on causal attention where keys are updated as more tokens are processed, giving tokens indirect access to information from future context without breaking the autoregressive property outright, framed as a refinement of the strict causal mask.
- **Gated Attention**: applies an element-wise sigmoid gate to the attention output (or another intermediate), `Y' = Y ⊙ sigmoid(X W_gate)`, a cheap addition on top of any of the above aimed at training stability rather than efficiency.
- **DuoAttention**: rather than a new attention pattern, an optimisation method to *classify* which heads in a pretrained model are "retrieval heads" (need full attention) versus "streaming heads" (fine with sliding-window + sinks), then applies sparse computation only to the streaming heads. Relevant if you ever want to sparsify an existing GGUF model's attention without retraining it.
- **Mixture-of-Depths Attention**: routes tokens through only a subset of layers per forward pass (a MoE-style gate operating over depth rather than over expert FFNs), tangential to attention proper but frequently grouped with these efficiency techniques since it targets the same per-token compute budget.

---

## What's actually relevant to your stack

For **LLAMA-DX's** current decode-split problem: the fused flash-attention shader with strided HLSL indexing is squarely FlashAttention-family (section 2.1), online softmax plus tiling to eliminate the CPU/GPU handoffs causing the 40-split pattern. That's orthogonal to which of GQA/MLA a given GGUF model uses upstream, ggml already handles that at the graph level; your shader just needs to execute whatever K/V head-sharing topology the model file specifies efficiently.

For **VAiSt's** occupancy work, most of section 3 (GQA/MLA/GTA) and section 5 (learned sparsity) are software-level graph changes, not something a Vulkan compute kernel needs separate code paths for beyond correctly handling the reduced KV head count GQA implies, and possibly the extra reconstruction matmul MLA needs if you ever target a model that uses it (DeepSeek-distilled models, GLM, Kimi K2).

If you want, I can go deeper on any one of these with actual HLSL/GLSL kernel sketches, particularly the online-softmax tiling loop, since that's the one section directly on your critical path right now.
