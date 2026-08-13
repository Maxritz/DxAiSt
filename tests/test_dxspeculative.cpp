#include "dxait/dxait.hpp"
#include "dxait/dxspeculative.hpp"
#include <cstdio>
#include <cstring>
#include <cmath>

int main() {
    printf("DXAiT Speculative Decoding Verification Test\n");
    printf("=============================================\n\n");

    auto adapters = dxait::Adapter::enumerate();
    if (adapters.empty()) { printf("No GPU found, skipping test.\n"); return 0; }
    auto device = dxait::Adapter::create_device(0);

    constexpr uint32_t vocab = 8;
    constexpr uint32_t num_drafts = 4;
    constexpr uint64_t prob_bytes = vocab * sizeof(float);
    constexpr uint64_t tok_bytes = num_drafts * sizeof(uint32_t);

    // Case 1: target == draft -> ratio 1 -> always accept (r=0.5 <= 1)
    auto tgt = device->create_buffer(prob_bytes, dxait::MemLocation::Upload);
    auto drf = device->create_buffer(prob_bytes, dxait::MemLocation::Upload);
    auto toks = device->create_buffer(tok_bytes, dxait::MemLocation::Upload);
    auto mask = device->create_buffer(tok_bytes, dxait::MemLocation::Readback);

    float* tp = (float*)tgt->map();
    float* dp = (float*)drf->map();
    uint32_t* tk = (uint32_t*)toks->map();
    for (uint32_t i = 0; i < vocab; ++i) { tp[i] = 0.125f; dp[i] = 0.125f; }
    for (uint32_t i = 0; i < num_drafts; ++i) tk[i] = i % vocab;
    tgt->unmap(); drf->unmap(); toks->unmap();

    dxait::SpeculativeEngine engine(device.get());
    auto q = device->create_queue(dxait::QueueType::Compute);
    auto fence = device->create_fence(0);

    printf("Case 1: target == draft (r=0.5). Expect all accepted.\n");
    engine.verify_draft_tokens(q.get(), mask.get(), tgt.get(), drf.get(), toks.get(), num_drafts, vocab, 0.5f);
    q->signal(*fence, 1);
    fence->wait(1);

    uint32_t* m = (uint32_t*)mask->map();
    bool all_accept = true;
    for (uint32_t i = 0; i < num_drafts; ++i) {
        printf("  draft[%u]=%u accept_mask=%u\n", i, tk[i], m[i]);
        if (m[i] != 1) all_accept = false;
    }
    mask->unmap();

    // Case 2: target prob of draft token = 0 -> ratio 0 -> reject
    auto mask2 = device->create_buffer(tok_bytes, dxait::MemLocation::Readback);
    for (uint32_t i = 0; i < vocab; ++i) { tp[i] = 0.0f; }
    tgt->map(); // already mapped? no, remap
    float* tp2 = (float*)tgt->map();
    for (uint32_t i = 0; i < vocab; ++i) tp2[i] = (i == 0) ? 1.0f : 0.0f;
    tgt->unmap();

    printf("\nCase 2: target P(draft)=0. Expect all rejected.\n");
    engine.verify_draft_tokens(q.get(), mask2.get(), tgt.get(), drf.get(), toks.get(), num_drafts, vocab, 0.5f);
    q->signal(*fence, 2);
    fence->wait(2);

    uint32_t* m2 = (uint32_t*)mask2->map();
    bool all_reject = true;
    for (uint32_t i = 0; i < num_drafts; ++i) {
        uint32_t tok = tk[i];
        // draft i votes token i%vocab; only token 0 has target prob
        bool expect = (tok == 0);
        printf("  draft[%u]=%u accept_mask=%u (expect %u)\n", i, tok, m2[i], expect ? 1u : 0u);
        if ((m2[i] != 0) == expect) { /* reject check: want 0 when expect false */ }
        if (m2[i] != (expect ? 1u : 0u)) all_reject = false;
    }
    mask2->unmap();

    printf("\nResult: %s\n", (all_accept && all_reject) ? "Speculative verification PASSED" : "FAILED");
    return (all_accept && all_reject) ? 0 : 1;
}
