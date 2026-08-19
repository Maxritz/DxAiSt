#include "dxait/dxait.hpp"
#include "dxait/dxla.hpp"
#include "dxait/dx_c_api.h"

#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <string>

using namespace dxait;

struct dx_device_s {
    std::unique_ptr<Device> dev;
    std::unique_ptr<LA> la;
    std::unique_ptr<Queue> queue;
    std::unique_ptr<Fence> fence;
    uint64_t fv = 0;
    ComPtr<ID3D12CommandAllocator> copy_alloc;
    ComPtr<ID3D12GraphicsCommandList> copy_list;
};

struct dx_queue_s {
    Queue* q;
};

struct dx_buffer_s {
    std::unique_ptr<Buffer> b;
    int loc; // 1=Default 2=Upload 3=Readback
};

namespace {

thread_local std::string g_last_error;

void set_error(const std::string& msg) {
    g_last_error = msg;
}

const char* format_hr(HRESULT hr) {
    char buf[256] = {};
    DWORD n = FormatMessageA(
        FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr,
        static_cast<DWORD>(hr), 0, buf, sizeof(buf), nullptr);
    while (n > 0 && (buf[n - 1] == '\n' || buf[n - 1] == '\r')) buf[--n] = 0;
    char tmp[320];
    snprintf(tmp, sizeof(tmp), "HRESULT 0x%08lX: %s",
             static_cast<unsigned long>(hr), n ? buf : "unknown error");
    set_error(tmp);
    return g_last_error.c_str();
}

HRESULT run_try(std::function<HRESULT()> fn) {
    try {
        return fn();
    } catch (const std::exception& e) {
        set_error(e.what());
        return E_FAIL;
    } catch (...) {
        set_error("unknown exception");
        return E_FAIL;
    }
}

HRESULT submit_wait(dx_device_s* d) {
    d->queue->signal(*d->fence, ++d->fv);
    d->fence->wait(d->fv);
    return S_OK;
}

// Default-heap copy via a tiny compute command list (CopyBufferRegion is valid
// on compute lists). Follows the repo pattern: no explicit barriers.
HRESULT do_copy(dx_device_s* d, ID3D12Resource* src, uint64_t soff,
                ID3D12Resource* dst, uint64_t doff, uint64_t bytes) {
    if (!d->copy_alloc) {
        ID3D12Device* dev = d->dev->get();
        HRESULT hr = dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE,
                                                 IID_PPV_ARGS(&d->copy_alloc));
        if (FAILED(hr)) return hr;
        hr = dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE,
                                    d->copy_alloc.Get(), nullptr,
                                    IID_PPV_ARGS(&d->copy_list));
        if (FAILED(hr)) return hr;
        d->copy_list->Close();
    }
    HRESULT hr = d->copy_alloc->Reset();
    if (FAILED(hr)) return hr;
    hr = d->copy_list->Reset(d->copy_alloc.Get(), nullptr);
    if (FAILED(hr)) return hr;
    d->copy_list->CopyBufferRegion(dst, doff, src, soff, bytes);
    d->copy_list->Close();
    ID3D12CommandList* lists[] = { d->copy_list.Get() };
    d->queue->execute(lists, 1);
    return submit_wait(d);
}

} // namespace

extern "C" {

HRESULT dx_create_device(uint32_t index, dx_device** out) {
    return run_try([&]() -> HRESULT {
        if (!out) return E_INVALIDARG;
        auto* d = new dx_device_s();
        d->dev = Adapter::create_device(index);
        if (!d->dev) { delete d; set_error("create_device failed"); return E_FAIL; }
        d->la = std::make_unique<LA>(d->dev.get());
        d->queue = d->dev->create_queue(QueueType::Compute);
        d->fence = d->dev->create_fence(0);
        *out = d;
        return S_OK;
    });
}

void dx_destroy_device(dx_device* dev) {
    if (!dev) return;
    delete static_cast<dx_device_s*>(dev);
}

HRESULT dx_device_queue(dx_device* dev, dx_queue** out) {
    return run_try([&]() -> HRESULT {
        auto* d = static_cast<dx_device_s*>(dev);
        if (!d || !out) return E_INVALIDARG;
        auto* q = new dx_queue_s{ d->queue.get() };
        *out = q;
        return S_OK;
    });
}

void dx_destroy_queue(dx_queue* q) {
    delete static_cast<dx_queue_s*>(q); // handle only; queue owned by device
}

HRESULT dx_create_buffer(dx_device* dev, uint64_t bytes, int loc, dx_buffer** out) {
    return run_try([&]() -> HRESULT {
        auto* d = static_cast<dx_device_s*>(dev);
        if (!d || !out) return E_INVALIDARG;
        if (loc < 1 || loc > 3) return E_INVALIDARG;
        auto* b = new dx_buffer_s();
        b->loc = loc;
        b->b = d->dev->create_buffer(bytes, static_cast<MemLocation>(loc));
        if (!b->b) { delete b; set_error("create_buffer failed"); return E_FAIL; }
        *out = b;
        return S_OK;
    });
}

void* dx_buffer_map(dx_buffer* b) {
    auto* B = static_cast<dx_buffer_s*>(b);
    if (!B) return nullptr;
    return B->b->map();
}

void dx_buffer_unmap(dx_buffer* b) {
    auto* B = static_cast<dx_buffer_s*>(b);
    if (!B) return;
    B->b->unmap();
}

void dx_destroy_buffer(dx_buffer* b) {
    delete static_cast<dx_buffer_s*>(b);
}

HRESULT dx_upload(dx_device* dev, dx_buffer* dst, uint64_t off, const void* src, uint64_t bytes) {
    return run_try([&]() -> HRESULT {
        auto* d = static_cast<dx_device_s*>(dev);
        auto* B = static_cast<dx_buffer_s*>(dst);
        if (!d || !B || !src) return E_INVALIDARG;
        if (B->loc == 3) return E_INVALIDARG; // cannot upload into a readback buffer
        if (B->loc == 2) { // Upload: direct map + copy
            void* p = B->b->map();
            if (!p) return E_FAIL;
            std::memcpy(static_cast<char*>(p) + off, src, bytes);
            B->b->unmap();
            return S_OK;
        }
        // Default: staging upload buffer + GPU copy
        auto staging = d->dev->create_buffer(bytes, MemLocation::Upload);
        void* p = staging->map();
        if (!p) return E_FAIL;
        std::memcpy(p, src, bytes);
        staging->unmap();
        return do_copy(d, staging->get(), 0, B->b->get(), off, bytes);
    });
}

HRESULT dx_download(dx_device* dev, dx_buffer* src, uint64_t off, void* dst, uint64_t bytes) {
    return run_try([&]() -> HRESULT {
        auto* d = static_cast<dx_device_s*>(dev);
        auto* B = static_cast<dx_buffer_s*>(src);
        if (!d || !B || !dst) return E_INVALIDARG;
        if (B->loc == 2) return E_INVALIDARG; // cannot download from an upload buffer
        if (B->loc == 3) { // Readback: direct map + copy
            void* p = B->b->map();
            if (!p) return E_FAIL;
            std::memcpy(dst, static_cast<const char*>(p) + off, bytes);
            B->b->unmap();
            return S_OK;
        }
        // Default: staging readback buffer + GPU copy
        auto staging = d->dev->create_buffer(bytes, MemLocation::Readback);
        HRESULT hr = do_copy(d, B->b->get(), off, staging->get(), 0, bytes);
        if (FAILED(hr)) return hr;
        void* p = staging->map();
        if (!p) return E_FAIL;
        std::memcpy(dst, p, bytes);
        staging->unmap();
        return S_OK;
    });
}

HRESULT dx_wait(dx_device* dev) {
    return run_try([&]() -> HRESULT {
        auto* d = static_cast<dx_device_s*>(dev);
        if (!d) return E_INVALIDARG;
        return submit_wait(d);
    });
}

HRESULT dx_device_desc(dx_device* dev, char* buf, int cap) {
    return run_try([&]() -> HRESULT {
        auto* d = static_cast<dx_device_s*>(dev);
        if (!d || !buf || cap <= 0) return E_INVALIDARG;
        const std::string& name = d->dev->caps().name;
        std::memcpy(buf, name.c_str(), static_cast<size_t>(cap) - 1 < name.size()
            ? static_cast<size_t>(cap) - 1 : name.size());
        buf[(cap < name.size() ? cap : static_cast<int>(name.size()))] = 0;
        return S_OK;
    });
}

const char* dx_last_error(void) {
    return g_last_error.c_str();
}

HRESULT dx_la_elementwise(dx_device* dev, dx_queue* q, dx_buffer* out, dx_buffer* a, dx_buffer* b, uint32_t count, int op, float alpha, float beta) {
    return run_try([&]() -> HRESULT {
        auto* d = static_cast<dx_device_s*>(dev);
        auto* Q = static_cast<dx_queue_s*>(q);
        auto* O = static_cast<dx_buffer_s*>(out);
        auto* A = static_cast<dx_buffer_s*>(a);
        auto* B = static_cast<dx_buffer_s*>(b);
        if (!d || !Q || !O || !A || !B) return E_INVALIDARG;
        d->la->elementwise(Q->q, O->b.get(), A->b.get(), B->b.get(), count,
                           static_cast<LAOp>(op), alpha, beta);
        return submit_wait(d);
    });
}

HRESULT dx_la_activation(dx_device* dev, dx_queue* q, dx_buffer* out, dx_buffer* in, uint32_t count, int act, float alpha) {
    return run_try([&]() -> HRESULT {
        auto* d = static_cast<dx_device_s*>(dev);
        auto* Q = static_cast<dx_queue_s*>(q);
        auto* O = static_cast<dx_buffer_s*>(out);
        auto* I_ = static_cast<dx_buffer_s*>(in);
        if (!d || !Q || !O || !I_) return E_INVALIDARG;
        d->la->activation(Q->q, O->b.get(), I_->b.get(), count,
                          static_cast<LAActivation>(act), alpha);
        return submit_wait(d);
    });
}

HRESULT dx_la_rmsnorm(dx_device* dev, dx_queue* q, dx_buffer* out, dx_buffer* in, dx_buffer* gamma, uint32_t rows, uint32_t dim, float eps) {
    return run_try([&]() -> HRESULT {
        auto* d = static_cast<dx_device_s*>(dev);
        auto* Q = static_cast<dx_queue_s*>(q);
        auto* O = static_cast<dx_buffer_s*>(out);
        auto* I_ = static_cast<dx_buffer_s*>(in);
        auto* G = static_cast<dx_buffer_s*>(gamma);
        if (!d || !Q || !O || !I_ || !G) return E_INVALIDARG;
        d->la->rmsnorm(Q->q, O->b.get(), I_->b.get(), G->b.get(), rows, dim, eps);
        return submit_wait(d);
    });
}

HRESULT dx_la_softmax(dx_device* dev, dx_queue* q, dx_buffer* out, dx_buffer* in, uint32_t rows, uint32_t dim) {
    return run_try([&]() -> HRESULT {
        auto* d = static_cast<dx_device_s*>(dev);
        auto* Q = static_cast<dx_queue_s*>(q);
        auto* O = static_cast<dx_buffer_s*>(out);
        auto* I_ = static_cast<dx_buffer_s*>(in);
        if (!d || !Q || !O || !I_) return E_INVALIDARG;
        d->la->softmax(Q->q, O->b.get(), I_->b.get(), rows, dim);
        return submit_wait(d);
    });
}

HRESULT dx_la_reduce(dx_device* dev, dx_queue* q, dx_buffer* out, dx_buffer* in, uint32_t rows, uint32_t dim, int op) {
    return run_try([&]() -> HRESULT {
        auto* d = static_cast<dx_device_s*>(dev);
        auto* Q = static_cast<dx_queue_s*>(q);
        auto* O = static_cast<dx_buffer_s*>(out);
        auto* I_ = static_cast<dx_buffer_s*>(in);
        if (!d || !Q || !O || !I_) return E_INVALIDARG;
        d->la->reduce(Q->q, O->b.get(), I_->b.get(), rows, dim,
                      static_cast<LAReduce>(op));
        return submit_wait(d);
    });
}

HRESULT dx_la_gemm_f16_dot2(dx_device* dev, dx_queue* q, dx_buffer* out, dx_buffer* a, dx_buffer* b, uint32_t M, uint32_t N, uint32_t K) {
    return run_try([&]() -> HRESULT {
        auto* d = static_cast<dx_device_s*>(dev);
        auto* Q = static_cast<dx_queue_s*>(q);
        auto* O = static_cast<dx_buffer_s*>(out);
        auto* A = static_cast<dx_buffer_s*>(a);
        auto* B = static_cast<dx_buffer_s*>(b);
        if (!d || !Q || !O || !A || !B) return E_INVALIDARG;
        d->la->gemm_f16_dot2(Q->q, O->b.get(), A->b.get(), B->b.get(), M, N, K);
        return submit_wait(d);
    });
}

HRESULT dx_la_gemm_f16_wmma(dx_device* dev, dx_queue* q, dx_buffer* out, dx_buffer* a, dx_buffer* b, uint32_t M, uint32_t N, uint32_t K) {
    return run_try([&]() -> HRESULT {
        auto* d = static_cast<dx_device_s*>(dev);
        auto* Q = static_cast<dx_queue_s*>(q);
        auto* O = static_cast<dx_buffer_s*>(out);
        auto* A = static_cast<dx_buffer_s*>(a);
        auto* B = static_cast<dx_buffer_s*>(b);
        if (!d || !Q || !O || !A || !B) return E_INVALIDARG;
        d->la->gemm_f16_wmma(Q->q, O->b.get(), A->b.get(), B->b.get(), M, N, K);
        return submit_wait(d);
    });
}

} // extern "C"
