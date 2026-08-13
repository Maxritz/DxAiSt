cmake_minimum_required(VERSION 3.20)
project(dxait CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (MSVC)
    add_compile_options(/O2 /Ot /Oi /fp:fast /W4 /WX-)
endif()

set(AGILITY_SDK_PATH "E:/DXllama/agility_temp/agility_1.720.0/build/native")
set(DXC_SDK_PATH "E:/DXllama/dxc-1.10.2605.2")
set(DSTORAGE_SDK_PATH "E:/DXllama/ds14/native")

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${AGILITY_SDK_PATH}/include
    ${DXC_SDK_PATH}/inc
    ${DSTORAGE_SDK_PATH}/include
)

link_directories(
    ${DXC_SDK_PATH}/lib/x64
    ${DSTORAGE_SDK_PATH}/lib/x64
)

set(DXAIT_SOURCES
    src/dxadapter/dxadapter.cpp
    src/dxruntime/dxruntime.cpp
    src/dxqueue/dxqueue.cpp
    src/dxmem/dxmem.cpp
    src/dxbarrier/dxbarrier.cpp
    src/dxio/dxio.cpp
    src/dxgraph/dxgraph.cpp
    src/dxtrace/dxtrace.cpp
    src/dxjit/dxjit.cpp
    src/dxblas/dxblas.cpp
    src/dxquant/dxquant.cpp
    src/dxkv/dxkv.cpp
    src/dxmath/dxmath.cpp
    src/dxmodel/dxmodel.cpp
    src/dxsched/dxsched.cpp
    src/dxrand/dxrand.cpp
    src/dxattention/dxattention.cpp
    src/dxfft/dxfft.cpp
    src/dxchunk/dxchunk.cpp
    src/dxshard/dxshard.cpp
    src/dxcache/dxcache.cpp
    src/dxcollective/dxcollective.cpp
    src/dxspeculative/dxspeculative.cpp
    src/dxcontext/dxcontext.cpp
    src/dxdb/dxdb.cpp
    src/dxmcp/dxmcp.cpp
    src/dxnetwork/dxnetwork.cpp
    src/dxtriton/dxtriton.cpp
)

add_library(dxait STATIC ${DXAIT_SOURCES})

target_link_libraries(dxait PRIVATE
    d3d12.lib
    dxgi.lib
    dxguid.lib
    dxcompiler.lib
    dstorage.lib
    Winmm.lib
    ws2_32.lib
)

enable_testing()

add_executable(test_device_init tests/test_device_init.cpp)
target_link_libraries(test_device_init PRIVATE dxait d3d12.lib dxgi.lib dxguid.lib dxcompiler.lib dstorage.lib Winmm.lib)
add_test(NAME test_device_init COMMAND test_device_init)

add_executable(test_mem_alloc tests/test_mem_alloc.cpp)
target_link_libraries(test_mem_alloc PRIVATE dxait d3d12.lib dxgi.lib dxguid.lib dxcompiler.lib dstorage.lib Winmm.lib)
add_test(NAME test_mem_alloc COMMAND test_mem_alloc)

add_executable(test_dxjit_blas tests/test_dxjit_blas.cpp)
target_link_libraries(test_dxjit_blas PRIVATE dxait d3d12.lib dxgi.lib dxguid.lib dxcompiler.lib dstorage.lib Winmm.lib)
add_test(NAME test_dxjit_blas COMMAND test_dxjit_blas)

add_executable(test_substrate_full tests/test_substrate_full.cpp)
target_link_libraries(test_substrate_full PRIVATE dxait d3d12.lib dxgi.lib dxguid.lib dxcompiler.lib dstorage.lib Winmm.lib)
add_test(NAME test_substrate_full COMMAND test_substrate_full)

add_executable(test_model_inference tests/test_model_inference.cpp)
target_link_libraries(test_model_inference PRIVATE dxait d3d12.lib dxgi.lib dxguid.lib dxcompiler.lib dstorage.lib Winmm.lib)
add_test(NAME test_model_inference COMMAND test_model_inference)

add_executable(test_attention_rand tests/test_attention_rand.cpp)
target_link_libraries(test_attention_rand PRIVATE dxait d3d12.lib dxgi.lib dxguid.lib dxcompiler.lib dstorage.lib Winmm.lib)
add_test(NAME test_attention_rand COMMAND test_attention_rand)

add_executable(test_q4_quant tests/test_q4_quant.cpp)
target_link_libraries(test_q4_quant PRIVATE dxait d3d12.lib dxgi.lib dxguid.lib dxcompiler.lib dstorage.lib Winmm.lib)
add_test(NAME test_q4_quant COMMAND test_q4_quant)

add_executable(test_model_formats tests/test_model_formats.cpp)
target_link_libraries(test_model_formats PRIVATE dxait d3d12.lib dxgi.lib dxguid.lib dxcompiler.lib dstorage.lib Winmm.lib)
add_test(NAME test_model_formats COMMAND test_model_formats)

add_executable(test_streaming_sharding tests/test_streaming_sharding.cpp)
target_link_libraries(test_streaming_sharding PRIVATE dxait d3d12.lib dxgi.lib dxguid.lib dxcompiler.lib dstorage.lib Winmm.lib)
add_test(NAME test_streaming_sharding COMMAND test_streaming_sharding)

add_executable(test_real_model_loading tests/test_real_model_loading.cpp)
target_link_libraries(test_real_model_loading PRIVATE dxait d3d12.lib dxgi.lib dxguid.lib dxcompiler.lib dstorage.lib Winmm.lib)
add_test(NAME test_real_model_loading COMMAND test_real_model_loading)

add_executable(test_large_gguf_spillover tests/test_large_gguf_spillover.cpp)
target_link_libraries(test_large_gguf_spillover PRIVATE dxait d3d12.lib dxgi.lib dxguid.lib dxcompiler.lib dstorage.lib Winmm.lib)
add_test(NAME test_large_gguf_spillover COMMAND test_large_gguf_spillover)

add_executable(test_50_50_offloading tests/test_50_50_offloading.cpp)
target_link_libraries(test_50_50_offloading PRIVATE dxait d3d12.lib dxgi.lib dxguid.lib dxcompiler.lib dstorage.lib Winmm.lib)
add_test(NAME test_50_50_offloading COMMAND test_50_50_offloading)

add_executable(test_sharding_perf tests/test_sharding_perf.cpp)
target_link_libraries(test_sharding_perf PRIVATE dxait d3d12.lib dxgi.lib dxguid.lib dxcompiler.lib dstorage.lib Winmm.lib)
add_test(NAME test_sharding_perf COMMAND test_sharding_perf)

add_executable(test_split_load_inference tests/test_split_load_inference.cpp)
target_link_libraries(test_split_load_inference PRIVATE dxait d3d12.lib dxgi.lib dxguid.lib dxcompiler.lib dstorage.lib Winmm.lib)
add_test(NAME test_split_load_inference COMMAND test_split_load_inference)

add_executable(test_chat_inference_harness tests/test_chat_inference_harness.cpp)
target_link_libraries(test_chat_inference_harness PRIVATE dxait d3d12.lib dxgi.lib dxguid.lib dxcompiler.lib dstorage.lib Winmm.lib)
add_test(NAME test_chat_inference_harness COMMAND test_chat_inference_harness)

add_executable(test_512k_rag_mcp tests/test_512k_rag_mcp.cpp)
target_link_libraries(test_512k_rag_mcp PRIVATE dxait d3d12.lib dxgi.lib dxguid.lib dxcompiler.lib dstorage.lib Winmm.lib ws2_32.lib)
add_test(NAME test_512k_rag_mcp COMMAND test_512k_rag_mcp)

add_executable(test_dxnetwork tests/test_dxnetwork.cpp)
target_link_libraries(test_dxnetwork PRIVATE dxait d3d12.lib dxgi.lib dxguid.lib dxcompiler.lib dstorage.lib Winmm.lib ws2_32.lib)
add_test(NAME test_dxnetwork COMMAND test_dxnetwork)

add_executable(test_dxtriton tests/test_dxtriton.cpp)
target_link_libraries(test_dxtriton PRIVATE dxait d3d12.lib dxgi.lib dxguid.lib dxcompiler.lib dstorage.lib Winmm.lib ws2_32.lib)
add_test(NAME test_dxtriton COMMAND test_dxtriton)

add_executable(test_dxfft tests/test_dxfft.cpp)
target_link_libraries(test_dxfft PRIVATE dxait d3d12.lib dxgi.lib dxguid.lib dxcompiler.lib dstorage.lib Winmm.lib ws2_32.lib)
add_test(NAME test_dxfft COMMAND test_dxfft)

add_executable(test_dxcache tests/test_dxcache.cpp)
target_link_libraries(test_dxcache PRIVATE dxait d3d12.lib dxgi.lib dxguid.lib dxcompiler.lib dstorage.lib Winmm.lib ws2_32.lib)
add_test(NAME test_dxcache COMMAND test_dxcache)

add_executable(test_dxspeculative tests/test_dxspeculative.cpp)
target_link_libraries(test_dxspeculative PRIVATE dxait d3d12.lib dxgi.lib dxguid.lib dxcompiler.lib dstorage.lib Winmm.lib ws2_32.lib)
add_test(NAME test_dxspeculative COMMAND test_dxspeculative)

add_executable(test_dxgraph tests/test_dxgraph.cpp)
target_link_libraries(test_dxgraph PRIVATE dxait d3d12.lib dxgi.lib dxguid.lib dxcompiler.lib dstorage.lib Winmm.lib ws2_32.lib)
add_test(NAME test_dxgraph COMMAND test_dxgraph)

add_executable(test_dxtrace tests/test_dxtrace.cpp)
target_link_libraries(test_dxtrace PRIVATE dxait d3d12.lib dxgi.lib dxguid.lib dxcompiler.lib dstorage.lib Winmm.lib ws2_32.lib)
add_test(NAME test_dxtrace COMMAND test_dxtrace)

add_executable(test_blas_gemm tests/test_blas_gemm.cpp)
target_link_libraries(test_blas_gemm PRIVATE dxait d3d12.lib dxgi.lib dxguid.lib dxcompiler.lib dstorage.lib Winmm.lib ws2_32.lib)
add_test(NAME test_blas_gemm COMMAND test_blas_gemm)

add_executable(test_attention_variants tests/test_attention_variants.cpp)
target_link_libraries(test_attention_variants PRIVATE dxait d3d12.lib dxgi.lib dxguid.lib dxcompiler.lib dstorage.lib Winmm.lib ws2_32.lib)
add_test(NAME test_attention_variants COMMAND test_attention_variants)

add_executable(test_attn_paged tests/test_attn_paged.cpp)
target_link_libraries(test_attn_paged PRIVATE dxait d3d12.lib dxgi.lib dxguid.lib dxcompiler.lib dstorage.lib Winmm.lib ws2_32.lib)
add_test(NAME test_attn_paged COMMAND test_attn_paged)

add_executable(test_attn_h2o tests/test_attn_h2o.cpp)
target_link_libraries(test_attn_h2o PRIVATE dxait d3d12.lib dxgi.lib dxguid.lib dxcompiler.lib dstorage.lib Winmm.lib ws2_32.lib)
add_test(NAME test_attn_h2o COMMAND test_attn_h2o)

add_executable(test_attn_chunked tests/test_attn_chunked.cpp)
target_link_libraries(test_attn_chunked PRIVATE dxait d3d12.lib dxgi.lib dxguid.lib dxcompiler.lib dstorage.lib Winmm.lib ws2_32.lib)
add_test(NAME test_attn_chunked COMMAND test_attn_chunked)

add_executable(test_attn_ring tests/test_attn_ring.cpp)
target_link_libraries(test_attn_ring PRIVATE dxait d3d12.lib dxgi.lib dxguid.lib dxcompiler.lib dstorage.lib Winmm.lib ws2_32.lib)
add_test(NAME test_attn_ring COMMAND test_attn_ring)

add_executable(test_attn_ring_multi tests/test_attn_ring_multi.cpp)
target_link_libraries(test_attn_ring_multi PRIVATE dxait d3d12.lib dxgi.lib dxguid.lib dxcompiler.lib dstorage.lib Winmm.lib ws2_32.lib)
add_test(NAME test_attn_ring_multi COMMAND test_attn_ring_multi)

add_executable(test_attn_gqa_iso tests/test_attn_gqa_iso.cpp)
target_link_libraries(test_attn_gqa_iso PRIVATE dxait d3d12.lib dxgi.lib dxguid.lib dxcompiler.lib dstorage.lib Winmm.lib ws2_32.lib)

# Tool targets
add_executable(dxinspect tools/dxinspect/main.cpp)
target_link_libraries(dxinspect PRIVATE dxait d3d12.lib dxgi.lib dxguid.lib dxcompiler.lib dstorage.lib Winmm.lib)

add_executable(dxbench tools/dxbench/main.cpp)
target_link_libraries(dxbench PRIVATE dxait d3d12.lib dxgi.lib dxguid.lib dxcompiler.lib dstorage.lib Winmm.lib)

add_executable(dxmodelinfo tools/dxmodelinfo/main.cpp)
target_link_libraries(dxmodelinfo PRIVATE dxait d3d12.lib dxgi.lib dxguid.lib dxcompiler.lib dstorage.lib Winmm.lib)

# Copy SDK DLLs alongside test and tool binaries
add_custom_command(TARGET test_device_init POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:test_device_init>/D3D12
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${AGILITY_SDK_PATH}/bin/x64/D3D12Core.dll" $<TARGET_FILE_DIR:test_device_init>/D3D12/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DXC_SDK_PATH}/bin/x64/dxcompiler.dll" $<TARGET_FILE_DIR:test_device_init>/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DXC_SDK_PATH}/bin/x64/dxil.dll" $<TARGET_FILE_DIR:test_device_init>/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DSTORAGE_SDK_PATH}/bin/x64/dstorage.dll" $<TARGET_FILE_DIR:test_device_init>/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DSTORAGE_SDK_PATH}/bin/x64/dstoragecore.dll" $<TARGET_FILE_DIR:test_device_init>/
)

add_custom_command(TARGET test_dxjit_blas POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:test_dxjit_blas>/D3D12
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${AGILITY_SDK_PATH}/bin/x64/D3D12Core.dll" $<TARGET_FILE_DIR:test_dxjit_blas>/D3D12/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DXC_SDK_PATH}/bin/x64/dxcompiler.dll" $<TARGET_FILE_DIR:test_dxjit_blas>/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DXC_SDK_PATH}/bin/x64/dxil.dll" $<TARGET_FILE_DIR:test_dxjit_blas>/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DSTORAGE_SDK_PATH}/bin/x64/dstorage.dll" $<TARGET_FILE_DIR:test_dxjit_blas>/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DSTORAGE_SDK_PATH}/bin/x64/dstoragecore.dll" $<TARGET_FILE_DIR:test_dxjit_blas>/
)

add_custom_command(TARGET test_substrate_full POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:test_substrate_full>/D3D12
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${AGILITY_SDK_PATH}/bin/x64/D3D12Core.dll" $<TARGET_FILE_DIR:test_substrate_full>/D3D12/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DXC_SDK_PATH}/bin/x64/dxcompiler.dll" $<TARGET_FILE_DIR:test_substrate_full>/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DXC_SDK_PATH}/bin/x64/dxil.dll" $<TARGET_FILE_DIR:test_substrate_full>/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DSTORAGE_SDK_PATH}/bin/x64/dstorage.dll" $<TARGET_FILE_DIR:test_substrate_full>/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DSTORAGE_SDK_PATH}/bin/x64/dstoragecore.dll" $<TARGET_FILE_DIR:test_substrate_full>/
)

add_custom_command(TARGET test_model_inference POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:test_model_inference>/D3D12
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${AGILITY_SDK_PATH}/bin/x64/D3D12Core.dll" $<TARGET_FILE_DIR:test_model_inference>/D3D12/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DXC_SDK_PATH}/bin/x64/dxcompiler.dll" $<TARGET_FILE_DIR:test_model_inference>/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DXC_SDK_PATH}/bin/x64/dxil.dll" $<TARGET_FILE_DIR:test_model_inference>/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DSTORAGE_SDK_PATH}/bin/x64/dstorage.dll" $<TARGET_FILE_DIR:test_model_inference>/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DSTORAGE_SDK_PATH}/bin/x64/dstoragecore.dll" $<TARGET_FILE_DIR:test_model_inference>/
)

add_custom_command(TARGET test_attention_rand POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:test_attention_rand>/D3D12
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${AGILITY_SDK_PATH}/bin/x64/D3D12Core.dll" $<TARGET_FILE_DIR:test_attention_rand>/D3D12/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DXC_SDK_PATH}/bin/x64/dxcompiler.dll" $<TARGET_FILE_DIR:test_attention_rand>/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DXC_SDK_PATH}/bin/x64/dxil.dll" $<TARGET_FILE_DIR:test_attention_rand>/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DSTORAGE_SDK_PATH}/bin/x64/dstorage.dll" $<TARGET_FILE_DIR:test_attention_rand>/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DSTORAGE_SDK_PATH}/bin/x64/dstoragecore.dll" $<TARGET_FILE_DIR:test_attention_rand>/
)

add_custom_command(TARGET test_q4_quant POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:test_q4_quant>/D3D12
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${AGILITY_SDK_PATH}/bin/x64/D3D12Core.dll" $<TARGET_FILE_DIR:test_q4_quant>/D3D12/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DXC_SDK_PATH}/bin/x64/dxcompiler.dll" $<TARGET_FILE_DIR:test_q4_quant>/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DXC_SDK_PATH}/bin/x64/dxil.dll" $<TARGET_FILE_DIR:test_q4_quant>/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DSTORAGE_SDK_PATH}/bin/x64/dstorage.dll" $<TARGET_FILE_DIR:test_q4_quant>/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DSTORAGE_SDK_PATH}/bin/x64/dstoragecore.dll" $<TARGET_FILE_DIR:test_q4_quant>/
)

add_custom_command(TARGET test_model_formats POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:test_model_formats>/D3D12
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${AGILITY_SDK_PATH}/bin/x64/D3D12Core.dll" $<TARGET_FILE_DIR:test_model_formats>/D3D12/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DXC_SDK_PATH}/bin/x64/dxcompiler.dll" $<TARGET_FILE_DIR:test_model_formats>/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DXC_SDK_PATH}/bin/x64/dxil.dll" $<TARGET_FILE_DIR:test_model_formats>/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DSTORAGE_SDK_PATH}/bin/x64/dstorage.dll" $<TARGET_FILE_DIR:test_model_formats>/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DSTORAGE_SDK_PATH}/bin/x64/dstoragecore.dll" $<TARGET_FILE_DIR:test_model_formats>/
)

add_custom_command(TARGET test_streaming_sharding POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:test_streaming_sharding>/D3D12
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${AGILITY_SDK_PATH}/bin/x64/D3D12Core.dll" $<TARGET_FILE_DIR:test_streaming_sharding>/D3D12/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DXC_SDK_PATH}/bin/x64/dxcompiler.dll" $<TARGET_FILE_DIR:test_streaming_sharding>/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DXC_SDK_PATH}/bin/x64/dxil.dll" $<TARGET_FILE_DIR:test_streaming_sharding>/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DSTORAGE_SDK_PATH}/bin/x64/dstorage.dll" $<TARGET_FILE_DIR:test_streaming_sharding>/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DSTORAGE_SDK_PATH}/bin/x64/dstoragecore.dll" $<TARGET_FILE_DIR:test_streaming_sharding>/
)

add_custom_command(TARGET test_real_model_loading POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:test_real_model_loading>/D3D12
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${AGILITY_SDK_PATH}/bin/x64/D3D12Core.dll" $<TARGET_FILE_DIR:test_real_model_loading>/D3D12/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DXC_SDK_PATH}/bin/x64/dxcompiler.dll" $<TARGET_FILE_DIR:test_real_model_loading>/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DXC_SDK_PATH}/bin/x64/dxil.dll" $<TARGET_FILE_DIR:test_real_model_loading>/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DSTORAGE_SDK_PATH}/bin/x64/dstorage.dll" $<TARGET_FILE_DIR:test_real_model_loading>/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DSTORAGE_SDK_PATH}/bin/x64/dstoragecore.dll" $<TARGET_FILE_DIR:test_real_model_loading>/
)

add_custom_command(TARGET test_large_gguf_spillover POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:test_large_gguf_spillover>/D3D12
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${AGILITY_SDK_PATH}/bin/x64/D3D12Core.dll" $<TARGET_FILE_DIR:test_large_gguf_spillover>/D3D12/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DXC_SDK_PATH}/bin/x64/dxcompiler.dll" $<TARGET_FILE_DIR:test_large_gguf_spillover>/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DXC_SDK_PATH}/bin/x64/dxil.dll" $<TARGET_FILE_DIR:test_large_gguf_spillover>/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DSTORAGE_SDK_PATH}/bin/x64/dstorage.dll" $<TARGET_FILE_DIR:test_large_gguf_spillover>/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DSTORAGE_SDK_PATH}/bin/x64/dstoragecore.dll" $<TARGET_FILE_DIR:test_large_gguf_spillover>/
)

add_custom_command(TARGET test_50_50_offloading POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:test_50_50_offloading>/D3D12
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${AGILITY_SDK_PATH}/bin/x64/D3D12Core.dll" $<TARGET_FILE_DIR:test_50_50_offloading>/D3D12/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DXC_SDK_PATH}/bin/x64/dxcompiler.dll" $<TARGET_FILE_DIR:test_50_50_offloading>/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DXC_SDK_PATH}/bin/x64/dxil.dll" $<TARGET_FILE_DIR:test_50_50_offloading>/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DSTORAGE_SDK_PATH}/bin/x64/dstorage.dll" $<TARGET_FILE_DIR:test_50_50_offloading>/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DSTORAGE_SDK_PATH}/bin/x64/dstoragecore.dll" $<TARGET_FILE_DIR:test_50_50_offloading>/
)

add_custom_command(TARGET dxinspect POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:dxinspect>/D3D12
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${AGILITY_SDK_PATH}/bin/x64/D3D12Core.dll" $<TARGET_FILE_DIR:dxinspect>/D3D12/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DXC_SDK_PATH}/bin/x64/dxcompiler.dll" $<TARGET_FILE_DIR:dxinspect>/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DXC_SDK_PATH}/bin/x64/dxil.dll" $<TARGET_FILE_DIR:dxinspect>/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DSTORAGE_SDK_PATH}/bin/x64/dstorage.dll" $<TARGET_FILE_DIR:dxinspect>/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DSTORAGE_SDK_PATH}/bin/x64/dstoragecore.dll" $<TARGET_FILE_DIR:dxinspect>/
)

add_custom_command(TARGET dxbench POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:dxbench>/D3D12
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${AGILITY_SDK_PATH}/bin/x64/D3D12Core.dll" $<TARGET_FILE_DIR:dxbench>/D3D12/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DXC_SDK_PATH}/bin/x64/dxcompiler.dll" $<TARGET_FILE_DIR:dxbench>/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DXC_SDK_PATH}/bin/x64/dxil.dll" $<TARGET_FILE_DIR:dxbench>/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DSTORAGE_SDK_PATH}/bin/x64/dstorage.dll" $<TARGET_FILE_DIR:dxbench>/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DSTORAGE_SDK_PATH}/bin/x64/dstoragecore.dll" $<TARGET_FILE_DIR:dxbench>/
)

add_custom_command(TARGET dxmodelinfo POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:dxmodelinfo>/D3D12
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${AGILITY_SDK_PATH}/bin/x64/D3D12Core.dll" $<TARGET_FILE_DIR:dxmodelinfo>/D3D12/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DXC_SDK_PATH}/bin/x64/dxcompiler.dll" $<TARGET_FILE_DIR:dxmodelinfo>/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DXC_SDK_PATH}/bin/x64/dxil.dll" $<TARGET_FILE_DIR:dxmodelinfo>/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DSTORAGE_SDK_PATH}/bin/x64/dstorage.dll" $<TARGET_FILE_DIR:dxmodelinfo>/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DSTORAGE_SDK_PATH}/bin/x64/dstoragecore.dll" $<TARGET_FILE_DIR:dxmodelinfo>/
)
