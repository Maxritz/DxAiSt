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
    src/dxla/dxla.cpp
    src/dxstream/dxstream.cpp
    src/dxiocp/dxiocp.cpp
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

# Copy the SDK DLLs next to a target's binary (D3D12Core, dxcompiler, dxil, dstorage, dstoragecore).
function(copy_sdk_dlls target)
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${target}>/D3D12
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${AGILITY_SDK_PATH}/bin/x64/D3D12Core.dll" $<TARGET_FILE_DIR:${target}>/D3D12/
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DXC_SDK_PATH}/bin/x64/dxcompiler.dll" $<TARGET_FILE_DIR:${target}>/
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DXC_SDK_PATH}/bin/x64/dxil.dll" $<TARGET_FILE_DIR:${target}>/
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DSTORAGE_SDK_PATH}/bin/x64/dstorage.dll" $<TARGET_FILE_DIR:${target}>/
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DSTORAGE_SDK_PATH}/bin/x64/dstoragecore.dll" $<TARGET_FILE_DIR:${target}>/
    )
endfunction()

enable_testing()

set(TEST_TARGETS
    test_device_init
    test_mem_alloc
    test_dxjit_blas
    test_substrate_full
    test_model_inference
    test_attention_rand
    test_q4_quant
    test_model_formats
    test_streaming_sharding
    test_real_model_loading
    test_large_gguf_spillover
    test_50_50_offloading
    test_sharding_perf
    test_split_load_inference
    test_chat_inference_harness
    test_512k_rag_mcp
    test_dxnetwork
    test_dxtriton
    test_dxfft
    test_dxcache
    test_dxspeculative
    test_dxgraph
    test_dxtrace
    test_blas_gemm
    test_attention_variants
    test_attn_paged
    test_attn_h2o
    test_attn_chunked
    test_attn_ring
    test_attn_ring_multi
    test_dxla
    test_dstream
)

set(TOOL_TARGETS
    dxinspect
    dxbench
    dxai_ingest
)

foreach(t ${TEST_TARGETS} ${TOOL_TARGETS})
    if (t MATCHES "^test_")
        add_executable(${t} tests/${t}.cpp)
    else()
        add_executable(${t} tools/${t}/main.cpp)
    endif()
    target_link_libraries(${t} PRIVATE dxait)
    copy_sdk_dlls(${t})
    if (t MATCHES "^test_")
        add_test(NAME ${t} COMMAND ${t})
    endif()
endforeach()
