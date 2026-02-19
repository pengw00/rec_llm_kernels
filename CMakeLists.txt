cmake_minimum_required(VERSION 3.18)
project(rec_llm_kernels LANGUAGES CXX CUDA)

# 定义一个选项，默认关闭（setup.py 会覆盖它）
option(USE_FLASHINFER "Enable FlashInfer integration" OFF)

set(FLASHINFER_SOURCE_DIR "" CACHE PATH "Path to a local FlashInfer checkout (skips network fetch).")
option(FLASHINFER_FETCH "Allow CMake to fetch FlashInfer via FetchContent" OFF)

set(FLASHINFER_INCLUDE_DIR "")
if(USE_FLASHINFER)
    if(FLASHINFER_SOURCE_DIR)
        set(FLASHINFER_INCLUDE_DIR "${FLASHINFER_SOURCE_DIR}/include")
        message(STATUS "FlashInfer integration enabled (local): ${FLASHINFER_INCLUDE_DIR}")
    else()
        # 默认不触发在线下载；需要显式开启 FLASHINFER_FETCH
        if(FETCHCONTENT_FULLY_DISCONNECTED OR NOT FLASHINFER_FETCH)
            message(FATAL_ERROR
                "USE_FLASHINFER=ON but no local FlashInfer checkout provided.\n"
                "Provide -DFLASHINFER_SOURCE_DIR=/path/to/flashinfer, or set -DUSE_FLASHINFER=OFF.\n"
                "If you really want CMake to download dependencies, pass -DFLASHINFER_FETCH=ON "
                "(requires network access).")
        endif()

        include(FetchContent)
        FetchContent_Declare(
            flashinfer
            GIT_REPOSITORY https://github.com/flashinfer-ai/flashinfer.git
            GIT_TAG        v0.1.6
            GIT_SHALLOW    TRUE
        )
        FetchContent_MakeAvailable(flashinfer)
        set(FLASHINFER_INCLUDE_DIR "${flashinfer_SOURCE_DIR}/include")
        message(STATUS "FlashInfer integration enabled (fetched): ${FLASHINFER_INCLUDE_DIR}")
    endif()

else()
    message(STATUS "FlashInfer integration disabled.")
endif()

# 1. 寻找 Python 开发库（新增）
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# 1. 获取 CMake 搜索路径 (Prefix Path)
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    RESULT_VARIABLE TORCH_CMAKE_RESULT
    OUTPUT_VARIABLE TORCH_CMAKE_PATH
    ERROR_VARIABLE TORCH_CMAKE_ERROR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT TORCH_CMAKE_RESULT EQUAL 0 OR "${TORCH_CMAKE_PATH}" STREQUAL "")
    message(FATAL_ERROR
        "Failed to query PyTorch CMake prefix path using: ${Python3_EXECUTABLE}\n"
        "Make sure PyTorch is installed in this Python environment.\n"
        "Error:\n${TORCH_CMAKE_ERROR}")
endif()
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PATH}")

# 2. 获取 ABI 状态 (0 或 1)
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))"
    RESULT_VARIABLE TORCH_ABI_RESULT
    OUTPUT_VARIABLE TORCH_ABI_VALUE  # 注意：这里改名为 ABI 专用变量
    ERROR_VARIABLE TORCH_ABI_ERROR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT TORCH_ABI_RESULT EQUAL 0 OR "${TORCH_ABI_VALUE}" STREQUAL "")
    message(FATAL_ERROR
        "Failed to query PyTorch ABI flag using: ${Python3_EXECUTABLE}\n"
        "Make sure PyTorch is installed in this Python environment.\n"
        "Error:\n${TORCH_ABI_ERROR}")
endif()

message(STATUS "[DEBUG] 探测到的 PyTorch ABI 值: ${TORCH_ABI_VALUE}")

# 3. 核心修复：将获取到的 1 传递给编译器
if("${TORCH_ABI_VALUE}" STREQUAL "")
    set(TORCH_ABI_VALUE "1") # 兜底逻辑：如果获取失败，Colab 环境通常为 1
endif()

# --- 3. 应用配置 ---
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=${TORCH_ABI_VALUE}) # 使用 ABI 变量

set(USE_KINETO OFF)
# --- 4. 正式查找 ---
find_package(Torch REQUIRED)

message(STATUS "-------------------------------------------")
message(STATUS "Check Torch ABI from CMake:")

# 尝试打印 Torch 自动带入的定义
if(TARGET torch_cuda)
    get_target_property(TORCH_DEFS torch_cuda INTERFACE_COMPILE_DEFINITIONS)
    message(STATUS "Torch CUDA Definitions: ${TORCH_DEFS}")
else()
    message(STATUS "Torch CUDA target not found (CPU-only Torch or different target names).")
endif()

# 编译你的算子库
add_library(rec_llm_kernels_lib SHARED
    csrc/flash_att.cu
    csrc/paged_cache.cu
    csrc/backend/rms_norm_kernel.cu
    csrc/binding.cpp)

target_compile_options(rec_llm_kernels_lib PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_80 --use_fast_math>
)

set_target_properties(rec_llm_kernels_lib PROPERTIES PREFIX "" OUTPUT_NAME "_C")
target_include_directories(rec_llm_kernels_lib PRIVATE 
    ${Torch_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS} 
    $<$<BOOL:${FLASHINFER_INCLUDE_DIR}>:${FLASHINFER_INCLUDE_DIR}>
)
target_link_libraries(rec_llm_kernels_lib ${TORCH_LIBRARIES})
