cmake_minimum_required(VERSION 3.18)
project(rec_llm_kernels LANGUAGES CXX CUDA)

# 定义一个选项，默认关闭（setup.py 会覆盖它）
option(USE_FLASHINFER "Enable FlashInfer integration" OFF)

if(USE_FLASHINFER)
    # 只有当 USE_FLASHINFER=ON 时，才会执行下载
    include(FetchContent)
    FetchContent_Declare(
        flashinfer
        GIT_REPOSITORY https://github.com/flashinfer-ai/flashinfer.git
        GIT_TAG        v0.1.6
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(flashinfer)
    
    # 将 FlashInfer 的 include 路径加入编译路径
    include_directories(${flashinfer_SOURCE_DIR}/include)
    message(STATUS "FlashInfer integration enabled.")

else()
    message(STATUS "FlashInfer integration disabled.")
endif()

# 1. 寻找 Python 开发库（新增）
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# 1. 获取 CMake 搜索路径 (Prefix Path)
execute_process(
    COMMAND python3 -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PATH}")

# 2. 获取 ABI 状态 (0 或 1)
execute_process(
    COMMAND python3 -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))" 
    OUTPUT_VARIABLE TORCH_ABI_VALUE  # 注意：这里改名为 ABI 专用变量
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS "[DEBUG] 探测到的 PyTorch ABI 值: ${TORCH_ABI_VALUE}")

# 3. 核心修复：将获取到的 1 传递给编译器
if("${TORCH_ABI_VALUE}" STREQUAL "")
    set(TORCH_ABI_VALUE "1") # 兜底逻辑：如果获取失败，Colab 环境通常为 1
endif()

# --- 3. 应用配置 ---
list(APPEND CMAKE_PREFIX_PATH "${TORCH_INSTALL_PATH}") # 使用路径变量
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=${TORCH_ABI_VALUE}) # 使用 ABI 变量

set(USE_KINETO OFF)
# --- 4. 正式查找 ---
find_package(Torch REQUIRED)

message(STATUS "-------------------------------------------")
message(STATUS "Check Torch ABI from CMake:")

# 尝试打印 Torch 自动带入的定义
get_target_property(TORCH_DEFS torch_cuda INTERFACE_COMPILE_DEFINITIONS)
message(STATUS "Torch CUDA Definitions: ${TORCH_DEFS}")

# 编译你的算子库
add_library(rec_llm_kernels_lib SHARED
    csrc/flash_att.cu
    csrc/paged_cache.cu
    csrc/binding.cpp)

target_compile_options(rec_llm_kernels_lib PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_80 --use_fast_math>
)

target_compile_definitions(rec_llm_kernels_lib PRIVATE 
    _GLIBCXX_USE_CXX11_ABI=${TORCH_ABI_VERSION}
)

set_target_properties(rec_llm_kernels_lib PROPERTIES PREFIX "" OUTPUT_NAME "_C")
target_include_directories(rec_llm_kernels_lib PRIVATE 
    ${Torch_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS} 
)
target_link_libraries(rec_llm_kernels_lib ${TORCH_LIBRARIES})
