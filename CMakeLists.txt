cmake_minimum_required(VERSION 3.18)
project(rec_llm_kernels LANGUAGES CXX CUDA)

# 模仿 vLLM：自动下载 FlashInfer
include(FetchContent)
FetchContent_Declare(
    flashinfer
    GIT_REPOSITORY https://github.com
    GIT_TAG        v0.1.6  # 锁定版本，防止意外
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(flashinfer)

# 设置包含路径
include_directories(${flashinfer_SOURCE_DIR}/include)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# 编译你的算子库
add_library(rec_llm_kernels_lib SHARED csrc/flash_att.cu csrc/binding.cpp)
target_link_libraries(rec_llm_kernels_lib torch c10)
