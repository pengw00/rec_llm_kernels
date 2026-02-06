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

# 2. 寻找 PyTorch 的依赖路径（这部分你已经有了）
execute_process(COMMAND python3 -c "import torch; print(torch.utils.cmake_prefix_path)" 
                OUTPUT_VARIABLE TORCH_CMAKE_PATH 
                OUTPUT_STRIP_TRAILING_WHITESPACE)
list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PATH})
find_package(Torch REQUIRED)

# 编译你的算子库
add_library(rec_llm_kernels_lib SHARED csrc/flash_att.cu csrc/binding.cpp)
set_target_properties(rec_llm_kernels_lib PROPERTIES PREFIX "" OUTPUT_NAME "_C")
target_include_directories(rec_llm_kernels_lib PRIVATE 
    ${Torch_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS} 
)
target_link_libraries(rec_llm_kernels_lib ${TORCH_LIBRARIES})
