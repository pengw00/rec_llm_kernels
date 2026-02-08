import os
import subprocess
import sys
import torch  # <--- 必须添加，用于获取 Torch 的路径

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext 

class CMakeBuild(build_ext):
    def run(self):
        # 1. 获取项目根目录的绝对路径
        project_root = os.path.abspath(os.path.dirname(__file__))
        # 2. 获取编译临时目录的绝对路径
        build_temp = os.path.abspath(self.build_temp)
        # 3. 获取最终二进制输出目录的绝对路径
        # 注意：这里建议直接指向包内部目录
        output_dir = os.path.abspath("rec_llm_kernels")

        if not os.path.exists(build_temp):
            os.makedirs(build_temp)

        # --- 核心修复：获取 Torch 的 CMake 路径 ---
        torch_cmake_path = torch.utils.cmake_prefix_path
        print(f"--- Torch CMake Path: {torch_cmake_path} ---")

        print(f"--- Project Root: {project_root} ---")
        print(f"--- Build Temp Dir: {build_temp} ---")
        print(f"--- Output Dir: {output_dir} ---")

        use_flashinfer = "ON" if os.environ.get("USE_FLASHINFER") == "1" else "OFF"
        print(f"--- FlashInfer Option: {use_flashinfer} ---")
        
        # 运行 CMake 配置
        cmake_args = [
            "cmake", project_root, 
            f"-DUSE_FLASHINFER={use_flashinfer}",
            "-DCMAKE_CUDA_ARCHITECTURES=80",
            f"-DCMAKE_PREFIX_PATH={torch_cmake_path}",  # <--- 关键修复：强制注入 Torch 路径
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={output_dir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}" 
        ]
        
        # 清理旧的缓存重新配置（防止之前的失败干扰）
        subprocess.check_call(cmake_args, cwd=build_temp)
        
        # 运行编译，建议加上并行编译选项加快速度
        subprocess.check_call(["cmake", "--build", ".", "--", "-j8"], cwd=build_temp)

setup(
    name="rec_llm_kernels",
    # 这里的名字要和 CMakeLists.txt 里的 _C 对应
    ext_modules=[Extension("rec_llm_kernels._C", sources=[])],
    cmdclass={"build_ext": CMakeBuild},
    packages=['rec_llm_kernels']
)
