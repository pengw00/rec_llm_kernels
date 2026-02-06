import os
import subprocess
import sys # 需要引入 sys

class CMakeBuild(build_ext):
    def run(self):
        # 1. 获取项目根目录的绝对路径
        project_root = os.path.abspath(os.path.dirname(__file__))
        # 2. 获取编译临时目录的绝对路径
        build_temp = os.path.abspath(self.build_temp)
        # 3. 获取最终二进制输出目录的绝对路径
        output_dir = os.path.abspath("rec_llm_kernels")

        if not os.path.exists(build_temp):
            os.makedirs(build_temp)

        print(f"--- Project Root: {project_root} ---")
        print(f"--- Build Temp Dir: {build_temp} ---")
        print(f"--- Output Dir: {output_dir} ---")

        # 运行 CMake 配置，传入 project_root 作为源码目录
        cmake_args = [
            "cmake", project_root, 
            "-DUSE_FLASHINFER=OFF", # 或者从环境变量读取
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={output_dir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}" # 确保 CMake 知道用哪个 Python
        ]
        
        subprocess.check_call(cmake_args, cwd=build_temp)
        
        # 运行编译
        subprocess.check_call(["cmake", "--build", "."], cwd=build_temp)

