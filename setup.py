import os
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

class CMakeBuild(build_ext):
    def run(self):
        # 确保 build 目录存在
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        
        # 运行 CMake 配置
        # 环境变量开关：USE_FLASHINFER=1
        use_flashinfer = "ON" if os.getenv("USE_FLASHINFER") == "1" else "OFF"
        
        subprocess.check_call([
            "cmake", "..", 
            f"-DUSE_FLASHINFER={use_flashinfer}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={os.path.abspath('rec_llm_kernels')}"
        ], cwd=self.build_temp)
        
        # 运行编译
        subprocess.check_call(["cmake", "--build", "."], cwd=self.build_temp)

setup(
    name="rec_llm_kernels",
    ext_modules=[Extension("rec_llm_kernels._C", sources=[])], # 占位符
    cmdclass={"build_ext": CMakeBuild},
)
