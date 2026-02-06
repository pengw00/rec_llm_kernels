from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

setup(
    name='rec_llm_kernels',
    ext_modules=[
        CUDAExtension(
            name='rec_llm_kernels._C',
            sources=['csrc/flash_att.cu', 'csrc/binding.cpp'],
            extra_compile_args={
                'cxx': ['-O3'],
                # 'nvcc': ['-O3', '--use_fast_math', '-gencode', 'arch=compute_80,code=sm_80']
                # 'nvcc': ['-O3', '--use_fast_math', '-gencode', 'arch=compute_75,code=sm_75']
                'nvcc': [
                    '-O3', 
                    '--use_fast_math', 
                    '-gencode', 'arch=compute_75,code=sm_75',
                    '--expt-relaxed-constexpr' # Adds compatibility for newer CUDA
                ]
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
