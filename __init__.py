import os
import importlib.util

# 环境变量：REC_PRECOMPILED_PATH=/path/to/your/so_file
precompiled_path = os.getenv("REC_PRECOMPILED_PATH")

if precompiled_path and os.path.exists(precompiled_path):
    # 动态加载指定的 .so 文件
    spec = importlib.util.spec_from_file_location("_C", precompiled_path)
    _C = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_C)
    print(f"Loaded precompiled kernel from {precompiled_path}")
else:
    # 正常加载模式
    try:
        from . import _C
    except ImportError:
        print("Kernel not found. Please compile or set REC_PRECOMPILED_PATH.")
