import torch
import rec_llm_kernels._C as ops
q = torch.randn(10, 10).cuda()
out = torch.empty_like(q)
ops.flash_att_forward(q, q, q, out) # 跑通就说明 Wheel 核心成了
print("CUDA Kernel 运行成功！")