import torch
import rec_llm_kernels._C as ops

# Create some test data on GPU
q = torch.ones(10, 10).cuda()
out = torch.empty_like(q)

# Call YOUR custom kernel
ops.flash_att_forward(q, q, q, out)

print("Kernel Output Check (should be 2.0 if using q+k logic):", out[0][0].item())
print("Success! Your T4 custom kernel is alive.")