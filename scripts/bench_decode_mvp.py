import argparse
import time

import torch

from rec_llm_runtime.runtime.decode_mvp import DecodeMVPConfig, DecodeOnlyMVP
from rec_llm_runtime.runtime.kv_cache import PagedKVCache


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--num-heads", type=int, default=32)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--num-blocks", type=int, default=4096)
    ap.add_argument("--max-blocks", type=int, default=256)
    args = ap.parse_args()

    device = torch.device(args.device)
    hidden = args.num_heads * args.head_dim

    model = DecodeOnlyMVP(DecodeMVPConfig(hidden_size=hidden, num_heads=args.num_heads, head_dim=args.head_dim)).to(device).eval()
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = model.to(dtype=dtype)

    cache = PagedKVCache.allocate(
        batch_size=args.batch,
        num_blocks=args.num_blocks,
        max_blocks=args.max_blocks,
        num_heads=args.num_heads,
        block_size=args.block_size,
        head_dim=args.head_dim,
        device=device,
        dtype=dtype,
    )

    x = torch.randn(args.batch, hidden, device=device, dtype=dtype)
    positions = torch.zeros(args.batch, device=device, dtype=torch.int64)

    # Warmup (also triggers compile / extension load)
    with torch.inference_mode():
        for _ in range(5):
            _ = model(x, positions, cache)
            positions += 1
    if device.type == "cuda":
        torch.cuda.synchronize()

    latencies = []
    tokens = args.batch * args.steps
    with torch.inference_mode():
        for _ in range(args.steps):
            t0 = time.perf_counter()
            _ = model(x, positions, cache)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)
            positions += 1

    latencies.sort()
    p50 = latencies[int(0.50 * len(latencies))]
    p95 = latencies[int(0.95 * len(latencies))]
    total_s = sum(latencies) / 1000.0
    toks_per_s = tokens / total_s if total_s > 0 else float("inf")

    mem_mb = None
    if device.type == "cuda":
        mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    print(f"tokens={tokens} steps={args.steps} batch={args.batch} device={device}")
    print(f"latency_ms p50={p50:.3f} p95={p95:.3f}")
    print(f"throughput tokens/s={toks_per_s:.1f}")
    if mem_mb is not None:
        print(f"cuda max_memory_allocated_mb={mem_mb:.1f}")


if __name__ == "__main__":
    main()
