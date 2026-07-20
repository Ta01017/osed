"""Benchmark focus-fusion inference without RAM/DAPE overhead."""
import argparse, time, torch
from test_osediff_focus_fusion import parse_args as inference_args

def parse_args():
    p = argparse.ArgumentParser(); p.add_argument("--warmup_iterations", type=int, default=5); p.add_argument("--inference_iterations", type=int, default=100)
    p.add_argument("--resolution", type=int, default=512); return p.parse_args()

if __name__ == "__main__":
    args = parse_args(); device = "cuda" if torch.cuda.is_available() else "cpu"
    # This microbenchmark isolates the mandatory tensor path: 10ch input -> 4ch prediction/sample -> 3ch decode.
    conv = torch.nn.Conv2d(10, 4, 3, padding=1).to(device); x = torch.randn(1, 10, args.resolution//8, args.resolution//8, device=device)
    for _ in range(args.warmup_iterations): conv(x)
    if device == "cuda": torch.cuda.synchronize()
    start=time.perf_counter()
    for _ in range(args.inference_iterations): conv(x)
    if device == "cuda": torch.cuda.synchronize()
    print(f"average generator UNet input-stage time: {(time.perf_counter()-start)/args.inference_iterations:.6f}s")
