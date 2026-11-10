# Analysis only (nothing in any experiment or the final file depends on it): KID of REAL training horses
# against the validation split, through image_score() itself, i.e. the score a perfect memoriser would get.
# The "samples" are the first 1,000 images of image_data("train") (its fixed seed-3802 order), no flips,
# handed out 50 per generate() call exactly as image_score() requests them.
# Usage: python tools/real_floor.py exps/exp_011_flow_strong.py [offset]
import runpy, sys, tempfile, torch
from pathlib import Path
ns = runpy.run_path(sys.argv[1], run_name="p6_floor")
offset = int(sys.argv[2]) if len(sys.argv) > 2 else 0
train, _ = ns["image_data"]("train")
real = train[offset:offset + 1000]
calls = []
def generate(model, count, device, generator):  # same signature as the program's generate()
    start = 50 * len(calls); calls.append(count)
    return real[start:start + count].to(device)
ns["run"].__globals__["generate"] = generate  # image_score is torch.no_grad-wrapped: patch the real globals
kid = ns["image_score"](torch.nn.Identity(), Path(tempfile.mkdtemp()), "cuda", "validation")["kid"]
assert calls == [50] * 20, calls
print(f"real training horses {offset}..{offset + 999} vs validation: KID x1000 {1000 * kid:.2f}", flush=True)
