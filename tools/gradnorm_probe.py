# Analysis only: gradient-norm statistics of the fixed flow loss at saved (end-of-training) weights, computed
# like train() does (batch 64 of training horses, bf16 autocast), to see how often clipping at 1.0 is active.
# Usage: python tools/gradnorm_probe.py exps/exp_NNN.py runs/<run>/weights/ema0.9995.pt [batches]
import runpy, sys, torch
ns = runpy.run_path(sys.argv[1], run_name="p6_probe")
model = ns["build_model"]().cuda()
model.load_state_dict(torch.load(sys.argv[2], map_location="cuda"))
model.train()
images, labels = ns["image_data"]()
torch.manual_seed(0)
norms = []
for _ in range(int(sys.argv[3]) if len(sys.argv) > 3 else 50):
    index = torch.randint(len(images), (64,))
    x, y = images[index].cuda(), labels[index].cuda()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = ns["flow_loss"](model, x, y)
    model.zero_grad(set_to_none=True)
    loss.backward()
    norms.append(torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf")).item())
norms = torch.tensor(norms)
print(f"grad norm over {len(norms)} batches: median {norms.median():.3f}, 90th pct {norms.quantile(0.9):.3f}, "
      f"max {norms.max():.3f}; fraction > 1.0: {(norms > 1).float().mean():.2f}", flush=True)
