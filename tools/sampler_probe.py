# Analysis only (never part of an experiment or the final file): is the fixed 25-step Euler sampler the
# bottleneck? Re-samples saved weights with Euler at 25 / 50 / 100 steps (same noise seed and KID code as
# image_score) and prints KID x1000 for each.
# Usage: python tools/sampler_probe.py exps/exp_NNN.py runs/<run>/weights/ema0.9995.pt [steps ...]
import runpy, sys, torch
ns = runpy.run_path(sys.argv[1], run_name="p6_probe")
model = ns["build_model"]().cuda()
model.load_state_dict(torch.load(sys.argv[2], map_location="cuda"))
model.eval()
for steps in [int(a) for a in sys.argv[3:]] or [25, 50, 100]:
    ns["generate"] = lambda m, n, device, generator: ns["sample_flow"](
        m, torch.zeros(n, dtype=torch.long, device=device), steps, device, generator)
    folder = __import__("pathlib").Path(f"/tmp/probe_{steps}"); folder.mkdir(exist_ok=True)
    kid = ns["image_score"](model, folder, "cuda", "validation")["kid"]
    print(f"Euler {steps:3d} steps: KID x1000 {1000 * kid:.2f}", flush=True)
