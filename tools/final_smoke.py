# Smoke test of a final program's train(): N updates (default 100), checks the single-EMA path end to end.
# Records every EMA update, checks its weights are max(1 - EMA_DECAY, 1/(step+1)), that the model ends up holding
# exactly the EMA (not the raw weights), and that generate() gives finite images. Touches no runs/ or index files.
import runpy, sys, torch
path, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 100
device = sys.argv[3] if len(sys.argv) > 3 else ("cuda" if torch.cuda.is_available() else "cpu")
ns = runpy.run_path(path, run_name="p6_final_smoke")
g = ns["run"].__globals__
g["CONFIG"]["steps"] = n
torch.manual_seed(3801)
model = ns["build_model"]().to(device)
calls, real_lerp = [], torch._foreach_lerp_
model_params = list(model.parameters())
def lerp(ema, params, weight):  # AdamW's foreach path also lerps (its first moment, towards the gradients)
    if len(params) == len(model_params) and all(a is b for a, b in zip(params, model_params)):
        calls.append((ema, weight, [p.detach().clone() for p in params]))
    return real_lerp(ema, params, weight)
torch._foreach_lerp_ = lerp
history = ns["train"](model, device)
torch._foreach_lerp_ = real_lerp
decay = g["EMA_DECAY"]
assert len(calls) == n, len(calls)
assert all(abs(w - max(1 - decay, 1 / (s + 1))) < 1e-12 for s, (_, w, _) in enumerate(calls, 1)), "EMA weights"
assert len({id(c[0]) for c in calls}) == 1, "one EMA list"
ema, raw = calls[-1][0], calls[-1][2]
params = list(model.parameters())
assert all(torch.equal(p, e) for p, e in zip(params, ema)), "model does not hold the EMA"
assert any(not torch.equal(p, r) for p, r in zip(params, raw)), "model holds the raw weights"
model.eval()
x = ns["generate"](model, 4, device, torch.Generator(device=device).manual_seed(0))
assert x.shape == (4, 3, 32, 32) and torch.isfinite(x).all()
print(f"{path}: {n} updates on {device} OK; EMA weights 1/(step+1) up to {calls[-1][1]:.4f}; model holds the EMA; "
      f"losses {history[0][1]:.3f} -> {history[-1][1]:.3f}; generate() finite", flush=True)
