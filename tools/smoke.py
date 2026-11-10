# Smoke test: train an experiment for N updates (default 200), report updates/s, peak memory, sampling time,
# and check the extra-EMA state dicts load. Does not touch runs/ or index.jsonl.
import runpy, sys, time, torch
path, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 200
ns = runpy.run_path(path, run_name="p6_smoke")
ns["CONFIG"]["steps"] = n
stamps = []
ns["train"].__globals__["print"] = lambda *a, **k: (stamps.append(time.time()), print(*a, **k))
torch.manual_seed(3801); torch.backends.cudnn.benchmark = True
model = ns["build_model"]().cuda()
t0 = time.time(); hist = ns["train"](model, "cuda"); torch.cuda.synchronize(); t1 = time.time()
for d, sd in ns["EXTRA_EMA"].items():
    ns["build_model"]().load_state_dict(sd)
model.eval(); g = torch.Generator(device="cuda").manual_seed(0)
t2 = time.time(); ns["generate"](model, 50, "cuda", g); torch.cuda.synchronize(); t3 = time.time()
steady = (n - n // 10) / (stamps[-1] - stamps[1])  # from the first progress print after update 1
print(f"{path}: compile+first update {stamps[0]-t0:.0f}s; steady {steady:.2f} upd/s; {n} updates in {t1-t0:.1f}s (incl. compile) -> {n/(t1-t0):.2f} upd/s; "
      f"peak mem {torch.cuda.max_memory_allocated()/2**20:.0f} MiB; generate(50) {t3-t2:.1f}s; "
      f"extra EMAs {sorted(ns['EXTRA_EMA'])}; last loss {hist[-1]}", flush=True)
