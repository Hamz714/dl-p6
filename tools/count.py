# Count parameters + buffers of an experiment program's build_model() and run a forward/backward smoke test.
import runpy, sys, torch
for path in sys.argv[1:]:
    ns = runpy.run_path(path, run_name="p6_count")
    m = ns["build_model"]()
    p, b = ns["model_size"](m)
    x, t = torch.randn(4, 3, 32, 32), torch.rand(4)
    out = m(x, t, torch.zeros(4, dtype=torch.long)); out.square().mean().backward()
    assert out.shape == x.shape
    print(f"{path}: params {p:,} buffers {b:,} total {p+b:,}  task={ns['CONFIG']['task']} steps={ns['CONFIG']['steps']}")
