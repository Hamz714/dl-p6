# Run one complete Practical 6 experiment program with a chosen seed and split.
# Usage: python runner.py exps/exp_012_name.py --seed 3801 --split validation
import argparse
import json
import runpy
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("exp")
parser.add_argument("--seed", type=int, default=3801)
parser.add_argument("--split", default="validation", choices=["validation", "test"])
parser.add_argument("--tag", default="")
parser.add_argument("--snapshots", default="", help="analysis only: e.g. 10000,15000 = also score the primary EMA "
                    "copied at those updates (needs the program's SNAPSHOT_AT hook)")
args = parser.parse_args()

exp = Path(args.exp).resolve()
name = f"{exp.stem}_s{args.seed}_{args.split}" + (f"_{args.tag}" if args.tag else "")
out = Path("runs") / name

# run_name is not "__main__", so the program's own run() call does not fire;
# runpy still sets __file__, which program_source() needs.
ns = runpy.run_path(str(exp), run_name="p6_experiment")
# Keep a handle on the model run() builds, to save its weights. run_path returns a COPY of the program's
# globals, so the wrapper must go into the dict run() actually reads (run.__globals__).
program, built = ns["run"].__globals__, []
build = program["build_model"]
program["build_model"] = lambda: built.append(build()) or built[-1]
if args.snapshots:
    assert "SNAPSHOT_AT" in program, "program has no SNAPSHOT_AT hook"
    program["SNAPSHOT_AT"] = tuple(int(s) for s in args.snapshots.split(","))
start = time.time()
result = ns["run"](out=str(out), seed=args.seed, split=args.split, show=False)
minutes = (time.time() - start) / 60
program["build_model"] = build


def save_weights(state, tag):
    """Weights for offline diagnostics only (git-ignored *.pt); never used for picking anything. Never fatal."""
    try:
        (out / "weights").mkdir(parents=True, exist_ok=True)
        ns["torch"].save(state, out / "weights" / f"{tag}.pt")
    except Exception as error:
        print("weights not saved:", repr(error), flush=True)

ema = ns.get("EMA_DECAYS", (None,))[0]
row = dict(exp=exp.stem, seed=args.seed, split=args.split, tag=args.tag or (f"ema{ema}" if ema else ""),
           kid1000=round(1000 * result["metrics"]["kid"], 3),
           params=result["parameters"], buffers=result["buffer_elements"],
           config=result["config"], minutes=round(minutes, 1),
           finished=time.strftime("%Y-%m-%d %H:%M:%S"))
Path("runs").mkdir(exist_ok=True)
with open("runs/index.jsonl", "a") as f:
    f.write(json.dumps(row) + "\n")
print("RESULT", json.dumps(row), flush=True)
save_weights(built[0].state_dict() if built else {}, row["tag"] or "main")

# Extra EMA decays tracked by train() (EXTRA_EMA = {decay: state_dict}): score each with the fixed
# image_score()/generate() into its own folder; logged as extra rows tagged ema<decay>.
for decay, state in ns.get("EXTRA_EMA", {}).items():
    model = ns["build_model"]().to(device := "cuda" if ns["torch"].cuda.is_available() else "cpu")
    model.load_state_dict(state)
    save_weights(state, f"ema{decay}")
    folder = out / f"ema{decay}"
    folder.mkdir(parents=True, exist_ok=True)
    metrics = ns["image_score"](model, folder, device, args.split)
    extra = dict(row, tag=f"ema{decay}", kid1000=round(1000 * metrics["kid"], 3),
                 minutes=round((time.time() - start) / 60, 1), finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    with open("runs/index.jsonl", "a") as f:
        f.write(json.dumps(extra) + "\n")
    print("RESULT", json.dumps(extra), flush=True)

# Mid-run snapshots of the primary EMA (analysis only): scored with the same image_score() path.
for step, tensors in sorted(ns.get("SNAPSHOTS", {}).items()):
    model = ns["build_model"]().to(device := "cuda" if ns["torch"].cuda.is_available() else "cpu")
    names = [name for name, _ in model.named_parameters()]
    model.load_state_dict({**built[0].state_dict(), **dict(zip(names, tensors))})
    tag = f"snap{step}_{row['tag']}"
    save_weights(model.state_dict(), tag)
    folder = out / tag
    folder.mkdir(parents=True, exist_ok=True)
    metrics = ns["image_score"](model, folder, device, args.split)
    extra = dict(row, tag=tag, kid1000=round(1000 * metrics["kid"], 3),
                 minutes=round((time.time() - start) / 60, 1), finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    with open("runs/index.jsonl", "a") as f:
        f.write(json.dumps(extra) + "\n")
    print("RESULT", json.dumps(extra), flush=True)
