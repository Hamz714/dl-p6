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
args = parser.parse_args()

exp = Path(args.exp).resolve()
name = f"{exp.stem}_s{args.seed}_{args.split}" + (f"_{args.tag}" if args.tag else "")
out = Path("runs") / name

# run_name is not "__main__", so the program's own run() call does not fire;
# runpy still sets __file__, which program_source() needs.
ns = runpy.run_path(str(exp), run_name="p6_experiment")
start = time.time()
result = ns["run"](out=str(out), seed=args.seed, split=args.split, show=False)
minutes = (time.time() - start) / 60

row = dict(exp=exp.stem, seed=args.seed, split=args.split, tag=args.tag,
           kid1000=round(1000 * result["metrics"]["kid"], 3),
           params=result["parameters"], buffers=result["buffer_elements"],
           config=result["config"], minutes=round(minutes, 1),
           finished=time.strftime("%Y-%m-%d %H:%M:%S"))
Path("runs").mkdir(exist_ok=True)
with open("runs/index.jsonl", "a") as f:
    f.write(json.dumps(row) + "\n")
print("RESULT", json.dumps(row), flush=True)
