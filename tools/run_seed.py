# Final-selection runs only: load a complete submission script without executing its __main__ block and call its own,
# unmodified run(seed=S) once, in the current working directory (so lab-06-xmzj54/ and the zip land there).
# program_source() still reads the script itself through __file__, which runpy sets.
# Usage (from the candidate's directory): python ~/p6/tools/run_seed.py lab06_final.py 3802
import runpy, sys
ns = runpy.run_path(sys.argv[1], run_name="p6_candidate")
ns["run"](seed=int(sys.argv[2]))
