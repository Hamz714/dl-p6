# Build a complete experiment program: starter with STUDENT_ID set and the model section replaced.
# Usage: python tools/splice.py tools/models/unet_a.py exps/exp_010_unetA.py
import sys
from pathlib import Path
start = Path("starter/dl-practical6.py").read_text().replace('STUDENT_ID = "studentID"', 'STUDENT_ID = "xmzj54"')
head, rest = start.split("# %% Model and training settings\n", 1)
_, tail = rest.split("MAX_PARAMETERS, MAX_STEPS", 1)
Path(sys.argv[2]).write_text(head + Path(sys.argv[1]).read_text() + "MAX_PARAMETERS, MAX_STEPS" + tail)
