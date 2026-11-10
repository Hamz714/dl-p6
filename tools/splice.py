# Build a complete experiment program: starter with STUDENT_ID set, the model section replaced and
# optionally train() replaced.
# Usage: python tools/splice.py tools/models/strong.py exps/exp_010.py [tools/train/ema.py]
import sys
from pathlib import Path
start = Path("starter/dl-practical6.py").read_text().replace('STUDENT_ID = "studentID"', 'STUDENT_ID = "xmzj54"')
head, rest = start.split("# %% Model and training settings\n", 1)
_, tail = rest.split("MAX_PARAMETERS, MAX_STEPS", 1)
if len(sys.argv) > 3:
    fixed, rest = tail.split("# %% Training\n", 1)
    _, rest = rest.split("@torch.no_grad()\ndef generate", 1)
    tail = fixed + Path(sys.argv[3]).read_text() + "@torch.no_grad()\ndef generate" + rest
Path(sys.argv[2]).write_text(head + Path(sys.argv[1]).read_text() + "MAX_PARAMETERS, MAX_STEPS" + tail)
