# Check the fixed parts of experiment programs are byte-identical to the starter:
# everything before the model section (bar STUDENT_ID), the MAX_PARAMETERS line through "# %% Training",
# and generate()/run() to the end of the file.
import sys
from pathlib import Path
def parts(s):
    head, rest = s.split("# %% Model and training settings\n", 1)
    rest = "MAX_PARAMETERS, MAX_STEPS" + rest.split("MAX_PARAMETERS, MAX_STEPS", 1)[1]
    fixed, rest = rest.split("# %% Training\n", 1)
    return head.replace('STUDENT_ID = "xmzj54"', 'STUDENT_ID = "studentID"'), fixed, "def generate" + rest.split("def generate", 1)[1]
ref = parts(Path("starter/dl-practical6.py").read_text())
for path in sys.argv[1:]:
    print(path, "OK" if parts(Path(path).read_text()) == ref else "DIFFERS")
