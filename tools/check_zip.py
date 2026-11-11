# Verify a submission zip against its script: exact entry set, embedded code byte-identical, JSON hash/seed/KID/sizes.
# Usage: python tools/check_zip.py <dir containing lab06_final.py and lab-06-xmzj54.zip> [...]
import hashlib, json, sys, zipfile
from pathlib import Path
expected = sorted(["lab-06-code.py", "lab-06-results.json"] + [f"samples/batch-{i:02d}.png" for i in range(1, 11)])
for d in map(Path, sys.argv[1:]):
    z = zipfile.ZipFile(d / "lab-06-xmzj54.zip")
    code = z.read("lab-06-code.py"); r = json.loads(z.read("lab-06-results.json"))
    ok = (sorted(z.namelist()) == expected and code == (d / "lab06_final.py").read_bytes()
          and r["code_sha256"] == hashlib.sha256(code).hexdigest())
    print(f"{d}: {'OK' if ok else 'PROBLEM'} seed {r['seed']} task {r['config']['task']} KID x1000 {1000 * r['metrics']['kid']:.3f} "
          f"params {r['parameters']} + {r['buffer_elements']} updates {r['updates']['model']} batch {r['config']['batch_size']}")
