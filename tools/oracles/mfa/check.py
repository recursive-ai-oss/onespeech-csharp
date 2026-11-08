"""Execute only the documented CLI; never expose installed-module traceback frames."""
import os
import subprocess
import sys

os.environ["PYTHONWARNINGS"] = "ignore"
result = subprocess.run(["mfa", "--help"], capture_output=True, text=True, check=False)
if result.returncode:
    print("MFA CLI failed with status", result.returncode)
    for line in result.stderr.splitlines():
        if line.startswith(("ModuleNotFoundError:", "ImportError:")):
            print(line)
    sys.exit(result.returncode)
print("MFA CLI available")
