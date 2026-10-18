"""Start a command in its own session (survives the terminal / agent session ending),
kept awake with caffeinate, output appended to a log.
    python scripts/launch_detached.py outputs/pipeline/pipeline.log env SKIP="client_c crossfit_c" bash scripts/run_pipeline.sh
"""
import os
import subprocess
import sys

log, cmd = sys.argv[1], sys.argv[2:]
with open(log, "a") as f:
    p = subprocess.Popen(["caffeinate", "-i", *cmd], stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         start_new_session=True, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(p.pid)
