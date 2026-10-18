#!/usr/bin/env bash
# Full four-client run: train A-D, fit the aggregation gate on fold 9, evaluate once on fold 10.
#   python scripts/launch_detached.py outputs/fourclient/run.log bash scripts/run_fourclient.sh [seeds...]
set -euo pipefail
cd "$(dirname "$0")/.."
SEEDS=${*:-42}
PY=.venv/bin/python
echo "[$(date '+%F %T')] start train (seeds $SEEDS)"
$PY -m cardiomoe.fourclient.train --seeds $SEEDS
echo "[$(date '+%F %T')] start aggregate"
$PY -m cardiomoe.fourclient.aggregate --seeds $SEEDS
echo "[$(date '+%F %T')] start evaluate"
$PY -m cardiomoe.fourclient.evaluate --seeds $SEEDS
echo "[$(date '+%F %T')] done"
