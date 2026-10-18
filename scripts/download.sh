#!/usr/bin/env bash
# Download PTB-XL 1.0.3 (open access, no credentialing) and verify it.
#
#   bash scripts/download.sh            # PTB-XL metadata + records100 (100 Hz)
#   bash scripts/download.sh --debug    # metadata + ~200 records
#   bash scripts/download.sh --wget     # use physionet.org + wget -r instead of the AWS mirror
#
# Default mode pulls from PhysioNet's official AWS open-data mirror (s3://physionet-open),
# which is much faster than physionet.org for many small files. Resumable: re-run after an
# interruption. Only records100/ is used; records500/ is skipped to save ~1.5 GB.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-.venv/bin/python}
DEBUG=""; WGET=0
for a in "$@"; do
  case $a in
    --debug) DEBUG="--debug" ;;
    --wget) WGET=1 ;;
    *) echo "unknown arg $a"; exit 1 ;;
  esac
done

if [[ $WGET == 1 ]]; then
  wget -r -N -c -np -nH --cut-dirs=3 -X /files/ptb-xl/1.0.3/records500 -P data/raw/ptbxl https://physionet.org/files/ptb-xl/1.0.3/
else
  $PY scripts/fetch_physionet.py ptb-xl/1.0.3 data/raw/ptbxl --exclude records500/ $DEBUG
fi

$PY scripts/verify_data.py $DEBUG
