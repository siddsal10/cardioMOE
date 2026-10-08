#!/usr/bin/env bash
# Phase 0: download all public datasets (no credentialing needed).
#
#   bash scripts/download.sh                 # PTB-XL (100 Hz), Chapman, Mendeley images
#   bash scripts/download.sh --debug         # metadata + ~200 records per dataset
#   bash scripts/download.sh --optional      # also PTB-XL+ and MTSamples (needs Kaggle API key)
#   bash scripts/download.sh --wget          # use physionet.org + wget -r (the plan's original recipe)
#
# Default mode pulls PhysioNet data from its official AWS open-data mirror
# (s3://physionet-open), which is much faster than physionet.org for tens of
# thousands of small files. Every step is resumable: re-run after an interruption.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-.venv/bin/python}
DEBUG=""; OPTIONAL=0; WGET=0
for a in "$@"; do
  case $a in
    --debug) DEBUG="--debug" ;;
    --optional) OPTIONAL=1 ;;
    --wget) WGET=1 ;;
    *) echo "unknown arg $a"; exit 1 ;;
  esac
done

if [[ $WGET == 1 ]]; then
  wget -r -N -c -np -nH --cut-dirs=3 -X /files/ptb-xl/1.0.3/records500 -P data/raw/ptbxl https://physionet.org/files/ptb-xl/1.0.3/
  wget -r -N -c -np -nH --cut-dirs=3 -P data/raw/chapman https://physionet.org/files/ecg-arrhythmia/1.0.0/
  [[ $OPTIONAL == 1 ]] && wget -r -N -c -np -nH --cut-dirs=3 -P data/raw/ptbxl_plus https://physionet.org/files/ptb-xl-plus/1.0.1/
else
  # Only records100/ is used for training; records500/ is skipped to save ~1.5 GB.
  $PY scripts/fetch_physionet.py ptb-xl/1.0.3 data/raw/ptbxl --exclude records500/ $DEBUG
  $PY scripts/fetch_physionet.py ecg-arrhythmia/1.0.0 data/raw/chapman $DEBUG
  [[ $OPTIONAL == 1 ]] && $PY scripts/fetch_physionet.py ptb-xl-plus/1.0.1 data/raw/ptbxl_plus $DEBUG
fi

# Mendeley ECG images (gwbz3fsgp8 v2) via the public Mendeley Data API.
# Manual alternative: "Download All" at https://data.mendeley.com/datasets/gwbz3fsgp8/2
# and unzip into data/raw/ecg_images_pk/.
$PY scripts/fetch_mendeley.py data/raw/ecg_images_pk $DEBUG

if [[ $OPTIONAL == 1 ]]; then
  kaggle datasets download -d tboyle10/medicaltranscriptions -p data/raw/mtsamples --unzip
fi

$PY scripts/verify_data.py $DEBUG
