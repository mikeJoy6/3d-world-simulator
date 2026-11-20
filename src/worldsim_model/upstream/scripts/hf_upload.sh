#!/bin/bash
# Upload a folder to the Hub (resumable; re-running continues where it stopped).
#
#   HF_TOKEN=... scripts/hf_upload.sh release jwang771/3d-world-simulator checkpoints
#   HF_TOKEN=... scripts/hf_upload.sh data_final/cloth jwang771/3d-world-simulator data/processed/cloth
#
# The token is read from the environment of this one command and never written to disk by the script.
set -euo pipefail
SRC=${1:?local dir}; REPO=${2:?repo id}; DEST=${3:-}; TYPE=dataset
[ "${4:-}" = --type ] && TYPE=${5:?dataset or model}
[ -n "${HF_TOKEN:-}" ] || { echo "set HF_TOKEN=<write token> in the command"; exit 1; }
echo "[hf] $SRC -> https://huggingface.co/$([ "$TYPE" = dataset ] && echo datasets/)$REPO/tree/main/$DEST"
exec hf upload "$REPO" "$SRC" "$DEST" --repo-type "$TYPE"
