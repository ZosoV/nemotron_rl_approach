#!/usr/bin/env bash
# Sync the dataset and/or kernel to Kaggle.
#
# Usage:
#   ./sync_kaggle.sh                           # sync both (default)
#   ./sync_kaggle.sh "message"                 # sync both with message
#   ./sync_kaggle.sh "message" --dataset       # dataset only
#   ./sync_kaggle.sh "message" --experiment    # experiment kernel only
set -e

MESSAGE=${1:-"Sync from local repo"}
TARGET=${2:-"--both"}

SYNC_DATASET=false
SYNC_EXPERIMENT=false

case "$TARGET" in
  --dataset)    SYNC_DATASET=true ;;
  --experiment) SYNC_EXPERIMENT=true ;;
  --both|*)     SYNC_DATASET=true; SYNC_EXPERIMENT=true ;;
esac

# ── 1. Dataset ─────────────────────────────────────────────────────────────────
if $SYNC_DATASET; then
  echo ">>> Syncing dataset zosov07/nemotron-rl-approach ..."

  UPLOAD_DIR=$(mktemp -d)
  trap "rm -rf $UPLOAD_DIR" EXIT

  cp -r nemotron_grpo notebooks offline_packages pyproject.toml CLAUDE.md README.md dataset-metadata.json "$UPLOAD_DIR/"

  kaggle datasets version -p "$UPLOAD_DIR" -m "$MESSAGE" --dir-mode tar
  echo "Dataset updated: https://www.kaggle.com/datasets/zosov07/nemotron-rl-approach"
fi

# ── 2. Experiment kernel ────────────────────────────────────────────────────────
# if $SYNC_EXPERIMENT; then
#   echo ""
#   echo ">>> Syncing kernel zosov07/nvidia-nemotron-experiment ..."

#   (cd notebooks/nemotron-experiment && kaggle kernels push -p .)
#   echo "Kernel updated: https://www.kaggle.com/code/zosov07/nvidia-nemotron-experiment"
# fi
