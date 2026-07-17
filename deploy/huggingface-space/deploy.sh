#!/usr/bin/env bash
# One-command deploy of the AquaScope dashboard to a Hugging Face Space.
#
# Prerequisite (one time):  hf auth login
# Then:                     bash deploy/huggingface-space/deploy.sh
set -euo pipefail

SPACE="${1:-Rekin226/aquascope-dashboard}"
HERE="$(cd "$(dirname "$0")" && pwd)"

# NOTE: as of 2026, hosting Docker/Gradio Spaces on free cpu-basic requires a
# Hugging Face PRO subscription — this script will 402 on a free account.
# The free hosting path is Streamlit Community Cloud (share.streamlit.io).
echo "→ Creating Space $SPACE (no-op if it already exists)…"
hf repo create "$SPACE" --repo-type space --space-sdk docker --exist-ok || true

echo "→ Uploading dashboard entry point, requirements, and Space card…"
hf upload "$SPACE" "$HERE" . --repo-type space --exclude "deploy.sh"

echo "✓ Deployed. The Space builds in a few minutes at:"
echo "  https://huggingface.co/spaces/$SPACE"
