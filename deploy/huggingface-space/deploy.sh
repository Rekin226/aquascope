#!/usr/bin/env bash
# One-command deploy of the AquaScope dashboard to a Hugging Face Space.
#
# Prerequisite (one time):  hf auth login
# Then:                     bash deploy/huggingface-space/deploy.sh
set -euo pipefail

SPACE="${1:-Rekin226/aquascope-dashboard}"
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "→ Creating Space $SPACE (no-op if it already exists)…"
hf repo create "$SPACE" --repo-type space --space-sdk streamlit --exist-ok || true

echo "→ Uploading dashboard entry point, requirements, and Space card…"
hf upload "$SPACE" "$HERE" . --repo-type space --exclude "deploy.sh"

echo "✓ Deployed. The Space builds in a few minutes at:"
echo "  https://huggingface.co/spaces/$SPACE"
