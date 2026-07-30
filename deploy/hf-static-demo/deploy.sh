#!/usr/bin/env bash
# Rebuild the wheel and redeploy the in-browser (stlite) demo to the free
# static Hugging Face Space.
#
# Prerequisite (one time):  hf auth login
# Then:                     bash deploy/hf-static-demo/deploy.sh
set -euo pipefail

SPACE="${1:-Rekin226/aquascope-dashboard}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
HERE="$ROOT/deploy/hf-static-demo"

echo "→ Building wheel…"
(cd "$ROOT" && python -m build --wheel -o dist)
cp "$ROOT"/dist/aquascope-*-py3-none-any.whl "$HERE/"

# NOTE: upload via the Python API, not `hf upload` — the CLI pre-flights a
# repos/create call that 402s on free accounts (it defaults the sdk to
# gradio), even when the target static Space already exists.
echo "→ Uploading to $SPACE…"
python - "$SPACE" "$HERE" <<'EOF'
import sys
from huggingface_hub import HfApi

space, folder = sys.argv[1], sys.argv[2]
HfApi().upload_folder(repo_id=space, repo_type="space", folder_path=folder,
                      commit_message="Redeploy stlite demo")
print("done")
EOF

echo "✓ Live at: https://huggingface.co/spaces/$SPACE"
