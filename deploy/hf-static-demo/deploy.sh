#!/usr/bin/env bash
# Deploy the redirect page to the retired static Hugging Face Space.
#
# The dashboard Space stopped being a demo in #235: its analyses are in the
# Explorer now, and this folder is the "moved" page that says so. No wheel is
# built or shipped - a Space serving a stale wheel is exactly what this
# replaces.
#
# Prerequisite (one time):  hf auth login
# Then:                     bash deploy/hf-static-demo/deploy.sh
set -euo pipefail

SPACE="${1:-Rekin226/aquascope-dashboard}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
HERE="$ROOT/deploy/hf-static-demo"

# NOTE: upload via the Python API, not `hf upload` - the CLI pre-flights a
# repos/create call that 402s on free accounts (it defaults the sdk to
# gradio), even when the target static Space already exists.
echo "→ Uploading the redirect to ${SPACE}…"
python - "$SPACE" "$HERE" <<'PY'
import sys
from huggingface_hub import HfApi

space, folder = sys.argv[1], sys.argv[2]
# delete_patterns clears what the old stlite deploy left behind (the wheel and
# its stylesheet); upload_folder does not remove remote files on its own, so
# without this the Space keeps serving aquascope-0.8.1-py3-none-any.whl.
HfApi().upload_folder(
    repo_id=space,
    repo_type="space",
    folder_path=folder,
    delete_patterns=["*.whl", "style.css"],
    commit_message="Redirect to the Explorer (#235)",
)
print("done")
PY

echo "✓ Live at: https://huggingface.co/spaces/$SPACE"
