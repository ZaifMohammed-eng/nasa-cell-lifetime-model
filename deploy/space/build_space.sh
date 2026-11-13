#!/usr/bin/env bash
# Assemble a Hugging Face Space folder from this repo.
#   bash deploy/space/build_space.sh <path-to-cloned-space-repo>
# Exports the current champion first if serve/model/champion.json is missing.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DEST="${1:?usage: build_space.sh <space-repo-dir>}"
[ -f "$ROOT/serve/model/champion.json" ] || (cd "$ROOT" && python -m mlops.export)
mkdir -p "$DEST/src" "$DEST/mlops" "$DEST/serve/static" "$DEST/serve/model" "$DEST/data"
cp "$ROOT"/src/__init__.py "$ROOT"/src/lifetime.py "$DEST/src/"
cp "$ROOT"/mlops/__init__.py "$ROOT"/mlops/core.py "$DEST/mlops/"
cp "$ROOT"/serve/__init__.py "$ROOT"/serve/app.py "$ROOT"/serve/cases.py "$DEST/serve/"
cp "$ROOT"/serve/static/index.html "$DEST/serve/static/"
cp "$ROOT"/serve/model/champion.json "$ROOT"/serve/model/champion.meta.json "$DEST/serve/model/"
cp "$ROOT"/data/capacity.csv "$ROOT"/data/shift_capacity.csv "$DEST/data/"
cp "$ROOT"/deploy/space/Dockerfile "$ROOT"/deploy/space/requirements-space.txt "$ROOT"/deploy/space/README.md "$DEST/"
cp "$ROOT"/LICENSE "$DEST/" 2>/dev/null || true
echo "Space folder ready: $DEST"
