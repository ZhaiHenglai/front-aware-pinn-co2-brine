#!/bin/bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 PROJECT_ROOT" >&2
  exit 2
fi

PROJECT_ROOT="$1"
MANIFEST="$PROJECT_ROOT/manifests/CODE_SHA256SUMS"
if [ ! -f "$MANIFEST" ]; then
  echo "Configuration K code manifest not found: $MANIFEST" >&2
  exit 9
fi

if ! (cd "$PROJECT_ROOT" && sha256sum --status -c "$MANIFEST"); then
  echo "Configuration K code checksum preflight failed; changed or missing files:" >&2
  (cd "$PROJECT_ROOT" && sha256sum -c "$MANIFEST") >&2 || true
  exit 9
fi
