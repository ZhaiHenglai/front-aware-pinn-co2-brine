#!/bin/bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 PYTHON_EXECUTABLE" >&2
  exit 2
fi

PY="$1"
if [ ! -x "$PY" ]; then
  echo "Configuration K GPU preflight: Python executable is not executable: $PY" >&2
  exit 3
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "Configuration K GPU preflight: nvidia-smi is unavailable in PATH." >&2
  exit 8
fi
if ! nvidia-smi -L >/dev/null 2>&1; then
  echo "Configuration K GPU preflight: nvidia-smi cannot see an allocated NVIDIA GPU." >&2
  exit 8
fi

"$PY" -c '
try:
    import torch
except Exception as exc:
    raise SystemExit(f"Configuration K GPU preflight: PyTorch CUDA inspection failed: {exc}")
if not torch.cuda.is_available():
    raise SystemExit("Configuration K GPU preflight: PyTorch CUDA is unavailable.")
count = int(torch.cuda.device_count())
if count < 1:
    raise SystemExit(f"Configuration K GPU preflight: PyTorch reports {count} visible CUDA devices.")
name = str(torch.cuda.get_device_name(0))
total_memory = int(torch.cuda.get_device_properties(0).total_memory)
minimum_memory = 30 * 1024**3
if "V100" not in name.upper():
    raise SystemExit(
        "Configuration K GPU preflight: the frozen H-case runtime requires a V100; "
        f"allocated device is {name!r}."
    )
if total_memory < minimum_memory:
    raise SystemExit(
        "Configuration K GPU preflight: allocated V100 memory is below the frozen "
        f"30 GiB minimum ({total_memory / 1024**3:.2f} GiB on {name!r})."
    )
print(
    "Configuration K GPU preflight: "
    f"device={name} visible_devices={count} total_memory_gib={total_memory / 1024**3:.2f}"
)
'
