#!/bin/bash

set -e

module --force purge
module load pytorch/2.4

NCU=/appl/spack/v020/install-tree/gcc-10.4.0/cuda-12.1.1-2ppwzf/bin/ncu
echo "ncu: $NCU"
"$NCU" --version

export TORCH_EXTENSIONS_DIR=/tmp/torch_ext_interactive
mkdir -p "$TORCH_EXTENSIONS_DIR"

echo "=== Pre-build ==="
python attention_template_gpu.py
echo "=== Pre-build OK ==="

echo "=== Running ncu ==="
"$NCU" \
  --target-processes all \
  --kernel-name-base function \
  --launch-skip 0 \
  --launch-count 3 \
  --section SpeedOfLight \
  --section MemoryWorkloadAnalysis \
  --section ComputeWorkloadAnalysis \
  --section Occupancy \
  --export attention_profile \
  --force-overwrite \
  python attention_template_gpu.py

echo ""
echo "Done! attention_profile.ncu-rep is ready."
