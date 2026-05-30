#!/bin/bash
#SBATCH --job-name=ncu_profile
#SBATCH --account=project_2019091
#SBATCH --partition=gpusmall
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=16G
#SBATCH --output=ncu_%j.out
#SBATCH --error=ncu_%j.err

module --force purge
module load pytorch/2.4

NCU=""
for glob in \
    /usr/local/cuda/bin/ncu \
    /appl/spack/v020/install-tree/gcc-*/cuda-12*/bin/ncu \
    /appl/spack/v017/install-tree/gcc-*/cuda-12*/bin/ncu \
    /appl/spack/*/install-tree/*/cuda-12*/bin/ncu; do
    for p in $glob; do
        [ -x "$p" ] && { NCU="$p"; break 2; }
    done
done

if [ -z "$NCU" ]; then
    echo "ERROR: ncu not found"; find /appl/spack -name "ncu" -type f 2>/dev/null; exit 1
fi
echo "ncu: $NCU"
"$NCU" --version
echo ""

PYTHON=$(python -c "import sys; print(sys.executable)")

export TORCH_EXTENSIONS_DIR=/tmp/torch_ext_${SLURM_JOB_ID}
mkdir -p "$TORCH_EXTENSIONS_DIR"

echo "=== Pre-build (no ncu) ==="
"$PYTHON" attention_template_gpu.py
if [ $? -ne 0 ]; then
    echo "Pre-build FAILED — aborting"; exit 1
fi
echo "=== Pre-build OK ==="
echo ""

echo "=== ncu profile (binary export) ==="
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
  "$PYTHON" attention_template_gpu.py

echo ""
echo "=== ncu text summary (SpeedOfLight per kernel) ==="
"$NCU" \
  --target-processes all \
  --kernel-name-base function \
  --launch-skip 0 \
  --launch-count 3 \
  --section SpeedOfLight \
  --print-summary per-kernel \
  "$PYTHON" attention_template_gpu.py
