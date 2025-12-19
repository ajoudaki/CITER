#!/bin/bash
export PATH="$HOME/.elan/bin:$PATH"
source /local/home/ajoudaki/citer/Lean/.venv/bin/activate
cd /local/home/ajoudaki/citer/Lean

# Use commit known to work with LeanDojo
# From LeanDojo Benchmark 4: 29dcec074de168ac2bf835a77ef68bbe069194c5
echo "Using LeanDojo-compatible Mathlib4 commit..."
python extract_mathlib_dag.py --commit 29dcec074de168ac2bf835a77ef68bbe069194c5 --output-dir ./output --formats json edgelist
