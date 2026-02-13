#!/bin/bash
if [ $SLURM_LOCALID -eq 0 ]; then
 nsys profile --gpu-metrics-devices=0 -t cuda,nvtx -o gevolution_test_${SLURM_NODEID} -f true "$@"
else
 "$@"
fi
