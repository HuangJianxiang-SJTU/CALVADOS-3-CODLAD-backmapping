#!/bin/bash
# Batch runner for PED00142 and PED00159
cd /MDdata/data04/jxhuang/cg_cascade
CACHE_DIR="logs/figure3_step100/cache"
TARGET=500
MAX_ITER=30

SYSTEMS="PED00142 PED00159"

for iter in $(seq 1 $MAX_ITER); do
    echo "=== Iteration $iter ($(date)) ==="
    incomplete=0
    pids=()
    gpu=0
    for ped in $SYSTEMS; do
        count=$(find ${CACHE_DIR} -name "${ped}_*.npz" 2>/dev/null | wc -l)
        if [ "$count" -lt "$TARGET" ]; then
            echo "  $ped: $count/$TARGET"
            incomplete=$((incomplete + 1))
            logf="logs/figure3_step100/fig4_${ped}_iter${iter}.log"
            conda run -n cg_ensemble python scripts/fig3_step100_worker.py --systems $ped --gpu $gpu > $logf 2>&1 &
            pids+=($!)
            gpu=$((gpu + 1))
        fi
    done

    if [ $incomplete -eq 0 ]; then
        echo "Both systems complete!"
        break
    fi

    for pid in "${pids[@]}"; do
        wait $pid 2>/dev/null
    done
    echo "Iteration $iter complete."
done
