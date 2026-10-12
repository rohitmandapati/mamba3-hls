#!/usr/bin/env bash
# =============================================================================
# csim_sweep.sh — Mamba-3 C-simulation sweep (ALL configs, csim only)
#
# Configs:
#   1.  baseline                         fp32, no pragmas
#   2.  pwl_fp<T>i<I>                    fixed+PWL  (precision sweep)
#   3.  cordic_fp<T>i<I>                 fixed+CORDIC  (precision sweep)
#   4.  pragma                           fp32 + HLS pragmas
#   5.  pragma_fixed_pwl_fp<T>i<I>       pragma + fixed+PWL  (precision sweep)
#   6.  pragma_fixed_cordic_fp<T>i<I>    pragma + fixed+CORDIC  (precision sweep)
#
# Usage:
#   ./csim_sweep.sh              # NSTEPS=64 by default
#   NSTEPS=1 ./csim_sweep.sh
# =============================================================================

NSTEPS="${NSTEPS:-64}"
OUT_DIR="important_results/x86"
FP_DIR="$OUT_DIR/fixed_pt_sweep"
mkdir -p "$OUT_DIR" "$FP_DIR"

TOTAL_BITS_LIST=(16 20 24 28 32)
INT_BITS_LIST=(4 6 8 10 12 16)

PASS_COUNT=0
FAIL_COUNT=0

# ---------------------------------------------------------------------------
# run_csim label out_dir [extra make flags...]
# ---------------------------------------------------------------------------
run_csim() {
    local label="$1"
    local dir="$2"
    shift 2
    local flags="$*"

    local out_file="$dir/csim_${label}.txt"
    local log="$dir/csim_${label}_full.log"

    echo "  -> make csim MAMBA3_NSTEPS=$NSTEPS $flags"
    # Always succeed so the script never exits early
    make csim MAMBA3_NSTEPS="$NSTEPS" $flags >"$log" 2>&1 || true

    # Grab summary block
    grep -A 25 "Global error summary:" "$log" > "$out_file" 2>/dev/null || true

    # Prepend config header
    {
        echo "# label:  $label"
        echo "# flags:  MAMBA3_NSTEPS=$NSTEPS $flags"
        echo "# date:   $(date)"
        echo ""
        cat "$out_file"
    } > "$out_file.tmp"
    mv "$out_file.tmp" "$out_file"

    if grep -q "\*\*\* PASS \*\*\*" "$out_file"; then
        echo "     PASS"
        PASS_COUNT=$(( PASS_COUNT + 1 ))
    elif grep -q "\*\*\* FAIL \*\*\*" "$out_file"; then
        out_max=$(grep "out " "$out_file" | grep -oP '[\d.e+\-]+' | head -1 || true)
        st_max=$(grep "state_out" "$out_file" | grep -oP '[\d.e+\-]+' | head -1 || true)
        echo "     FAIL  (out_max=${out_max:-?}  state_max=${st_max:-?})"
        FAIL_COUNT=$(( FAIL_COUNT + 1 ))
    else
        echo "     ERROR (no result — check $log)"
        FAIL_COUNT=$(( FAIL_COUNT + 1 ))
    fi
}

# ---------------------------------------------------------------------------
# fp_sweep prefix out_dir extra_flags
#   Iterates over TOTAL_BITS x INT_BITS, adds MAMBA3_USE_FIXED_POINT=1
# ---------------------------------------------------------------------------
fp_sweep() {
    local prefix="$1"
    local dir="$2"
    shift 2
    local extra="$*"  # e.g. "MAMBA3_USE_CORDIC=1" or "MAMBA3_ENABLE_HLS_PRAGMAS=1"

    for tb in "${TOTAL_BITS_LIST[@]}"; do
        for ib in "${INT_BITS_LIST[@]}"; do
            local frac=$(( tb - ib ))
            if [ "$ib" -ge "$tb" ] || [ "$frac" -lt 4 ]; then
                continue
            fi
            local label="${prefix}_fp${tb}i${ib}"
            echo ""
            echo "  [total=$tb int=$ib frac=$frac]  $label"
            run_csim "$label" "$dir" \
                "MAMBA3_USE_FIXED_POINT=1 MAMBA3_FIXED_TOTAL_BITS=$tb MAMBA3_FIXED_INT_BITS=$ib $extra"
        done
    done
}

# =============================================================================
# 1. Baseline — fp32, no pragmas
# =============================================================================
echo ""
echo "================================================================="
echo " [1] BASELINE  (fp32, NSTEPS=$NSTEPS)"
echo "================================================================="
run_csim "baseline" "$OUT_DIR"

# =============================================================================
# 2. Fixed + PWL  — precision sweep
# =============================================================================
echo ""
echo "================================================================="
echo " [2] FIXED POINT + PWL  (precision sweep)"
echo "================================================================="
fp_sweep "pwl" "$FP_DIR"

# =============================================================================
# 3. Fixed + CORDIC  — precision sweep
# =============================================================================
echo ""
echo "================================================================="
echo " [3] FIXED POINT + CORDIC  (precision sweep)"
echo "================================================================="
fp_sweep "cordic" "$FP_DIR" "MAMBA3_USE_PWL_NONLINEARITIES=0 MAMBA3_USE_CORDIC=1"

# =============================================================================
# 4. Pragma only — fp32 + HLS pragmas
# =============================================================================
echo ""
echo "================================================================="
echo " [4] PRAGMA ONLY  (fp32 + HLS pragmas)"
echo "================================================================="
run_csim "pragma" "$OUT_DIR" "MAMBA3_ENABLE_HLS_PRAGMAS=1"

# =============================================================================
# 5. Pragma + Fixed + PWL  — precision sweep
# =============================================================================
echo ""
echo "================================================================="
echo " [5] PRAGMA + FIXED POINT + PWL  (precision sweep)"
echo "================================================================="
fp_sweep "pragma_pwl" "$FP_DIR" "MAMBA3_ENABLE_HLS_PRAGMAS=1"

# =============================================================================
# 6. Pragma + Fixed + CORDIC  — precision sweep
# =============================================================================
echo ""
echo "================================================================="
echo " [6] PRAGMA + FIXED POINT + CORDIC  (precision sweep)"
echo "================================================================="
fp_sweep "pragma_cordic" "$FP_DIR" "MAMBA3_ENABLE_HLS_PRAGMAS=1 MAMBA3_USE_PWL_NONLINEARITIES=0 MAMBA3_USE_CORDIC=1"

# =============================================================================
# Final summary
# =============================================================================
echo ""
echo "================================================================="
echo " ALL SWEEPS COMPLETE"
echo "   PASS: $PASS_COUNT    FAIL: $FAIL_COUNT"
echo " Flat results:         $OUT_DIR/"
echo " Fixed-pt sweep:       $FP_DIR/"
echo "================================================================="
