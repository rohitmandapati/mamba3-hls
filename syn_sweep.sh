#!/usr/bin/env bash
# =============================================================================
# syn_sweep.sh — HLS Synthesis + Bitstream for 6 ideal configs
# =============================================================================
# Saves:
#   important_results/syn/syn_<label>.rpt          — csynth report
#   important_results/syn/syn_<label>_build.log    — full build log
#   important_results/bitstreams/<label>.bit        — final bitstream
#
# Usage:
#   ./syn_sweep.sh              # NSTEPS=64 by default
#   NSTEPS=64 ./syn_sweep.sh
# =============================================================================

NSTEPS="${NSTEPS:-64}"
SYN_DIR="important_results/syn"
BIT_DIR="important_results/bitstreams"
mkdir -p "$SYN_DIR" "$BIT_DIR"

FP_TOTAL=24
FP_INT=10

# ---------------------------------------------------------------------------
# 6 configs: label -> make flags
# ---------------------------------------------------------------------------
LABELS=(
    baseline
    pragma
    fixed_pwl
    fixed_cordic
    pragma_fixed_pwl
    pragma_fixed_cordic
)

declare -A FLAGS=(
    [baseline]=""
    [pragma]="MAMBA3_ENABLE_HLS_PRAGMAS=1"
    [fixed_pwl]="MAMBA3_USE_FIXED_POINT=1 MAMBA3_FIXED_TOTAL_BITS=$FP_TOTAL MAMBA3_FIXED_INT_BITS=$FP_INT"
    [fixed_cordic]="MAMBA3_USE_FIXED_POINT=1 MAMBA3_USE_PWL_NONLINEARITIES=0 MAMBA3_USE_CORDIC=1 MAMBA3_FIXED_TOTAL_BITS=$FP_TOTAL MAMBA3_FIXED_INT_BITS=$FP_INT"
    [pragma_fixed_pwl]="MAMBA3_ENABLE_HLS_PRAGMAS=1 MAMBA3_USE_FIXED_POINT=1 MAMBA3_FIXED_TOTAL_BITS=$FP_TOTAL MAMBA3_FIXED_INT_BITS=$FP_INT"
    [pragma_fixed_cordic]="MAMBA3_ENABLE_HLS_PRAGMAS=1 MAMBA3_USE_FIXED_POINT=1 MAMBA3_USE_PWL_NONLINEARITIES=0 MAMBA3_USE_CORDIC=1 MAMBA3_FIXED_TOTAL_BITS=$FP_TOTAL MAMBA3_FIXED_INT_BITS=$FP_INT"
)

# ---------------------------------------------------------------------------
# Rename stale project dirs within the SAME NFS filesystem.
# KEY INSIGHT: mv to /tmp fails (cross-device copy+delete hits NFS ghost).
#              mv within the same dir is a single metadata op — ALWAYS works.
# ---------------------------------------------------------------------------
nuke_projects() {
    local label="$1"

    # In-place rename: same filesystem, so this is just a directory entry update.
    # NFS ghost .nfs* files are irrelevant because we never try to delete them.
    for d in mamba3_step.prj mamba3.prj kria.prj; do
        if [[ -d "$d" ]]; then
            local backup="${d}.stale_${label}_$$"
            if mv "$d" "$backup" 2>/dev/null; then
                echo "  -> Renamed $d -> $backup (NFS-safe)"
            else
                echo "  !! Cannot rename $d — leaving it (Vitis -reset will handle it)"
            fi
        fi
    done

    # Clean other artifacts normally
    rm -rf build logs .Xil result kria_dir 2>/dev/null || true
    rm -f *.dat *.log *.jou 2>/dev/null || true
}

for label in "${LABELS[@]}"; do
    flags="${FLAGS[$label]}"

    echo ""
    echo "================================================================="
    echo " [$label]  SYNTHESIS + BITSTREAM"
    echo "   MAMBA3_NSTEPS=$NSTEPS $flags"
    echo "================================================================="

    # ------------------------------------------------------------------
    # Clean everything — in-place NFS rename (same filesystem, always safe)
    # ------------------------------------------------------------------
    nuke_projects "$label"

    # ------------------------------------------------------------------
    # 1. HLS synthesis  (make dut)
    # ------------------------------------------------------------------
    echo "  -> Running HLS synthesis..."
    make dut MAMBA3_NSTEPS="$NSTEPS" $flags 2>&1 | tee "$SYN_DIR/syn_${label}_build.log" || true

    # Copy the synthesis report — project dir is mamba3_step.prj
    RPT="mamba3_step.prj/solution1/syn/report/dut_csynth.rpt"
    if [[ -f "$RPT" ]]; then
        cp "$RPT" "$SYN_DIR/syn_${label}.rpt"
        echo "  -> Saved synthesis report: $SYN_DIR/syn_${label}.rpt"
    else
        echo "  -> WARNING: $RPT not found. Check $SYN_DIR/syn_${label}_build.log"
    fi

    # ------------------------------------------------------------------
    # 2. Bitstream  (make bitstream — reuses existing HLS output)
    # ------------------------------------------------------------------
    echo "  -> Building bitstream..."
    make bitstream MAMBA3_NSTEPS="$NSTEPS" $flags 2>&1 | tee -a "$SYN_DIR/syn_${label}_build.log" || true

    if [[ -f build/bitstream.bit ]]; then
        cp build/bitstream.bit "$BIT_DIR/${label}.bit"
        echo "  -> Saved bitstream: $BIT_DIR/${label}.bit"
    else
        echo "  -> WARNING: build/bitstream.bit not found."
    fi

    # Back up kria_dir so the host binary + data match this config
    if [[ -d kria_dir ]]; then
        KRIA_BK="$BIT_DIR/kria_${label}"
        rm -rf "$KRIA_BK"
        cp -r kria_dir "$KRIA_BK"
        echo "  -> kria_dir backed up: $KRIA_BK"
    fi

    echo "  -> Done: $label"
done

echo ""
echo "================================================================="
echo " SYNTHESIS + BITSTREAM SWEEP COMPLETE"
echo " Reports:    $SYN_DIR/"
echo " Bitstreams: $BIT_DIR/"
echo "================================================================="
