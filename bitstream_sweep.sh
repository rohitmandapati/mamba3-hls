#!/usr/bin/env bash
# =============================================================================
# bitstream_sweep.sh — Run all bitstreams on the Kria and collect results
# =============================================================================
# Expects to be run FROM the Kria in the directory that deploy_on_kria.sh
# synced to, e.g.:
#   cd ~/mamba3_hls
#   ./bitstream_sweep.sh
#
# Bitstreams are expected alongside this script:  ./<label>.bit
# Host binary is:                                 ./mamba3-host
# Golden data is:                                 ./data/
#
# Results are saved locally and, optionally, copied back to a build machine:
#   RESULTS_HOST        (env var, e.g. my-workstation)
#   RESULTS_REMOTE_DIR  (remote path, e.g. ~/mamba3_hls/important_results/fpga)
#
# Usage:
#   ./bitstream_sweep.sh
#   RESULTS_HOST=my-workstation RESULTS_REMOTE_DIR=~/mamba3_hls/important_results/fpga ./bitstream_sweep.sh
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FPGA_DIR="$SCRIPT_DIR/fpga_results"
mkdir -p "$FPGA_DIR"

# Optional build machine to copy results back to (override via env)
RESULTS_USER="${RESULTS_USER:-$USER}"
RESULTS_HOST="${RESULTS_HOST:-}"
RESULTS_REMOTE_DIR="${RESULTS_REMOTE_DIR:-}"

LABELS=(baseline pragma fixed_pwl fixed_cordic pragma_fixed_pwl pragma_fixed_cordic)

PASS_COUNT=0
FAIL_COUNT=0

for label in "${LABELS[@]}"; do
    bit_file="$SCRIPT_DIR/${label}.bit"
    out_file="$FPGA_DIR/fpga_${label}.txt"
    full_log="$FPGA_DIR/fpga_${label}_full.log"

    echo ""
    echo "================================================================="
    echo " Testing: $label"
    echo "   bitstream: $bit_file"
    echo "================================================================="

    if [[ ! -f "$bit_file" ]]; then
        echo "  -> SKIP: ${label}.bit not found."
        continue
    fi

    # Load the bitstream
    echo "  -> Loading bitstream..."
    sudo load_bitstream "$bit_file"

    # Run the host binary
    echo "  -> Running host..."
    HOST_BIN="$SCRIPT_DIR/mamba3-host"
    if [[ ! -f "$HOST_BIN" ]]; then
        echo "  -> ERROR: host binary not found at $HOST_BIN"
        echo "            Re-run deploy_on_kria.sh to compile it."
        FAIL_COUNT=$(( FAIL_COUNT + 1 ))
        continue
    fi
    "$HOST_BIN" 2>&1 | tee "$full_log" || true

    # Save result summary
    {
        echo "# Config:    $label"
        echo "# Bitstream: $bit_file"
        echo "# Date:      $(date)"
        echo ""
    } > "$out_file"
    grep -A 25 "Global error summary:" "$full_log" >> "$out_file" 2>/dev/null || true
    grep -E "TPOT|PASS|FAIL|total time|msecs" "$full_log" >> "$out_file" 2>/dev/null || true

    if grep -q "PASS" "$full_log" && ! grep -q "FAIL" "$full_log"; then
        echo "  -> PASS"
        PASS_COUNT=$(( PASS_COUNT + 1 ))
    else
        echo "  -> FAIL"
        FAIL_COUNT=$(( FAIL_COUNT + 1 ))
    fi

    # Copy result back to the build machine if configured
    if [[ -n "$RESULTS_HOST" && -n "$RESULTS_REMOTE_DIR" ]]; then
        echo "  -> SSHing result back to $RESULTS_USER@$RESULTS_HOST:$RESULTS_REMOTE_DIR/"
        ssh "$RESULTS_USER@$RESULTS_HOST" "mkdir -p $RESULTS_REMOTE_DIR"
        scp "$out_file" "$RESULTS_USER@$RESULTS_HOST:$RESULTS_REMOTE_DIR/fpga_${label}.txt"
        scp "$full_log" "$RESULTS_USER@$RESULTS_HOST:$RESULTS_REMOTE_DIR/fpga_${label}_full.log"
    fi
done

echo ""
echo "================================================================="
echo " FPGA SWEEP COMPLETE: $PASS_COUNT PASS  $FAIL_COUNT FAIL"
echo " Local results: $FPGA_DIR/"

# SSH entire results dir back at the end
if [[ -n "$RESULTS_HOST" && -n "$RESULTS_REMOTE_DIR" ]]; then
    echo " Final sync to $RESULTS_USER@$RESULTS_HOST:$RESULTS_REMOTE_DIR/ ..."
    rsync -avz "$FPGA_DIR/" "$RESULTS_USER@$RESULTS_HOST:$RESULTS_REMOTE_DIR/"
    echo " Sync complete."
fi
echo "================================================================="
