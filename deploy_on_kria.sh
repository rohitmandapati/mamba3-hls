#!/bin/bash
# =============================================================================
# deploy_on_kria.sh
# =============================================================================
# Builds kria_dir/ fresh from the current project state, then rsyncs it to
# the specified Kria board and compiles the host binary natively on ARM.
#
# kria_dir/ contents:
#   *.bit              — all bitstreams from important_results/bitstreams/
#   src_kria/          — host source (host.cpp + src/hls headers)
#   data/              — golden reference data (from kria_dir_data/ or data/)
#   bitstream_sweep.sh — sweep runner
#
# Usage:
#   ./deploy_on_kria.sh <board-ip-or-hostname>
#   KRIA_USER=ubuntu ./deploy_on_kria.sh 192.168.1.50
# =============================================================================

KRIA_USER="${KRIA_USER:-ubuntu}"

function usage() {
    echo "Usage: $0 <board-ip-or-hostname>"
    echo "  Env: KRIA_USER (default: ubuntu)"
}

if [ -z "$1" ]; then echo "Error: No board address provided."; usage; exit 1; fi
KRIA_NAME="$1"
KRIA_IP="$1"
if ! ping -c 1 "$KRIA_IP" &>/dev/null; then echo "Error: $KRIA_IP not reachable."; exit 1; fi

# =============================================================================
# Build kria_dir/ from the current project state
# =============================================================================
BIT_DIR="important_results/bitstreams"
DEPLOY_KRIA="kria_dir"

echo "================================================================="
echo " Building kria_dir/ from current project state..."
echo "================================================================="

rm -rf "$DEPLOY_KRIA"
mkdir -p "$DEPLOY_KRIA/src_kria"

# 1. All bitstreams
LABELS=(baseline pragma fixed_pwl fixed_cordic pragma_fixed_pwl pragma_fixed_cordic)
for label in "${LABELS[@]}"; do
    src="$BIT_DIR/${label}.bit"
    if [[ -f "$src" ]]; then
        cp "$src" "$DEPLOY_KRIA/${label}.bit"
        echo "  -> ${label}.bit"
    else
        echo "  -> MISSING: ${label}.bit"
    fi
done

# 2. Host source (latest host.cpp + all headers)
if [[ -d "src/kria" ]]; then
    cp src/kria/* "$DEPLOY_KRIA/src_kria/"
fi
if [[ -d "src/hls" ]]; then
    cp src/hls/*.h "$DEPLOY_KRIA/src_kria/" 2>/dev/null || true
fi

# 3. Golden reference data — host.cpp looks for data/seq_N/step_M/x.bin
mkdir -p "$DEPLOY_KRIA/data"
for data_src in data src/hls/data; do
    if [[ -d "$data_src" ]] && ls "$data_src"/seq_* &>/dev/null 2>&1; then
        echo "  -> Copying seq_* from $data_src/ into kria_dir/data/ ..."
        cp -r "$data_src"/seq_* "$DEPLOY_KRIA/data/"
        break
    fi
done

# 5. Python prompt script + mamba_ssm package
if [[ -f "prompt.py" ]]; then
    cp prompt.py "$DEPLOY_KRIA/prompt.py"
    echo "  -> prompt.py"
else
    echo "  -> WARNING: prompt.py not found, skipping"
fi

if [[ -d "src/hls/mamba_ssm" ]]; then
    cp -r "src/hls/mamba_ssm" "$DEPLOY_KRIA/mamba_ssm"
    echo "  -> mamba_ssm/"
else
    echo "  -> WARNING: src/hls/mamba_ssm not found, skipping"
fi

# 6. Optional: pre-trained weights (one level up from project root)
WEIGHTS_SRC="../weights.pt"
if [[ -f "$WEIGHTS_SRC" ]]; then
    cp "$WEIGHTS_SRC" "$DEPLOY_KRIA/weights.pt"
    echo "  -> weights.pt"
else
    echo "  -> No weights.pt found at $WEIGHTS_SRC — board will use random weights"
fi

# 4. bitstream_sweep.sh
cp bitstream_sweep.sh "$DEPLOY_KRIA/bitstream_sweep.sh"
chmod +x "$DEPLOY_KRIA/bitstream_sweep.sh"

echo ""
echo " kria_dir/ contents:"
ls -lh "$DEPLOY_KRIA/"
echo "================================================================="

# =============================================================================
# Deploy to Kria
# =============================================================================
CURRENT_DIR_NAME=$(basename "$PWD")
REMOTE_DIR="/home/$KRIA_USER/$CURRENT_DIR_NAME"

echo ""
echo "Deploying to $KRIA_USER@$KRIA_IP -> $REMOTE_DIR/ ..."
ssh "$KRIA_USER@$KRIA_IP" "mkdir -p $REMOTE_DIR"
rsync -avzL --progress "$DEPLOY_KRIA/" "$KRIA_USER@$KRIA_IP":"$REMOTE_DIR/"

# Also sync artifacts/kria/ if present
if [[ -d "artifacts/kria" ]]; then
    rsync -avzL artifacts/kria/ "$KRIA_USER@$KRIA_IP":"$REMOTE_DIR/"
fi

# Compile host binary natively on the Kria
echo ""
echo "Compiling host binary on $KRIA_NAME ..."
ssh "$KRIA_USER@$KRIA_IP" "
    set -e
    cd '$REMOTE_DIR/src_kria'
    make clean 2>/dev/null || true
    make host
    cp mamba3-host '$REMOTE_DIR/mamba3-host'
    echo '  -> mamba3-host compiled and installed to $REMOTE_DIR'
"

# Install Python dependencies for prompt.py on the Kria
echo ""
echo "Installing Python dependencies on $KRIA_NAME ..."
ssh "$KRIA_USER@$KRIA_IP" "
    set -e
    python3 -m pip install --quiet --break-system-packages torch --index-url https://download.pytorch.org/whl/cpu 2>/dev/null \
        || python3 -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu 2>/dev/null \
        || echo '  [pip] torch already installed or pip unavailable — skipping'
    echo '  -> Python deps OK'
"

echo ""
echo "================================================================="
echo " Deploy complete! On the Kria, run:"
echo "   cd $REMOTE_DIR && ./bitstream_sweep.sh"
echo ""
echo " To run interactive text generation:"
echo "   cd $REMOTE_DIR && python3 prompt.py"
echo "================================================================="