#!/usr/bin/env python3
"""
generate_golden.py
==================
Runs the pure-PyTorch reference function for the Mamba-3 selective state
update and saves all inputs and golden outputs as raw fp32 binary files
for the C++ HLS testbench.

Supports multi-step autoregressive generation (--nsteps):
  When nsteps > 1, each sequence runs nsteps consecutive reference calls,
  feeding state_out from step t as state for step t+1 (mimicking real
  autoregressive inference).  Constant tensors are saved once per sequence;
  per-step tensors are saved in seq_Y/step_Z/ subdirectories.

  When nsteps == 1, falls back to the legacy sample_X/ layout for
  backward compatibility.

Usage:
    python3 generate_golden.py
    python3 generate_golden.py --nsteps=4
    python3 generate_golden.py --nsteps=64 --samples=3
"""

import os
import argparse
import numpy as np
import sys
import torch
from mamba3_step_fn import selective_state_update_fused_ref_v2


def save_tensor(path, tensor):
    """Save a tensor as raw fp32 binary (row-major, C-contiguous)."""
    arr = tensor.detach().cpu().float().contiguous().numpy().flatten()
    with open(path, "wb") as f:
        f.write(arr.tobytes())
    print(f"  Saved {path} ({arr.size} floats, {arr.size*4} bytes)")


def main():
    parser = argparse.ArgumentParser(description="Generate golden data for Mamba-3 HLS testbench")
    parser.add_argument("--B", type=int, default=1,  help="Batch size")
    parser.add_argument("--H", type=int, default=4,  help="Number of heads")
    parser.add_argument("--D", type=int, default=64, help="Head dimension")
    parser.add_argument("--N", type=int, default=32, help="State dimension (dstate)")
    parser.add_argument("--R", type=int, default=2,  help="MIMO rank")
    parser.add_argument("--outdir", type=str, default="data", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--samples", type=int, default=5, help="Number of sequences to generate")
    parser.add_argument("--nsteps", type=int, default=1, help="Steps per sequence (autoregressive)")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device (cpu or cuda)")
    args = parser.parse_args()

    B, H, D, N, R = args.B, args.H, args.D, args.N, args.R
    nsteps = args.nsteps
    device = args.device
    os.makedirs(args.outdir, exist_ok=True)

    print("=" * 60)
    print(" Mamba-3 Golden Reference Generator (PyTorch)")
    print("=" * 60)
    print(f"Config: B={B} H={H} D={D} N={N} R={R} NSTEPS={nsteps}")
    print(f"Device: {device}")
    print(f"Output: {args.outdir}/")
    print()

    torch.manual_seed(args.seed)
    dtype = torch.float32

    # ------------------------------------------------------------------
    #  Generate sequences
    # ------------------------------------------------------------------
    for s in range(args.samples):
        print(f"\n[{s+1}/{args.samples}] Generating Sequence {s} ({nsteps} step(s))...")

        if nsteps == 1:
            seq_dir = os.path.join(args.outdir, f"sample_{s}")
        else:
            seq_dir = os.path.join(args.outdir, f"seq_{s}")
        os.makedirs(seq_dir, exist_ok=True)

        # --- Constant tensors (same across all steps) ---
        A       = -torch.abs(torch.randn((B, H), device=device, dtype=torch.float32)) * 0.5
        D_vec   = torch.randn((H,),         device=device, dtype=torch.float32)
        xproj   = torch.randn((R, H, D),    device=device, dtype=dtype)
        outproj = torch.randn((R, H, D),    device=device, dtype=dtype)
        zproj   = torch.randn((R, H, D),    device=device, dtype=dtype)

        save_tensor(os.path.join(seq_dir, "A.bin"),       A)
        save_tensor(os.path.join(seq_dir, "D.bin"),       D_vec)
        save_tensor(os.path.join(seq_dir, "xproj.bin"),   xproj)
        save_tensor(os.path.join(seq_dir, "outproj.bin"), outproj)
        save_tensor(os.path.join(seq_dir, "zproj.bin"),   zproj)

        # --- Initial state ---
        state = torch.randn((B, H, D, N), device=device, dtype=dtype)

        # --- Autoregressive loop ---
        for step in range(nsteps):
            if nsteps == 1:
                step_dir = seq_dir
            else:
                step_dir = os.path.join(seq_dir, f"step_{step}")
                os.makedirs(step_dir, exist_ok=True)

            print(f"  Step {step}:")

            # Save state input for step 0
            if step == 0:
                save_tensor(os.path.join(step_dir, "state.bin"), state)

            # Per-step varying tensors
            Bstate = torch.randn((B, R, H, N), device=device, dtype=dtype)
            Xstate = torch.randn((B, H, D),    device=device, dtype=dtype)
            B_mat  = torch.randn((B, R, H, N), device=device, dtype=dtype)
            C_mat  = torch.randn((B, R, H, N), device=device, dtype=dtype)
            x      = torch.randn((B, H, D),    device=device, dtype=dtype)
            dt     = torch.abs(torch.randn((B, H), device=device, dtype=torch.float32)) * 0.05 + 0.01
            trap   = torch.sigmoid(torch.randn((B, H), device=device, dtype=torch.float32))
            z      = torch.randn((B, H, D),    device=device, dtype=dtype)

            save_tensor(os.path.join(step_dir, "Bstate.bin"), Bstate)
            save_tensor(os.path.join(step_dir, "Xstate.bin"), Xstate)
            save_tensor(os.path.join(step_dir, "B.bin"),      B_mat)
            save_tensor(os.path.join(step_dir, "C.bin"),      C_mat)
            save_tensor(os.path.join(step_dir, "x.bin"),      x)
            save_tensor(os.path.join(step_dir, "dt.bin"),     dt)
            save_tensor(os.path.join(step_dir, "trap.bin"),   trap)
            save_tensor(os.path.join(step_dir, "z.bin"),      z)

            # Run reference
            state_input = state.clone()
            ref_out, ref_state_out = selective_state_update_fused_ref_v2(
                state_input, A, B_mat, C_mat, xproj, x, zproj, z, dt, Bstate, Xstate, trap, D_vec, outproj,
                compute_dtype=torch.float32
            )

            # Save per-step golden output
            save_tensor(os.path.join(step_dir, "golden_out.bin"), ref_out)

            # Feed state forward for next step (autoregressive)
            state = ref_state_out

        # Save the final golden state_out (after all steps)
        save_tensor(os.path.join(seq_dir, "golden_state_out.bin"), state)

        config_path = os.path.join(seq_dir, "config.txt")
        with open(config_path, "w") as f:
            f.write(f"{B} {H} {D} {N} {R} {nsteps}\n")

    print("\nDone! Copy the data/ directory to your HLS project.")


if __name__ == "__main__":
    main()
