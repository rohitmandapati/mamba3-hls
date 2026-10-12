//===========================================================================
// model_config.h
//===========================================================================
// @brief: Model configuration for the Mamba-3 MIMO step function.
//         D must be >= 64 to match the CUDA kernel's default tile_D.
//         Adjust as needed for your target model configuration.

#ifndef MODEL_CONFIG_H
#define MODEL_CONFIG_H

// -----------------------------------------------------------------------
//  Primary model dimensions
// -----------------------------------------------------------------------
//  B: batch size           (number of sequences processed in parallel)
//  H: number of SSM heads
//  D: head dimension       (per-head hidden size)
//  N: state dimension      (dstate — SSM recurrence width)
//  R: MIMO rank            (number of input/output projections)
// -----------------------------------------------------------------------
static const int BATCH  = 1;
static const int NHEADS = 4;
static const int HDIM   = 64;
static const int DSTATE = 32;
static const int MIMO   = 2;

// -----------------------------------------------------------------------
//  Multi-step batching
// -----------------------------------------------------------------------
//  NSTEPS: number of autoregressive steps per DUT invocation.
//  When NSTEPS > 1, constant tensors (A, D, xproj, outproj, zproj)
//  are streamed once, state evolves in-place, and only per-step
//  varying tensors are re-streamed for each step.
// -----------------------------------------------------------------------
#ifndef MAMBA3_NSTEPS
#define MAMBA3_NSTEPS 1
#endif
static const int NSTEPS = MAMBA3_NSTEPS;

// -----------------------------------------------------------------------
//  Derived sizes (total number of elements per tensor)
// -----------------------------------------------------------------------
// state:   (B, H, D, N)
static const int STATE_SIZE = BATCH * NHEADS * HDIM * DSTATE;

// Bstate, B, C: (B, R, H, N)
static const int BC_SIZE = BATCH * MIMO * NHEADS * DSTATE;

// x, Xstate, z, out: (B, H, D)
static const int X_SIZE = BATCH * NHEADS * HDIM;

// A, dt, trap: (B, H)
static const int AH_SIZE = BATCH * NHEADS;

// xproj, zproj, outproj: (R, H, D)
static const int PROJ_SIZE = MIMO * NHEADS * HDIM;

// D_vec: (H)
static const int D_VEC_SIZE = NHEADS;

// -----------------------------------------------------------------------
//  Input/output stream sizes for multi-step batching
// -----------------------------------------------------------------------

// Constant tensors (loaded once per DUT invocation):
//   A (AH_SIZE) + D (D_VEC_SIZE) + xproj (PROJ_SIZE) +
//   outproj (PROJ_SIZE) + zproj (PROJ_SIZE)
static const int CONSTANT_INPUT_FLOATS =
    AH_SIZE +      // A
    D_VEC_SIZE +   // D
    PROJ_SIZE +    // xproj
    PROJ_SIZE +    // outproj
    PROJ_SIZE;     // zproj

// Per-step varying tensors:
//   Bstate (BC_SIZE) + Xstate (X_SIZE) + B (BC_SIZE) + C (BC_SIZE) +
//   x (X_SIZE) + dt (AH_SIZE) + trap (AH_SIZE) + z (X_SIZE)
static const int PER_STEP_INPUT_FLOATS =
    BC_SIZE +      // Bstate
    X_SIZE +       // Xstate
    BC_SIZE +      // B
    BC_SIZE +      // C
    X_SIZE +       // x
    AH_SIZE +      // dt
    AH_SIZE +      // trap
    X_SIZE;        // z

// Total input = constants + initial state + NSTEPS * per-step
static const int TOTAL_INPUT_FLOATS =
    CONSTANT_INPUT_FLOATS +
    STATE_SIZE +   // initial state (loaded once)
    NSTEPS * PER_STEP_INPUT_FLOATS;

// Total output = NSTEPS * out + final state_out
static const int TOTAL_OUTPUT_FLOATS =
    NSTEPS * X_SIZE +  // one out per step
    STATE_SIZE;        // final state_out

#endif
