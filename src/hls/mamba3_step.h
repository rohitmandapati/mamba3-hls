//===========================================================================
// mamba3_step.h
//===========================================================================
// @brief: Header for the single-step Mamba-3 MIMO HLS accelerator.

#ifndef MAMBA3_STEP_H
#define MAMBA3_STEP_H

#include "typedefs.h"
#include "model_config.h"
#include <hls_stream.h>

#ifndef MAMBA3_ENABLE_HLS_PRAGMAS
#define MAMBA3_ENABLE_HLS_PRAGMAS 0
#endif

#ifndef MAMBA3_PIPELINE_II
#define MAMBA3_PIPELINE_II 1
#endif

#ifndef MAMBA3_UNROLL_MIMO
#define MAMBA3_UNROLL_MIMO 1
#endif

#ifndef MAMBA3_NSTEPS
#define MAMBA3_NSTEPS 1
#endif

// -----------------------------------------------------------------------
//  Top-level DUT function (HLS synthesis entry point)
//  Communicates via 32-bit AXI streams, packing floats as uint32.
//  Each invocation consumes exactly one autoregressive step worth of tensors
//  and produces exactly one step worth of outputs.
// -----------------------------------------------------------------------
void dut(hls::stream<bit32_t> &strm_in, hls::stream<bit32_t> &strm_out);

#endif
