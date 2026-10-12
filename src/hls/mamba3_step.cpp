//===========================================================================
// mamba3_step.cpp
//===========================================================================
// @brief: Multi-step top-level HLS DUT for the Mamba-3 selective state update.
//         When NSTEPS == 1 this is functionally identical to the original
//         single-step design.  When NSTEPS > 1, constant tensors (A, D,
//         xproj, outproj, zproj) are loaded once and the state evolves
//         in-place across steps to amortize memory/stream overhead.

#include "mamba3_step.h"
#include "ssm_ops.h"

void dut(hls::stream<bit32_t> &strm_in, hls::stream<bit32_t> &strm_out) {
#pragma HLS INTERFACE ap_fifo port=strm_in
#pragma HLS INTERFACE ap_fifo port=strm_out

    //  Local buffers
    data_t state[STATE_SIZE];
    data_t state_out[STATE_SIZE];
    data_t Bstate_buf[BC_SIZE];
    data_t Xstate_buf[X_SIZE];
    data_t A_buf[AH_SIZE];
    data_t B_buf[BC_SIZE];
    data_t C_buf[BC_SIZE];
    data_t D_buf[D_VEC_SIZE];
    data_t x_buf[X_SIZE];
    data_t dt_buf[AH_SIZE];
    data_t trap_buf[AH_SIZE];
    data_t xproj_buf[PROJ_SIZE];
    data_t outproj_buf[PROJ_SIZE];
    data_t z_buf[X_SIZE];
    data_t zproj_buf[PROJ_SIZE];

    data_t out_buf[X_SIZE];
    data_t out_r[BATCH * MIMO * NHEADS * HDIM];
    data_t alpha[AH_SIZE];
    data_t beta[AH_SIZE];
    data_t gamma[AH_SIZE];

    // Always-on: force HLS to use 2-port BRAM/LUTRAM for large arrays so it
    // does NOT auto-create dozens of parallel copies (which would explode LUTs).
#pragma HLS BIND_STORAGE variable=state      type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=state_out  type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=B_buf      type=ram_2p impl=lutram
#pragma HLS BIND_STORAGE variable=Bstate_buf type=ram_2p impl=lutram
#pragma HLS BIND_STORAGE variable=C_buf      type=ram_2p impl=lutram
#pragma HLS BIND_STORAGE variable=out_r      type=ram_2p impl=bram

#if MAMBA3_ENABLE_HLS_PRAGMAS
#pragma HLS ARRAY_PARTITION variable=alpha    complete dim=1
#pragma HLS ARRAY_PARTITION variable=beta     complete dim=1
#pragma HLS ARRAY_PARTITION variable=gamma    complete dim=1
#pragma HLS ARRAY_PARTITION variable=A_buf    complete dim=1
#pragma HLS ARRAY_PARTITION variable=dt_buf   complete dim=1
#pragma HLS ARRAY_PARTITION variable=trap_buf complete dim=1
#pragma HLS ARRAY_PARTITION variable=D_buf    complete dim=1
    // Bind the small AH/D arrays to LUTRAM so ARRAY_PARTITION doesn't
    // inadvertently infer registers instead of the intended compact storage.
#pragma HLS BIND_STORAGE variable=A_buf    type=ram_1p impl=lutram
#pragma HLS BIND_STORAGE variable=dt_buf   type=ram_1p impl=lutram
#pragma HLS BIND_STORAGE variable=trap_buf type=ram_1p impl=lutram
#pragma HLS BIND_STORAGE variable=D_buf    type=ram_1p impl=lutram
#endif

    float_uint32 conv;

#define READ_ARRAY(arr, size)                    \
    for (int i = 0; i < (size); ++i) {           \
        conv.u = (uint32_t)strm_in.read();       \
        arr[i] = (data_t)conv.f;                 \
    }

    //  Phase 1: Read CONSTANT tensors (loaded once per DUT invocation)
    READ_ARRAY(A_buf, AH_SIZE);
    READ_ARRAY(D_buf, D_VEC_SIZE);
    READ_ARRAY(xproj_buf, PROJ_SIZE);
    READ_ARRAY(outproj_buf, PROJ_SIZE);
    READ_ARRAY(zproj_buf, PROJ_SIZE);

    //  Phase 2: Read initial state
    READ_ARRAY(state, STATE_SIZE);

    //  Phase 3: Multi-step autoregressive loop
    for (int step = 0; step < NSTEPS; ++step) {

        //  Read per-step varying tensors
        READ_ARRAY(Bstate_buf, BC_SIZE);
        READ_ARRAY(Xstate_buf, X_SIZE);
        READ_ARRAY(B_buf, BC_SIZE);
        READ_ARRAY(C_buf, BC_SIZE);
        READ_ARRAY(x_buf, X_SIZE);
        READ_ARRAY(dt_buf, AH_SIZE);
        READ_ARRAY(trap_buf, AH_SIZE);
        READ_ARRAY(z_buf, X_SIZE);

        // Step 1-3: alpha = exp(A * dt), beta = (1 - trap) * dt * alpha,
        // gamma = trap * dt.
        for (int b = 0; b < BATCH; ++b) {
            for (int h = 0; h < NHEADS; ++h) {
#if MAMBA3_ENABLE_HLS_PRAGMAS
#pragma HLS PIPELINE II=MAMBA3_PIPELINE_II
#endif
                int idx = IDX_BH(b, h);
                data_t a_val = A_buf[idx];
                data_t dt_val = dt_buf[idx];
                data_t trap_val = trap_buf[idx];

                alpha[idx] = ssm_exp(a_val * dt_val);
                beta[idx] = ((data_t)1.0f - trap_val) * dt_val * alpha[idx];
                gamma[idx] = trap_val * dt_val;
            }
        }

        // Step 4-7: state update.
        for (int b = 0; b < BATCH; ++b) {
            for (int h = 0; h < NHEADS; ++h) {
                int bh = IDX_BH(b, h);
                data_t alpha_val = alpha[bh];
                data_t beta_val = beta[bh];
                data_t gamma_val = gamma[bh];

                for (int d = 0; d < HDIM; ++d) {
                    data_t x_base = x_buf[IDX_BHD(b, h, d)];
                    data_t xs_base = Xstate_buf[IDX_BHD(b, h, d)];
                    data_t x_term[MIMO];
                    data_t xs_term[MIMO];

#if MAMBA3_ENABLE_HLS_PRAGMAS && MAMBA3_UNROLL_MIMO
#pragma HLS ARRAY_PARTITION variable=x_term complete dim=1
#pragma HLS ARRAY_PARTITION variable=xs_term complete dim=1
#endif

                    for (int r = 0; r < MIMO; ++r) {
#if MAMBA3_ENABLE_HLS_PRAGMAS && MAMBA3_UNROLL_MIMO
#pragma HLS UNROLL
#endif
                        data_t proj = xproj_buf[IDX_RHD(r, h, d)];
                        x_term[r] = x_base * proj * gamma_val;
                        xs_term[r] = xs_base * proj * beta_val;
                    }

                    for (int n = 0; n < DSTATE; ++n) {
                        // Always pipeline the inner N-loop to prevent HLS from
                        // loop-flattening the full b*h*d*n space into
                        // one massive parallel resource blob in the baseline.
#pragma HLS PIPELINE II=MAMBA3_PIPELINE_II
                        data_t new_val = state[IDX_BHDN(b, h, d, n)] * alpha_val;

                        for (int r = 0; r < MIMO; ++r) {
#if MAMBA3_ENABLE_HLS_PRAGMAS && MAMBA3_UNROLL_MIMO
#pragma HLS UNROLL
#endif
                            new_val += x_term[r] * B_buf[IDX_BRHN(b, r, h, n)];
                            new_val += xs_term[r] * Bstate_buf[IDX_BRHN(b, r, h, n)];
                        }

                        state_out[IDX_BHDN(b, h, d, n)] = new_val;
                    }
                }
            }
        }

        // Step 8: C projection.
        for (int b = 0; b < BATCH; ++b) {
            for (int r = 0; r < MIMO; ++r) {
                for (int h = 0; h < NHEADS; ++h) {
                    for (int d = 0; d < HDIM; ++d) {
                        // Pipeline the d-loop always so the n-accumulation inside
                        // is serialized rather than unrolled into parallel adders.
#pragma HLS PIPELINE II=MAMBA3_PIPELINE_II
                        data_t accum = (data_t)0.0f;
                        for (int n = 0; n < DSTATE; ++n) {
                            accum += state_out[IDX_BHDN(b, h, d, n)] * C_buf[IDX_BRHN(b, r, h, n)];
                        }
                        out_r[IDX_BRHD(b, r, h, d)] = accum;
                    }
                }
            }
        }

        // Step 9: skip connection.
        for (int b = 0; b < BATCH; ++b) {
            for (int r = 0; r < MIMO; ++r) {
                for (int h = 0; h < NHEADS; ++h) {
                    data_t d_val = D_buf[h];
                    for (int d = 0; d < HDIM; ++d) {
#pragma HLS PIPELINE II=MAMBA3_PIPELINE_II
                        data_t x_val = x_buf[IDX_BHD(b, h, d)] * xproj_buf[IDX_RHD(r, h, d)];
                        out_r[IDX_BRHD(b, r, h, d)] += d_val * x_val;
                    }
                }
            }
        }

        // Step 10-11: SiLU gate.
        for (int b = 0; b < BATCH; ++b) {
            for (int r = 0; r < MIMO; ++r) {
                for (int h = 0; h < NHEADS; ++h) {
                    for (int d = 0; d < HDIM; ++d) {
#pragma HLS PIPELINE II=MAMBA3_PIPELINE_II
                        data_t z_val = z_buf[IDX_BHD(b, h, d)] * zproj_buf[IDX_RHD(r, h, d)];
                        out_r[IDX_BRHD(b, r, h, d)] *= ssm_silu(z_val);
                    }
                }
            }
        }

        // Step 12: reduce MIMO rank back to (B, H, D).
        for (int b = 0; b < BATCH; ++b) {
            for (int h = 0; h < NHEADS; ++h) {
                for (int d = 0; d < HDIM; ++d) {
#pragma HLS PIPELINE II=MAMBA3_PIPELINE_II
                    data_t accum = (data_t)0.0f;
                    for (int r = 0; r < MIMO; ++r) {
#if MAMBA3_ENABLE_HLS_PRAGMAS && MAMBA3_UNROLL_MIMO
#pragma HLS UNROLL
#endif
                        accum += out_r[IDX_BRHD(b, r, h, d)] * outproj_buf[IDX_RHD(r, h, d)];
                    }
                    out_buf[IDX_BHD(b, h, d)] = accum;
                }
            }
        }

        // Write this step's output immediately (pipeline-friendly).
#define WRITE_ARRAY(arr, size)                    \
    for (int i = 0; i < (size); ++i) {            \
        conv.f = (float)arr[i];                   \
        strm_out.write((bit32_t)conv.u);          \
    }

        WRITE_ARRAY(out_buf, X_SIZE);

        // Carry state forward: state_out → state for next step.
        for (int i = 0; i < STATE_SIZE; ++i) {
#pragma HLS PIPELINE II=1
            state[i] = state_out[i];
        }

    } // end multi-step loop

    //  Phase 4: Write final state_out
    WRITE_ARRAY(state_out, STATE_SIZE);

#undef READ_ARRAY
#undef WRITE_ARRAY
}
