//===========================================================================
// mamba3_step_test.cpp
//===========================================================================
// @brief: Testbench for the Mamba-3 MIMO step function HLS accelerator.
//
// Supports multi-step batching (NSTEPS > 1):
//   Data directory layout: seq_Y/step_Z/  (or sample_X/ for NSTEPS==1 compat)
//   Constants are streamed once per sequence; per-step inputs for each step.
//   DUT outputs one out per step, then final state_out.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>

#include "mamba3_step.h"
#include "timer.h"

using namespace std;

static bool load_bin(const char *name, float *arr, int count) {
    string path1 = string("data/") + name;
    ifstream f1(path1.c_str(), ios::binary);
    if (f1.is_open()) {
        f1.read(reinterpret_cast<char *>(arr), count * sizeof(float));
        if (f1.gcount() == (streamsize)(count * sizeof(float))) {
            cout << "  Loaded " << path1 << endl;
            return true;
        }
    }

    string path2 = string("src/hls/data/") + name;
    ifstream f2(path2.c_str(), ios::binary);
    if (f2.is_open()) {
        f2.read(reinterpret_cast<char *>(arr), count * sizeof(float));
        if (f2.gcount() == (streamsize)(count * sizeof(float))) {
            cout << "  Loaded " << path2 << endl;
            return true;
        }
    }

    cerr << "  ERROR: Could not load " << name << endl;
    return false;
}

static bool file_exists(const char *name) {
    string path1 = string("data/") + name;
    string path2 = string("src/hls/data/") + name;
    ifstream f1(path1.c_str(), ios::binary);
    ifstream f2(path2.c_str(), ios::binary);
    return f1.is_open() || f2.is_open();
}

int main() {
    cout << "========================================" << endl;
    cout << " Mamba-3 Step Function - HLS Testbench" << endl;
    cout << "========================================" << endl;
    cout << "Config: B=" << BATCH << " H=" << NHEADS
         << " D=" << HDIM << " N=" << DSTATE
         << " R=" << MIMO << " NSTEPS=" << NSTEPS << endl << endl;

    // ------------------------------------------------------------------
    //  Detect data layout: seq_Y/ (multi-step) or sample_X/ (legacy)
    // ------------------------------------------------------------------
    bool use_seq_layout = false;
    int num_sequences = 0;

    // First check for seq_Y/ layout
    while (true) {
        char path[256];
        sprintf(path, "seq_%d/step_0/x.bin", num_sequences);
        if (file_exists(path)) {
            use_seq_layout = true;
            ++num_sequences;
        } else {
            break;
        }
    }

    // Fall back to sample_X/ layout (legacy, NSTEPS==1 only)
    if (!use_seq_layout) {
        while (true) {
            char path[256];
            sprintf(path, "sample_%d/state.bin", num_sequences);
            if (file_exists(path)) {
                ++num_sequences;
            } else {
                break;
            }
        }
    }

    if (num_sequences == 0) {
        cerr << "\nERROR: Missing input files! No seq_X or sample_X directories found." << endl;
        cerr << "Run generate_golden.py first," << endl;
        cerr << "then copy the data/ directory here if needed." << endl;
        return 1;
    }

    if (!use_seq_layout && NSTEPS > 1) {
        cerr << "\nERROR: NSTEPS=" << NSTEPS << " but data uses legacy sample_X/ layout." << endl;
        cerr << "Regenerate golden data with: python3 generate_golden.py --nsteps=" << NSTEPS << endl;
        return 1;
    }

    if (use_seq_layout) {
        char path_check[256];
        sprintf(path_check, "seq_0/step_%d/x.bin", NSTEPS);
        if (file_exists(path_check)) {
            cerr << "\nERROR: Compiled with MAMBA3_NSTEPS=" << NSTEPS 
                 << " but golden data has MORE steps! (e.g. step_" << NSTEPS << " exists)" << endl;
            cerr << "Run `make csim MAMBA3_NSTEPS=<match_golden_nsteps>`." << endl;
            return 1;
        }
        sprintf(path_check, "seq_0/step_%d/x.bin", NSTEPS - 1);
        if (!file_exists(path_check)) {
            cerr << "\nERROR: Compiled with MAMBA3_NSTEPS=" << NSTEPS 
                 << " but golden data has FEWER steps! (step_" << (NSTEPS - 1) << " is missing)" << endl;
            cerr << "Regenerate golden data or adjust your build flags." << endl;
            return 1;
        }
    }

    cout << "Data layout: " << (use_seq_layout ? "seq_Y/step_Z/" : "sample_X/ (legacy)")
         << ", " << num_sequences << " sequence(s), " << NSTEPS << " step(s) each." << endl;

    // ------------------------------------------------------------------
    //  Buffers
    // ------------------------------------------------------------------
    float A[AH_SIZE], D_vec[D_VEC_SIZE];
    float xproj[PROJ_SIZE], outproj[PROJ_SIZE], zproj[PROJ_SIZE];
    float state[STATE_SIZE];
    float Bstate[BC_SIZE], Xstate[X_SIZE];
    float B_mat[BC_SIZE], C_mat[BC_SIZE];
    float x[X_SIZE], dt_arr[AH_SIZE], trap_arr[AH_SIZE];
    float z[X_SIZE];

    float py_state_out[STATE_SIZE];
    float py_out[X_SIZE];
    float dut_state_out[STATE_SIZE];
    float dut_out[X_SIZE];

    const float threshold = 5e-1f;
    bool all_pass = true;

    float global_max_err_state = 0.0f;
    float global_max_err_out = 0.0f;
    double global_abs_err_state = 0.0;
    double global_abs_err_out = 0.0;
    double global_sq_err_state = 0.0;
    double global_sq_err_out = 0.0;
    long long total_state_elems = 0;
    long long total_out_elems = 0;

    cout << "\nRunning DUT..." << endl;
    Timer timer("mamba3_step DUT");
    timer.start();

    for (int seq = 0; seq < num_sequences; ++seq) {
        hls::stream<bit32_t> strm_in;
        hls::stream<bit32_t> strm_out;
        float_uint32 conv;
        char path[256];

        cout << "\n>> Sequence " << seq << endl;

        // Helper to stream an array into strm_in
#define LOAD_STREAM(file, arr, size)                    \
        if (!load_bin(path, arr, size)) return 1;       \
        for (int i = 0; i < (size); ++i) {              \
            conv.f = arr[i];                            \
            strm_in.write((bit32_t)conv.u);             \
        }

        // --- Load and stream constant tensors ---
        if (use_seq_layout) {
            sprintf(path, "seq_%d/A.bin", seq);
        } else {
            sprintf(path, "sample_%d/A.bin", seq);
        }
        LOAD_STREAM(path, A, AH_SIZE);

        if (use_seq_layout) {
            sprintf(path, "seq_%d/D.bin", seq);
        } else {
            sprintf(path, "sample_%d/D.bin", seq);
        }
        LOAD_STREAM(path, D_vec, D_VEC_SIZE);

        if (use_seq_layout) {
            sprintf(path, "seq_%d/xproj.bin", seq);
        } else {
            sprintf(path, "sample_%d/xproj.bin", seq);
        }
        LOAD_STREAM(path, xproj, PROJ_SIZE);

        if (use_seq_layout) {
            sprintf(path, "seq_%d/outproj.bin", seq);
        } else {
            sprintf(path, "sample_%d/outproj.bin", seq);
        }
        LOAD_STREAM(path, outproj, PROJ_SIZE);

        if (use_seq_layout) {
            sprintf(path, "seq_%d/zproj.bin", seq);
        } else {
            sprintf(path, "sample_%d/zproj.bin", seq);
        }
        LOAD_STREAM(path, zproj, PROJ_SIZE);

        // --- Load and stream initial state ---
        if (use_seq_layout) {
            sprintf(path, "seq_%d/step_0/state.bin", seq);
        } else {
            sprintf(path, "sample_%d/state.bin", seq);
        }
        LOAD_STREAM(path, state, STATE_SIZE);

        // --- Load and stream per-step varying tensors ---
        for (int step = 0; step < NSTEPS; ++step) {
            if (use_seq_layout) {
                sprintf(path, "seq_%d/step_%d/Bstate.bin", seq, step);
            } else {
                sprintf(path, "sample_%d/Bstate.bin", seq);
            }
            LOAD_STREAM(path, Bstate, BC_SIZE);

            if (use_seq_layout) {
                sprintf(path, "seq_%d/step_%d/Xstate.bin", seq, step);
            } else {
                sprintf(path, "sample_%d/Xstate.bin", seq);
            }
            LOAD_STREAM(path, Xstate, X_SIZE);

            if (use_seq_layout) {
                sprintf(path, "seq_%d/step_%d/B.bin", seq, step);
            } else {
                sprintf(path, "sample_%d/B.bin", seq);
            }
            LOAD_STREAM(path, B_mat, BC_SIZE);

            if (use_seq_layout) {
                sprintf(path, "seq_%d/step_%d/C.bin", seq, step);
            } else {
                sprintf(path, "sample_%d/C.bin", seq);
            }
            LOAD_STREAM(path, C_mat, BC_SIZE);

            if (use_seq_layout) {
                sprintf(path, "seq_%d/step_%d/x.bin", seq, step);
            } else {
                sprintf(path, "sample_%d/x.bin", seq);
            }
            LOAD_STREAM(path, x, X_SIZE);

            if (use_seq_layout) {
                sprintf(path, "seq_%d/step_%d/dt.bin", seq, step);
            } else {
                sprintf(path, "sample_%d/dt.bin", seq);
            }
            LOAD_STREAM(path, dt_arr, AH_SIZE);

            if (use_seq_layout) {
                sprintf(path, "seq_%d/step_%d/trap.bin", seq, step);
            } else {
                sprintf(path, "sample_%d/trap.bin", seq);
            }
            LOAD_STREAM(path, trap_arr, AH_SIZE);

            if (use_seq_layout) {
                sprintf(path, "seq_%d/step_%d/z.bin", seq, step);
            } else {
                sprintf(path, "sample_%d/z.bin", seq);
            }
            LOAD_STREAM(path, z, X_SIZE);
        }

#undef LOAD_STREAM

        // --- Run DUT ---
        dut(strm_in, strm_out);

        // --- Read and compare per-step outputs ---
        for (int step = 0; step < NSTEPS; ++step) {
            if (use_seq_layout) {
                sprintf(path, "seq_%d/step_%d/golden_out.bin", seq, step);
            } else {
                sprintf(path, "sample_%d/golden_out.bin", seq);
            }
            if (!load_bin(path, py_out, X_SIZE)) return 1;

            float step_max_err_out = 0.0f;
            double step_abs_err_out = 0.0;
            double step_sq_err_out = 0.0;

            for (int i = 0; i < X_SIZE; ++i) {
                conv.u = (uint32_t)strm_out.read();
                dut_out[i] = conv.f;

                float err = fabsf(dut_out[i] - py_out[i]);
                if (err > step_max_err_out) {
                    step_max_err_out = err;
                }
                step_abs_err_out += err;
                step_sq_err_out += (double)err * (double)err;
            }

            global_max_err_out = std::max(global_max_err_out, step_max_err_out);
            global_abs_err_out += step_abs_err_out;
            global_sq_err_out += step_sq_err_out;
            total_out_elems += X_SIZE;

            cout << "  Seq " << seq << " Step " << step
                 << " | out max err: " << step_max_err_out
                 << " | out mean err: " << (step_abs_err_out / X_SIZE)
                 << endl;

            if (step_max_err_out > threshold) {
                all_pass = false;
            }
        }

        // --- Read and compare final state_out ---
        if (use_seq_layout) {
            sprintf(path, "seq_%d/golden_state_out.bin", seq);
        } else {
            sprintf(path, "sample_%d/golden_state_out.bin", seq);
        }
        if (!load_bin(path, py_state_out, STATE_SIZE)) return 1;

        float seq_max_err_state = 0.0f;
        double seq_abs_err_state = 0.0;
        double seq_sq_err_state = 0.0;

        for (int i = 0; i < STATE_SIZE; ++i) {
            conv.u = (uint32_t)strm_out.read();
            dut_state_out[i] = conv.f;

            float err = fabsf(dut_state_out[i] - py_state_out[i]);
            if (err > seq_max_err_state) {
                seq_max_err_state = err;
            }
            seq_abs_err_state += err;
            seq_sq_err_state += (double)err * (double)err;
        }

        global_max_err_state = std::max(global_max_err_state, seq_max_err_state);
        global_abs_err_state += seq_abs_err_state;
        global_sq_err_state += seq_sq_err_state;
        total_state_elems += STATE_SIZE;

        cout << "  Seq " << seq
             << " | final state max err: " << seq_max_err_state
             << " | state mean err: " << (seq_abs_err_state / STATE_SIZE)
             << endl;

        if (seq_max_err_state > threshold) {
            all_pass = false;
        }
    }

    timer.stop();

    cout << "\nGlobal error summary:" << endl;
    cout << "  state_out max / mean / rmse: "
         << global_max_err_state << " / "
         << (global_abs_err_state / total_state_elems) << " / "
         << std::sqrt(global_sq_err_state / total_state_elems) << endl;
    cout << "  out       max / mean / rmse: "
         << global_max_err_out << " / "
         << (global_abs_err_out / total_out_elems) << " / "
         << std::sqrt(global_sq_err_out / total_out_elems) << endl;

    cout << endl;
    if (all_pass) {
        cout << "*** PASS *** (threshold=" << threshold
             << ", " << num_sequences << " sequence(s) x "
             << NSTEPS << " step(s) verified)" << endl;
    } else {
        cout << "*** FAIL *** (threshold=" << threshold << ")" << endl;
    }

    return all_pass ? 0 : 1;
}
