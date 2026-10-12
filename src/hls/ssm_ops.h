//===========================================================================
// ssm_ops.h
//===========================================================================
// @brief: Helper functions and index utilities for the Mamba-3 step kernel.
//         Exact math is preserved for the fp32 baseline; fixed-point mode
//         defaults to range-reduced / piecewise-linear nonlinearities so the
//         design can be explored without hls::exp/tanh on ap_fixed.

#ifndef SSM_OPS_H
#define SSM_OPS_H

#include "typedefs.h"
#include "model_config.h"
#include <cmath>
#ifdef __SYNTHESIS__
#include <hls_math.h>
#endif

#if MAMBA3_USE_PWL_NONLINEARITIES
#include "ssm_pwl_tables.h"
#endif

// -----------------------------------------------------------------------
//  Math helpers
// -----------------------------------------------------------------------

#if MAMBA3_USE_PWL_NONLINEARITIES

static const int SSM_EXP_MAX_SHIFT = 14;

#if MAMBA3_USE_FIXED_POINT
typedef ap_fixed<32, 16, AP_RND_CONV, AP_SAT> ssm_lut_math_t;
#else
typedef data_t ssm_lut_math_t;
#endif

inline int ssm_floor_to_int(data_t x) {
    int xi = (int)x;
    if ((data_t)xi > x) {
        --xi;
    }
    return xi;
}

template <int TABLE_SIZE>
inline data_t ssm_lookup_uniform_positive(
    data_t x,
    data_t x_max,
    const float (&table)[TABLE_SIZE]
) {
    const int segments = TABLE_SIZE - 1;

    if (x <= (data_t)0.0f) {
        return (data_t)table[0];
    }
    if (x >= x_max) {
        return (data_t)table[segments];
    }

    const ssm_lut_math_t inv_step =
        (ssm_lut_math_t)segments / (ssm_lut_math_t)x_max;
    ssm_lut_math_t scaled = (ssm_lut_math_t)x * inv_step;
    int idx = (int)scaled;
    if (idx > segments - 1) {
        idx = segments - 1;
    }

    ssm_lut_math_t frac = scaled - (ssm_lut_math_t)idx;
    data_t y0 = (data_t)table[idx];
    data_t y1 = (data_t)table[idx + 1];
    return y0 + (data_t)frac * (y1 - y0);
}

// exp: 256 segments over [0, ln2] -> kExpPwlTable[257]
inline data_t ssm_exp(data_t x) {
    const data_t clip_min = (data_t)-14.0f;
    const data_t clip_max = (data_t)0.0f;
    const data_t ln2 = (data_t)0.693147181f;
    const data_t inv_ln2 = (data_t)1.442695041f;

    data_t clipped = x;
    if (clipped < clip_min) {
        clipped = clip_min;
    }
    if (clipped > clip_max) {
        clipped = clip_max;
    }

    int k = ssm_floor_to_int(clipped * inv_ln2);
    data_t reduced = clipped - (data_t)k * ln2;
    data_t result = ssm_lookup_uniform_positive(reduced, ln2, kExpPwlTable);

    const int shift_count = -k;
    for (int i = 0; i < SSM_EXP_MAX_SHIFT; ++i) {
        if (i < shift_count) {
            result *= (data_t)0.5f;
        }
    }

    return result;
}

// tanh: 512 segments over [0, 4] -> kTanhPwlTable[513]
inline data_t ssm_tanh(data_t x) {
    bool negative = (x < (data_t)0.0f);
    data_t mag = x;
    if (negative) {
        mag = (data_t)(-x);
    }
    data_t approx = ssm_lookup_uniform_positive(mag, (data_t)4.0f, kTanhPwlTable);
    if (negative) {
        return (data_t)(-approx);
    }
    return approx;
}

inline data_t ssm_sigmoid(data_t x) {
    // sigmoid(x) = 0.5 * (1 + tanh(x / 2))
    return ((data_t)0.5f * ((data_t)1.0f + ssm_tanh((data_t)0.5f * x)));
}

#elif MAMBA3_USE_CORDIC

// -----------------------------------------------------------------------
// Hyperbolic CORDIC helpers
// -----------------------------------------------------------------------

// Practical 16-step hyperbolic schedule with repeated iterations.
// Shift sequence corresponds to repeated hyperbolic iterations needed
// for convergence.
static const int CORDIC_HYP_ITERS = 24;
static const int CORDIC_HYP_SHIFT[CORDIC_HYP_ITERS] = {
    1, 2, 3, 4, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22
};

static const math_t CORDIC_ATANH[CORDIC_HYP_ITERS] = {
    math_t(0.549306144334054780),
    math_t(0.255412811882995361),
    math_t(0.125657214140453027),
    math_t(0.062581571477003009),
    math_t(0.062581571477003009),
    math_t(0.031260178490666993),
    math_t(0.015626271752052209),
    math_t(0.007812658951540421),
    math_t(0.003906269868396826),
    math_t(0.001953127483532550),
    math_t(0.000976562810441036),
    math_t(0.000488281288805113),
    math_t(0.000244140629850639),
    math_t(0.000122070313106330),
    math_t(0.000122070313106330),
    math_t(0.000061035156325791),
    math_t(0.000030517578134474),
    math_t(0.000015258789063684),
    math_t(0.000007629394531398),
    math_t(0.000003814697265644),
    math_t(0.000001907348632815),
    math_t(0.000000953674316407),
    math_t(0.000000476837158203),
    math_t(0.000000238418579102)
};

// Gain compensation: x_init = 1/K where K = prod(sqrt(1 - 2^{-2s})) ≈ 0.828.
// Using K directly (0.828) causes ~31% systematic error — verified by Python sim.
static const math_t CORDIC_K_INV = math_t(1.2074970677630608);

// Range reduction constants for exp(x)
static const math_t CORDIC_LN2     = math_t(0.6931471805599453);
static const math_t CORDIC_INV_LN2 = math_t(1.4426950408889634);

template <typename T>
inline T ssm_clip(T x, T lo, T hi) {
#pragma HLS INLINE
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

// -----------------------------------------------------------------------
//  CORDIC shift helpers — ap_fixed uses >>, float uses ldexp-style scaling.
// -----------------------------------------------------------------------
inline math_t cordic_rshift_math(math_t v, int sh) {
#pragma HLS INLINE
#if MAMBA3_USE_FIXED_POINT
    // Prevent asymmetric bitshift rounding towards negative infinity for negative numbers.
    return (v >= (math_t)0) ? (math_t)(v >> sh) : (math_t)(-((math_t)(-v) >> sh));
#else
    // For float: v / 2^sh.  Use a loop or shift to avoid heavy fp division.
    for (int i = 0; i < sh; ++i) {
#pragma HLS UNROLL
        v = v * (math_t)0.5;
    }
    return v;
#endif
}

inline acc_t cordic_lshift_acc(acc_t v, int sh) {
#pragma HLS INLINE
#if MAMBA3_USE_FIXED_POINT
    return v << sh;
#else
    for (int i = 0; i < sh; ++i) {
#pragma HLS UNROLL
        v = v * (acc_t)2.0;
    }
    return v;
#endif
}

inline acc_t cordic_rshift_acc(acc_t v, int sh) {
#pragma HLS INLINE
#if MAMBA3_USE_FIXED_POINT
    // Prevent asymmetric bitshift rounding towards negative infinity for negative numbers.
    return (v >= (acc_t)0) ? (acc_t)(v >> sh) : (acc_t)(-((acc_t)(-v) >> sh));
#else
    for (int i = 0; i < sh; ++i) {
#pragma HLS UNROLL
        v = v * (acc_t)0.5;
    }
    return v;
#endif
}

// Computes sinh(z) and cosh(z) using hyperbolic CORDIC rotation mode.
// This should be used on a reduced-range input for best accuracy.
inline void cordic_sinh_cosh(math_t z_in, math_t &s_out, math_t &c_out) {
#pragma HLS INLINE

    math_t x = CORDIC_K_INV;
    math_t y = 0;
    math_t z = z_in;

    for (int k = 0; k < CORDIC_HYP_ITERS; ++k) {
#pragma HLS UNROLL
        int sh = CORDIC_HYP_SHIFT[k];
        math_t x_shift = cordic_rshift_math(x, sh);
        math_t y_shift = cordic_rshift_math(y, sh);

        math_t x_new, y_new;
        if (z >= 0) {
            x_new = x + y_shift;
            y_new = y + x_shift;
            z     = z - CORDIC_ATANH[k];
        } else {
            x_new = x - y_shift;
            y_new = y - x_shift;
            z     = z + CORDIC_ATANH[k];
        }

        x = x_new;
        y = y_new;
    }

    c_out = x;
    s_out = y;
}

// -----------------------------------------------------------------------
// Nonlinear math helpers using fixed-point + CORDIC
// -----------------------------------------------------------------------

// exp(x) via:
//   x = k*ln2 + r
//   exp(x) = 2^k * exp(r)
//   exp(r) = cosh(r) + sinh(r), where r is range-reduced
inline data_t ssm_exp(data_t x_in) {
#pragma HLS INLINE

    // Keep x in a numerically friendly range.
    // exp(8) still fits comfortably in ap_fixed<32,16>.
    math_t x = ssm_clip((math_t)x_in, math_t(-8.0), math_t(8.0));

    math_t scaled = x * CORDIC_INV_LN2;
    int k;
    if (scaled >= 0)
        k = (int)(scaled + math_t(0.5));
    else
        k = (int)(scaled - math_t(0.5));

    // The conv. radius for 24-iter hyperbolic CORDIC is ~1.118.
    // Clamp r before the CORDIC call so it never escapes the valid domain.
    math_t r = ssm_clip((math_t)(x - math_t(k) * CORDIC_LN2), math_t(-1.1), math_t(1.1));

    math_t s, c;
    cordic_sinh_cosh(r, s, c);

    acc_t e = (acc_t)c + (acc_t)s; // exp(r)

    if (k > 0) {
        e = cordic_lshift_acc(e, k);
    } else if (k < 0) {
        e = cordic_rshift_acc(e, -k);
    }

    return (data_t)e;
}

inline data_t ssm_sigmoid(data_t x) {
#pragma HLS INLINE

    if (x >= (data_t)8.0)  return (data_t)1.0;
    if (x <= (data_t)-8.0) return (data_t)0.0;

    acc_t denom = (acc_t)1.0 + (acc_t)ssm_exp((data_t)(-x));
    return (data_t)((acc_t)1.0 / denom);
}

// ssm_silu is defined after #endif — shared across all paths.

// tanh(x) = 2*sigmoid(2x) - 1
// This keeps tanh built from the same CORDIC-backed exp path.
inline data_t ssm_tanh(data_t x) {
#pragma HLS INLINE

    if (x >= (data_t)4.0)  return (data_t)1.0;
    if (x <= (data_t)-4.0) return (data_t)-1.0;

    data_t sig = ssm_sigmoid((data_t)(x * (data_t)2.0));
    return (data_t)(sig * (data_t)2.0 - (data_t)1.0);
}

#else

inline data_t ssm_exp(data_t x) {
#ifdef __SYNTHESIS__
    return hls::exp(x);
#else
    return std::exp(x);
#endif
}

inline data_t ssm_tanh(data_t x) {
#ifdef __SYNTHESIS__
    return hls::tanh(x);
#else
    return std::tanh(x);
#endif
}

inline data_t ssm_sigmoid(data_t x) {
    return (data_t)1.0f / ((data_t)1.0f + ssm_exp(-x));
}

#endif

inline data_t ssm_silu(data_t x) {
    return x * ssm_sigmoid(x);
}

// -----------------------------------------------------------------------
//  Index flattening helpers
//  Row-major order matching the Python tensor layouts.
// -----------------------------------------------------------------------

// state / state_out: (B, H, D, N)  ->  b*H*D*N + h*D*N + d*N + n
inline int IDX_BHDN(int b, int h, int d, int n) {
    return ((b * NHEADS + h) * HDIM + d) * DSTATE + n;
}

// Bstate, B, C: (B, R, H, N)  ->  b*R*H*N + r*H*N + h*N + n
inline int IDX_BRHN(int b, int r, int h, int n) {
    return ((b * MIMO + r) * NHEADS + h) * DSTATE + n;
}

// x, Xstate, z, out: (B, H, D)  ->  b*H*D + h*D + d
inline int IDX_BHD(int b, int h, int d) {
    return (b * NHEADS + h) * HDIM + d;
}

// A, dt, trap: (B, H)  ->  b*H + h
inline int IDX_BH(int b, int h) {
    return b * NHEADS + h;
}

// xproj, zproj, outproj: (R, H, D)  ->  r*H*D + h*D + d
inline int IDX_RHD(int r, int h, int d) {
    return (r * NHEADS + h) * HDIM + d;
}

// out_r (intermediate): (B, R, H, D) -> b*R*H*D + r*H*D + h*D + d
inline int IDX_BRHD(int b, int r, int h, int d) {
    return ((b * MIMO + r) * NHEADS + h) * HDIM + d;
}

#endif
