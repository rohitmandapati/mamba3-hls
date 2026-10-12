//===========================================================================
// typedefs.h
//===========================================================================
// @brief: Common datatype/config definitions for the Mamba-3 HLS flow.
//         The external stream stays fp32, while the internal compute type
//         can be switched between float and ap_fixed for design exploration.

#ifndef TYPEDEFS_H
#define TYPEDEFS_H

#include <cstdint>

#ifndef MAMBA3_FORCE_XILINX_HEADERS
#define MAMBA3_FORCE_XILINX_HEADERS 0
#endif

#if MAMBA3_FORCE_XILINX_HEADERS
#include <ap_int.h>
#define MAMBA3_HAVE_AP_INT 1
#if MAMBA3_USE_FIXED_POINT
#include <ap_fixed.h>
#define MAMBA3_HAVE_AP_FIXED 1
#endif
#elif defined(__has_include)
#if __has_include(<ap_int.h>)
#include <ap_int.h>
#define MAMBA3_HAVE_AP_INT 1
#endif
#if __has_include(<ap_fixed.h>)
#include <ap_fixed.h>
#define MAMBA3_HAVE_AP_FIXED 1
#endif
#endif

#ifndef MAMBA3_HAVE_AP_INT
#define MAMBA3_HAVE_AP_INT 0
#endif

#ifndef MAMBA3_HAVE_AP_FIXED
#define MAMBA3_HAVE_AP_FIXED 0
#endif

#ifndef MAMBA3_USE_FIXED_POINT
#define MAMBA3_USE_FIXED_POINT 0
#endif

#ifndef MAMBA3_USE_CORDIC
#define MAMBA3_USE_CORDIC 0
#endif

#ifndef MAMBA3_FIXED_TOTAL_BITS
#define MAMBA3_FIXED_TOTAL_BITS 24
#endif

#ifndef MAMBA3_FIXED_INT_BITS
#define MAMBA3_FIXED_INT_BITS 8
#endif

#ifndef MAMBA3_USE_PWL_NONLINEARITIES
#define MAMBA3_USE_PWL_NONLINEARITIES MAMBA3_USE_FIXED_POINT
#endif

#if MAMBA3_USE_FIXED_POINT
#if !MAMBA3_HAVE_AP_FIXED
#error "Fixed-point mode requires ap_fixed.h to be available."
#endif
typedef ap_fixed<MAMBA3_FIXED_TOTAL_BITS, MAMBA3_FIXED_INT_BITS, AP_RND_CONV, AP_SAT> data_t;
#else
typedef float data_t;
#endif

// -----------------------------------------------------------------------
//  CORDIC internal types (wider precision for intermediate computation)
// -----------------------------------------------------------------------
#if MAMBA3_USE_CORDIC
  #if MAMBA3_USE_FIXED_POINT
    typedef ap_fixed<32, 16, AP_RND_CONV, AP_SAT> math_t;
    typedef ap_fixed<48, 24, AP_RND_CONV, AP_SAT> acc_t;
  #else
    typedef float  math_t;
    typedef double acc_t;
  #endif
#endif

#if MAMBA3_HAVE_AP_INT
typedef ap_uint<32> bit32_t;
#else
typedef uint32_t bit32_t;
#endif

// Union for reinterpreting fp32 payloads over the 32-bit stream interface.
union float_uint32 {
    float f;
    uint32_t u;
};

#endif
