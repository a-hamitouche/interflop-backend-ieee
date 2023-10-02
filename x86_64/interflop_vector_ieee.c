/*****************************************************************************\
 *                                                                           *\
 *  This file is part of the Verificarlo project,                            *\
 *  under the Apache License v2.0 with LLVM Exceptions.                      *\
 *  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception.                 *\
 *  See https://llvm.org/LICENSE.txt for license information.                *\
 *                                                                           *\
 *                                                                           *\
 *  Copyright (c) 2015                                                       *\
 *     Universite de Versailles St-Quentin-en-Yvelines                       *\
 *     CMLA, Ecole Normale Superieure de Cachan                              *\
 *                                                                           *\
 *  Copyright (c) 2018                                                       *\
 *     Universite de Versailles St-Quentin-en-Yvelines                       *\
 *                                                                           *\
 *  Copyright (c) 2019-2023                                                  *\
 *     Verificarlo Contributors                                              *\
 *                                                                           *\
 ****************************************************************************/

#include <stdio.h>
#include "interflop_vinterface.h"

#if defined(__x86_64__)
#include <immintrin.h>

#if defined(VECT512)
#include "interflop_vector_ieee_avx512.h"
#endif
#if defined(VECT256)
#include "interflop_vector_ieee_avx.h"
#endif
#if defined(VECT128)
#include "interflop_vector_ieee_sse.h"
#endif
#if defined(SCALAR)
#include "interflop_vector_ieee_scalar.h"
#endif

#elif defined(__ARM_ARCH)
#include <arm_neon.h>
#include <arm_sve.h>
#endif

#include "interflop/iostream/logger.h"

static File *logger_stderr;

#define DEBUG_HEADER "Decimal "
#define DEBUG_BINARY_HEADER "Binary "

/* This macro print the debug information for a, b and c */
/* the debug_print function handles automatically the format */
/* (decimal or binary) depending on the context */
#define DEBUG_PRINT(context, typeop, op, a, b, c, d)                           \
  {                                                                            \
    ieee_context_t *ctx = (ieee_context_t *)context;                           \
    bool debug = ctx->debug ? true : false;                                    \
    bool debug_binary = ctx->debug_binary ? true : false;                      \
    bool subnormal_normalized =                                                \
        ctx->print_subnormal_normalized ? true : false;                        \
    if (debug || debug_binary) {                                               \
      bool print_header = ctx->no_backend_name ? false : true;                 \
      char *header = (debug) ? DEBUG_HEADER : DEBUG_BINARY_HEADER;             \
      char *a_float_fmt =                                                      \
          (subnormal_normalized) ? FMT_SUBNORMAL_NORMALIZED(a) : FMT(a);       \
      char *b_float_fmt =                                                      \
          (subnormal_normalized) ? FMT_SUBNORMAL_NORMALIZED(b) : FMT(b);       \
      char *c_float_fmt =                                                      \
          (subnormal_normalized) ? FMT_SUBNORMAL_NORMALIZED(c) : FMT(c);       \
      char *d_float_fmt =                                                      \
          (subnormal_normalized) ? FMT_SUBNORMAL_NORMALIZED(d) : FMT(d);       \
      if (print_header) {                                                      \
        if (ctx->print_new_line)                                               \
          logger_info("%s\n", header);                                         \
        else                                                                   \
          logger_info("%s", header);                                           \
      }                                                                        \
      if (typeop == ARITHMETIC) {                                              \
        debug_print(context, a_float_fmt, "%g %s ", a, op);                    \
        debug_print(context, b_float_fmt, "%g -> ", b);                        \
        debug_print(context, c_float_fmt, "%g\n", c);                          \
      } else if (typeop == COMPARISON) {                                       \
        debug_print(context, a_float_fmt, "%g [%s] ", a, op);                  \
        debug_print(context, b_float_fmt, "%g -> %s\n", b,                     \
                    c ? "true" : "false");                                     \
      } else if (typeop == CAST) {                                             \
        debug_print(context, a_float_fmt, "%g %s -> ", a, op);                 \
        debug_print(context, b_float_fmt, "%g\n", b);                          \
      } else if (typeop == FMA) {                                              \
        debug_print(context, a_float_fmt, "%g * ", a);                         \
        debug_print(context, b_float_fmt, "%g + ", b);                         \
        debug_print(context, c_float_fmt, "%g -> ", c);                        \
        debug_print(context, d_float_fmt, "%g\n", d);                          \
      }                                                                        \
    }                                                                          \
  }

void INTERFLOP_VECTOR_IEEE_API(add_float_1)(float *a, float *b, float *c,
                                          void *context) {
#if defined(__x86_64__)
#if defined (__AVX512F__) || defined (__AVX2__) || defined (__SSE2__)
  __m128 reg_a = _mm_load_ss (a);
  __m128 reg_b = _mm_load_ss (b);
  __m128 reg_c = _mm_add_ss (reg_a, reg_b);

  _mm_store_ss (c, reg_c);
#endif
#elif defined(__ARM_ARCH)
  svfloat32_t reg_a = svld1_f32(a);
  svfloat32_t reg_b = svld1_f32(b);
  svfloat32_t reg_c = *c = svadd_f32(reg_a, reg_b);

  svst1_f32(c, reg_c);
#else
  *c = (*a) + (*b);
#endif
}

void INTERFLOP_VECTOR_IEEE_API(add_float_4)(float *a, float *b, float *c,
                                          void *context) {
#if defined(__x86_64__)
#if defined (__AVX512F__) || defined (__AVX2__) || defined (__SSE2__)
  __m128 reg_a = _mm_loadu_ps (a);
  __m128 reg_b = _mm_loadu_ps (b);
  __m128 reg_c = _mm_add_ps (reg_a, reg_b);

  _mm_storeu_ps (c, reg_c);
#endif
#else
  c[0] = a[0] + b[0];
  c[1] = a[1] + b[1];
  c[2] = a[2] + b[2];
  c[3] = a[3] + b[3];
#endif
}

void INTERFLOP_VECTOR_IEEE_API(add_float_8)(float *a, float *b, float *c,
                                          void *context) {
#if defined(__x86_64__)
#if defined (__AVX512F__) || defined (__AVX2__)
  __m256 reg_a = _mm256_loadu_ps (a);
  __m256 reg_b = _mm256_loadu_ps (b);
  __m256 reg_c = _mm256_add_ps (reg_a, reg_b);

  _mm256_storeu_ps (c, reg_c);
#elif defined(__SSE4_2__)
  for (size_t i = 0; i < 2; i++)
  {
    __m128 reg_a = _mm_loadu_ps (a + i*4);
    __m128 reg_b = _mm_loadu_ps (b + i*4);
    __m128 reg_c = _mm_add_ps (reg_a, reg_b);

    _mm_storeu_ps (c + i*4, reg_c);
  }
  
#endif
#else
  for (size_t i = 0; i < 2; i++)
  {
    c[0 + i*4] = a[0 + i*4] + b[0 + i*4];
    c[1 + i*4] = a[1 + i*4] + b[1 + i*4];
    c[2 + i*4] = a[2 + i*4] + b[2 + i*4];
    c[3 + i*4] = a[3 + i*4] + b[3 + i*4];
  }
#endif
}

void INTERFLOP_VECTOR_IEEE_API(add_float_16)(float *a, float *b, float *c,
                                          void *context) {
#if defined(__x86_64__)
#if defined (__AVX512F__)
  __m512 reg_a = _mm512_loadu_ps (a);
  __m512 reg_b = _mm512_loadu_ps (b);
  __m512 reg_c = _mm512_add_ps (reg_a, reg_b);

  _mm512_storeu_ps (c, reg_c);
#elif defined (__AVX2__)
  for (size_t i = 0; i < 2; i++)
  {
  __m256 reg_a = _mm256_loadu_ps (a+i*8);
  __m256 reg_b = _mm256_loadu_ps (b+i*8);
  __m256 reg_c = _mm256_add_ps (reg_a, reg_b);

  _mm256_storeu_ps (c+i*8, reg_c);
  }
#elif defined(__SSE4_2__)
  for (size_t i = 0; i < 4; i++)
  {
    __m128 reg_a = _mm_loadu_ps (a + i*4);
    __m128 reg_b = _mm_loadu_ps (b + i*4);
    __m128 reg_c = _mm_add_ps (reg_a, reg_b);

    _mm_storeu_ps (c + i*4, reg_c);
  }
  
#endif
#else
  for (size_t i = 0; i < 4; i++)
  {
    c[0 + i*4] = a[0 + i*4] + b[0 + i*4];
    c[1 + i*4] = a[1 + i*4] + b[1 + i*4];
    c[2 + i*4] = a[2 + i*4] + b[2 + i*4];
    c[3 + i*4] = a[3 + i*4] + b[3 + i*4];
  }
#endif
}

void INTERFLOP_VECTOR_IEEE_API(sub_float_1)(float *a, float *b, float *c,
                                          void *context) {
#if defined (__AVX512F__) || defined (__AVX2__) || defined (__SSE2__)
  __m128 reg_a = _mm_load_ss (a);
  __m128 reg_b = _mm_load_ss (b);
  __m128 reg_c = _mm_sub_ss (reg_a, reg_b);

  _mm_store_ss (c, reg_c);
#else
  *c = (*a) - (*b);
#endif
}

void INTERFLOP_VECTOR_IEEE_API(sub_float_4)(float *a, float *b, float *c,
                                          void *context) {
#if defined(__x86_64__)
#if defined (__AVX512F__) || defined (__AVX2__) || defined (__SSE2__)
  __m128 reg_a = _mm_loadu_ps (a);
  __m128 reg_b = _mm_loadu_ps (b);
  __m128 reg_c = _mm_sub_ps (reg_a, reg_b);

  _mm_storeu_ps (c, reg_c);
#endif
#else
  c[0] = a[0] - b[0];
  c[1] = a[1] - b[1];
  c[2] = a[2] - b[2];
  c[3] = a[3] - b[3];
#endif
}

void INTERFLOP_VECTOR_IEEE_API(sub_float_8)(float *a, float *b, float *c,
                                          void *context) {
#if defined(__x86_64__)
#if defined (__AVX512F__) || defined (__AVX2__)
  __m256 reg_a = _mm256_loadu_ps (a);
  __m256 reg_b = _mm256_loadu_ps (b);
  __m256 reg_c = _mm256_sub_ps (reg_a, reg_b);

  _mm256_storeu_ps (c, reg_c);
#elif defined(__SSE4_2__)
  for (size_t i = 0; i < 2; i++)
  {
    __m128 reg_a = _mm_loadu_ps (a + i*4);
    __m128 reg_b = _mm_loadu_ps (b + i*4);
    __m128 reg_c = _mm_sub_ps (reg_a, reg_b);

    _mm_storeu_ps (c + i*4, reg_c);
  }
  
#endif
#else
  for (size_t i = 0; i < 2; i++)
  {
    c[0 + i*4] = a[0 + i*4] - b[0 + i*4];
    c[1 + i*4] = a[1 + i*4] - b[1 + i*4];
    c[2 + i*4] = a[2 + i*4] - b[2 + i*4];
    c[3 + i*4] = a[3 + i*4] - b[3 + i*4];
  }
#endif
}

void INTERFLOP_VECTOR_IEEE_API(sub_float_16)(float *a, float *b, float *c,
                                          void *context) {
#if defined(__x86_64__)
#if defined (__AVX512F__)
  __m512 reg_a = _mm512_loadu_ps (a);
  __m512 reg_b = _mm512_loadu_ps (b);
  __m512 reg_c = _mm512_sub_ps (reg_a, reg_b);

  _mm512_storeu_ps (c, reg_c);
#elif defined (__AVX2__)
  for (size_t i = 0; i < 2; i++)
  {
  __m256 reg_a = _mm256_loadu_ps (a+i*8);
  __m256 reg_b = _mm256_loadu_ps (b+i*8);
  __m256 reg_c = _mm256_sub_ps (reg_a, reg_b);

  _mm256_storeu_ps (c+i*8, reg_c);
  }
#elif defined(__SSE4_2__)
  for (size_t i = 0; i < 4; i++)
  {
    __m128 reg_a = _mm_loadu_ps (a + i*4);
    __m128 reg_b = _mm_loadu_ps (b + i*4);
    __m128 reg_c = _mm_sub_ps (reg_a, reg_b);

    _mm_storeu_ps (c + i*4, reg_c);
  }
  
#endif
#else
  for (size_t i = 0; i < 4; i++)
  {
    c[0 + i*4] = a[0 + i*4] - b[0 + i*4];
    c[1 + i*4] = a[1 + i*4] - b[1 + i*4];
    c[2 + i*4] = a[2 + i*4] - b[2 + i*4];
    c[3 + i*4] = a[3 + i*4] - b[3 + i*4];
  }
#endif
}

void INTERFLOP_VECTOR_IEEE_API(mul_float_1)(float *a, float *b, float *c,
                                          void *context) {
#if defined (__AVX512F__) || defined (__AVX2__) || defined (__SSE2__)
  __m128 reg_a = _mm_load_ss (a);
  __m128 reg_b = _mm_load_ss (b);
  __m128 reg_c = _mm_mul_ss (reg_a, reg_b);

  _mm_store_ss (c, reg_c);
#else
  *c = (*a) * (*b);
#endif
}

void INTERFLOP_VECTOR_IEEE_API(mul_float_4)(float *a, float *b, float *c,
                                          void *context) {
#if defined(__x86_64__)
#if defined (__AVX512F__) || defined (__AVX2__) || defined (__SSE2__)
  __m128 reg_a = _mm_loadu_ps (a);
  __m128 reg_b = _mm_loadu_ps (b);
  __m128 reg_c = _mm_mul_ps (reg_a, reg_b);

  _mm_storeu_ps (c, reg_c);
#endif
#else
  c[0] = a[0] * b[0];
  c[1] = a[1] * b[1];
  c[2] = a[2] * b[2];
  c[3] = a[3] * b[3];
#endif
}

void INTERFLOP_VECTOR_IEEE_API(mul_float_8)(float *a, float *b, float *c,
                                          void *context) {
#if defined(__x86_64__)
#if defined (__AVX512F__) || defined (__AVX2__)
  __m256 reg_a = _mm256_loadu_ps (a);
  __m256 reg_b = _mm256_loadu_ps (b);
  __m256 reg_c = _mm256_mul_ps (reg_a, reg_b);

  _mm256_storeu_ps (c, reg_c);
#elif defined(__SSE4_2__)
  for (size_t i = 0; i < 2; i++)
  {
    __m128 reg_a = _mm_loadu_ps (a + i*4);
    __m128 reg_b = _mm_loadu_ps (b + i*4);
    __m128 reg_c = _mm_mul_ps (reg_a, reg_b);

    _mm_storeu_ps (c + i*4, reg_c);
  }
  
#endif
#else
  for (size_t i = 0; i < 2; i++)
  {
    c[0 + i*4] = a[0 + i*4] * b[0 + i*4];
    c[1 + i*4] = a[1 + i*4] * b[1 + i*4];
    c[2 + i*4] = a[2 + i*4] * b[2 + i*4];
    c[3 + i*4] = a[3 + i*4] * b[3 + i*4];
  }
#endif
}

void INTERFLOP_VECTOR_IEEE_API(mul_float_16)(float *a, float *b, float *c,
                                          void *context) {
#if defined(__x86_64__)
#if defined (__AVX512F__)
  __m512 reg_a = _mm512_loadu_ps (a);
  __m512 reg_b = _mm512_loadu_ps (b);
  __m512 reg_c = _mm512_mul_ps (reg_a, reg_b);

  _mm512_storeu_ps (c, reg_c);
#elif defined (__AVX2__)
  for (size_t i = 0; i < 2; i++)
  {
  __m256 reg_a = _mm256_loadu_ps (a+i*8);
  __m256 reg_b = _mm256_loadu_ps (b+i*8);
  __m256 reg_c = _mm256_mul_ps (reg_a, reg_b);

  _mm256_storeu_ps (c+i*8, reg_c);
  }
#elif defined(__SSE4_2__)
  for (size_t i = 0; i < 4; i++)
  {
    __m128 reg_a = _mm_loadu_ps (a + i*4);
    __m128 reg_b = _mm_loadu_ps (b + i*4);
    __m128 reg_c = _mm_mul_ps (reg_a, reg_b);

    _mm_storeu_ps (c + i*4, reg_c);
  }
  
#endif
#else
  for (size_t i = 0; i < 4; i++)
  {
    c[0 + i*4] = a[0 + i*4] * b[0 + i*4];
    c[1 + i*4] = a[1 + i*4] * b[1 + i*4];
    c[2 + i*4] = a[2 + i*4] * b[2 + i*4];
    c[3 + i*4] = a[3 + i*4] * b[3 + i*4];
  }
#endif
}

void INTERFLOP_VECTOR_IEEE_API(div_float_1)(float *a, float *b, float *c,
                                          void *context) {
#if defined (__AVX512F__) || defined (__AVX2__) || defined (__SSE2__)
  __m128 reg_a = _mm_load_ss (a);
  __m128 reg_b = _mm_load_ss (b);
  __m128 reg_c = _mm_div_ss (reg_a, reg_b);

  _mm_store_ss (c, reg_c);
#else
  *c = (*a) / (*b);
#endif
}

void INTERFLOP_VECTOR_IEEE_API(div_float_4)(float *a, float *b, float *c,
                                          void *context) {
#if defined(__x86_64__)
#if defined (__AVX512F__) || defined (__AVX2__) || defined (__SSE2__)
  __m128 reg_a = _mm_loadu_ps (a);
  __m128 reg_b = _mm_loadu_ps (b);
  __m128 reg_c = _mm_div_ps (reg_a, reg_b);

  _mm_storeu_ps (c, reg_c);
#endif
#else
  c[0] = a[0] / b[0];
  c[1] = a[1] / b[1];
  c[2] = a[2] / b[2];
  c[3] = a[3] / b[3];
#endif
}

void INTERFLOP_VECTOR_IEEE_API(div_float_8)(float *a, float *b, float *c,
                                          void *context) {
#if defined(__x86_64__)
#if defined (__AVX512F__) || defined (__AVX2__)
  __m256 reg_a = _mm256_loadu_ps (a);
  __m256 reg_b = _mm256_loadu_ps (b);
  __m256 reg_c = _mm256_div_ps (reg_a, reg_b);

  _mm256_storeu_ps (c, reg_c);
#elif defined(__SSE4_2__)
  for (size_t i = 0; i < 2; i++)
  {
    __m128 reg_a = _mm_loadu_ps (a + i*4);
    __m128 reg_b = _mm_loadu_ps (b + i*4);
    __m128 reg_c = _mm_div_ps (reg_a, reg_b);

    _mm_storeu_ps (c + i*4, reg_c);
  }
  
#endif
#else
  for (size_t i = 0; i < 2; i++)
  {
    c[0 + i*4] = a[0 + i*4] / b[0 + i*4];
    c[1 + i*4] = a[1 + i*4] / b[1 + i*4];
    c[2 + i*4] = a[2 + i*4] / b[2 + i*4];
    c[3 + i*4] = a[3 + i*4] / b[3 + i*4];
  }
#endif
}

void INTERFLOP_VECTOR_IEEE_API(div_float_16)(float *a, float *b, float *c,
                                          void *context) {
#if defined(__x86_64__)
#if defined (__AVX512F__)
  __m512 reg_a = _mm512_loadu_ps (a);
  __m512 reg_b = _mm512_loadu_ps (b);
  __m512 reg_c = _mm512_div_ps (reg_a, reg_b);

  _mm512_storeu_ps (c, reg_c);
#elif defined (__AVX2__)
  for (size_t i = 0; i < 2; i++)
  {
  __m256 reg_a = _mm256_loadu_ps (a+i*8);
  __m256 reg_b = _mm256_loadu_ps (b+i*8);
  __m256 reg_c = _mm256_div_ps (reg_a, reg_b);

  _mm256_storeu_ps (c+i*8, reg_c);
  }
#elif defined(__SSE4_2__)
  for (size_t i = 0; i < 4; i++)
  {
    __m128 reg_a = _mm_loadu_ps (a + i*4);
    __m128 reg_b = _mm_loadu_ps (b + i*4);
    __m128 reg_c = _mm_div_ps (reg_a, reg_b);

    _mm_storeu_ps (c + i*4, reg_c);
  }
  
#endif
#else
  for (size_t i = 0; i < 4; i++)
  {
    c[0 + i*4] = a[0 + i*4] / b[0 + i*4];
    c[1 + i*4] = a[1 + i*4] / b[1 + i*4];
    c[2 + i*4] = a[2 + i*4] / b[2 + i*4];
    c[3 + i*4] = a[3 + i*4] / b[3 + i*4];
  }
#endif
}
struct interflop_vector_type_t INTERFLOP_VECTOR_IEEE_API(init)(void *context)
{
  struct interflop_vector_type_t vbackend = {
    add : {
      op_vector_float_1 : INTERFLOP_VECTOR_IEEE_API(add_float_1),
      op_vector_float_4 : INTERFLOP_VECTOR_IEEE_API(add_float_4),
      op_vector_float_8 : INTERFLOP_VECTOR_IEEE_API(add_float_8),
      op_vector_float_16 : INTERFLOP_VECTOR_IEEE_API(add_float_16)
    },
    sub : {
      op_vector_float_1 : INTERFLOP_VECTOR_IEEE_API(sub_float_1),
      op_vector_float_4 : INTERFLOP_VECTOR_IEEE_API(sub_float_4),
      op_vector_float_8 : INTERFLOP_VECTOR_IEEE_API(sub_float_8),
      op_vector_float_16 : INTERFLOP_VECTOR_IEEE_API(sub_float_16)
    },
    mul : {
      op_vector_float_1 : INTERFLOP_VECTOR_IEEE_API(mul_float_1),
      op_vector_float_4 : INTERFLOP_VECTOR_IEEE_API(mul_float_4),
      op_vector_float_8 : INTERFLOP_VECTOR_IEEE_API(mul_float_8),
      op_vector_float_16 : INTERFLOP_VECTOR_IEEE_API(mul_float_16)
    },
    div : {
      op_vector_float_1 : INTERFLOP_VECTOR_IEEE_API(div_float_1),
      op_vector_float_4 : INTERFLOP_VECTOR_IEEE_API(div_float_4),
      op_vector_float_8 : INTERFLOP_VECTOR_IEEE_API(div_float_8),
      op_vector_float_16 : INTERFLOP_VECTOR_IEEE_API(div_float_16)
    }
  };
  return vbackend;
}