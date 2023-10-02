#ifndef __INTERFLOP_VECTOR_IEEE_AVX512_H__
#define __INTERFLOP_VECTOR_IEEE_AVX512_H__

#define INTERFLOP_VECTOR_IEEE_API(name) interflop_vector_ieee_##name##_##avx512

void INTERFLOP_VECTOR_IEEE_API(add_float_1)(float *a, float *b, float *c,
                                          void *context);
void INTERFLOP_VECTOR_IEEE_API(add_float_4)(float *a, float *b, float *c,
                                          void *context);
void INTERFLOP_VECTOR_IEEE_API(add_float_8)(float *a, float *b, float *c,
                                          void *context);
void INTERFLOP_VECTOR_IEEE_API(add_float_16)(float *a, float *b, float *c,
                                          void *context);


void INTERFLOP_VECTOR_IEEE_API(sub_float_1)(float *a, float *b, float *c,
                                          void *context);
void INTERFLOP_VECTOR_IEEE_API(sub_float_4)(float *a, float *b, float *c,
                                          void *context);
void INTERFLOP_VECTOR_IEEE_API(sub_float_8)(float *a, float *b, float *c,
                                          void *context);
void INTERFLOP_VECTOR_IEEE_API(sub_float_16)(float *a, float *b, float *c,
                                          void *context);


void INTERFLOP_VECTOR_IEEE_API(mul_float_1)(float *a, float *b, float *c,
                                          void *context);
void INTERFLOP_VECTOR_IEEE_API(mul_float_4)(float *a, float *b, float *c,
                                          void *context);
void INTERFLOP_VECTOR_IEEE_API(mul_float_8)(float *a, float *b, float *c,
                                          void *context);
void INTERFLOP_VECTOR_IEEE_API(mul_float_16)(float *a, float *b, float *c,
                                          void *context);


void INTERFLOP_VECTOR_IEEE_API(div_float_1)(float *a, float *b, float *c,
                                          void *context);
void INTERFLOP_VECTOR_IEEE_API(div_float_4)(float *a, float *b, float *c,
                                          void *context);
void INTERFLOP_VECTOR_IEEE_API(div_float_8)(float *a, float *b, float *c,
                                          void *context);
void INTERFLOP_VECTOR_IEEE_API(div_float_16)(float *a, float *b, float *c,
                                          void *context);
/*
void INTERFLOP_VECTOR_IEEE_API(add_double_1)(double *a, double *b, double *c,
                                          void *context);
void INTERFLOP_VECTOR_IEEE_API(add_double_2)(double *a, double *b, double *c,
                                          void *context);
void INTERFLOP_VECTOR_IEEE_API(add_double_4)(double *a, double *b, double *c,
                                          void *context);
void INTERFLOP_VECTOR_IEEE_API(add_double_8)(double *a, double *b, double *c,
                                          void *context);


void INTERFLOP_VECTOR_IEEE_API(sub_double_1)(double *a, double *b, double *c,
                                          void *context);
void INTERFLOP_VECTOR_IEEE_API(sub_double_2)(double *a, double *b, double *c,
                                          void *context);
void INTERFLOP_VECTOR_IEEE_API(sub_double_4)(double *a, double *b, double *c,
                                          void *context);
void INTERFLOP_VECTOR_IEEE_API(sub_double_8)(double *a, double *b, double *c,
                                          void *context);


void INTERFLOP_VECTOR_IEEE_API(mul_double_1)(double *a, double *b, double *c,
                                          void *context);
void INTERFLOP_VECTOR_IEEE_API(mul_double_2)(double *a, double *b, double *c,
                                          void *context);
void INTERFLOP_VECTOR_IEEE_API(mul_double_4)(double *a, double *b, double *c,
                                          void *context);
void INTERFLOP_VECTOR_IEEE_API(mul_double_8)(double *a, double *b, double *c,
                                          void *context);


void INTERFLOP_VECTOR_IEEE_API(div_double_1)(double *a, double *b, double *c,
                                          void *context);
void INTERFLOP_VECTOR_IEEE_API(div_double_2)(double *a, double *b, double *c,
                                          void *context);
void INTERFLOP_VECTOR_IEEE_API(div_double_4)(double *a, double *b, double *c,
                                          void *context);
void INTERFLOP_VECTOR_IEEE_API(div_double_8)(double *a, double *b, double *c,
                                          void *context);
                                          

//void INTERFLOP_VECTOR_IEEE_API(finalize)(void *context);

char * INTERFLOP_VECTOR_IEEE_API(get_backend_name)(void);
char * INTERFLOP_VECTOR_IEEE_API(get_backend_version)(void);
//void INTERFLOP_VECTOR_IEEE_API(configure)(void *configure, void *context);*/
struct interflop_vector_type_t INTERFLOP_VECTOR_IEEE_API(init)(void *context);
#endif  