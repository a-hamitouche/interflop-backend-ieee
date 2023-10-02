project("interflop_ieee")

set (INTERFLOP_IEEE_SRC
    "interflop_ieee.c"
    "common/printf_specifier.c"
)

set (INTERFLOP_VIEEE_SRC
    "x86_64/interflop_vector_ieee.c"
)

add_library(interflop_ieee_base   OBJECT ${INTERFLOP_IEEE_SRC})
target_compile_definitions(interflop_ieee_base PRIVATE  ${CRT_COMPILE_DEFINITIONS}
"SCALAR" "VECT128" "VECT256" "VECT512")
target_compile_options (interflop_ieee_base PRIVATE ${CRT_PREPROCESS_OPTIONS} ${CRT_COMPILE_OPTIONS})

add_library(interflop_ieee_scalar OBJECT ${INTERFLOP_VIEEE_SRC})
target_compile_definitions(interflop_ieee_scalar PRIVATE  ${CRT_COMPILE_DEFINITIONS} "SCALAR")
target_compile_options (interflop_ieee_scalar PRIVATE ${CRT_PREPROCESS_OPTIONS} ${CRT_COMPILE_OPTIONS} "-mno-sse")

add_library(interflop_ieee_sse    OBJECT ${INTERFLOP_VIEEE_SRC})
target_compile_definitions(interflop_ieee_sse PRIVATE  ${CRT_COMPILE_DEFINITIONS} "VECT128")
target_compile_options (interflop_ieee_sse PRIVATE ${CRT_PREPROCESS_OPTIONS} ${CRT_COMPILE_OPTIONS} "-msse2")

add_library(interflop_ieee_avx    OBJECT ${INTERFLOP_VIEEE_SRC})
target_compile_definitions(interflop_ieee_avx PRIVATE  ${CRT_COMPILE_DEFINITIONS} "VECT256")
target_compile_options (interflop_ieee_avx PRIVATE ${CRT_PREPROCESS_OPTIONS} ${CRT_COMPILE_OPTIONS} "-mavx2")

add_library(interflop_ieee_avx512    OBJECT ${INTERFLOP_VIEEE_SRC})
target_compile_definitions(interflop_ieee_avx512 PRIVATE  ${CRT_COMPILE_DEFINITIONS} "VECT512")
target_compile_options (interflop_ieee_avx512 PRIVATE ${CRT_PREPROCESS_OPTIONS} ${CRT_COMPILE_OPTIONS} "-mavx512f")

add_library (interflop_ieee SHARED  $<TARGET_OBJECTS:interflop_ieee_base>
                                    $<TARGET_OBJECTS:interflop_ieee_scalar>
                                    $<TARGET_OBJECTS:interflop_ieee_sse>
                                    $<TARGET_OBJECTS:interflop_ieee_avx>
                                    $<TARGET_OBJECTS:interflop_ieee_avx512>
)
target_link_options (interflop_ieee PRIVATE ${CRT_LINK_OPTIONS})
target_link_libraries (interflop_ieee ${CRT_LINK_LIBRARIES} interflop_stdlib)