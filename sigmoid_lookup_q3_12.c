#include <stdint.h>
#include "sigmoid_table_fixed_q3_12.h"

// Basic version: Q3.12 fixed-point lookup without interpolation
int16_t sigmoid_lookup_q3_12(int16_t x_fp) {
    if (x_fp <= SIGMOID_MIN_FP) return 0;
    if (x_fp >= SIGMOID_MAX_FP) return 4096;  // 1.0 in Q3.12

    int32_t offset = (int32_t)(x_fp - SIGMOID_MIN_FP);  // Q3.12
    int32_t index = (offset * SIGMOID_RECIP_STEP_Q16) >> 16;  // Q3.12 * Q16.16 >> 16 = int

    return sigmoid_table[index];
}

// Interpolated version: Q3.12 lookup with linear interpolation
int16_t sigmoid_lookup_q3_12_interp(int16_t x_fp) {
    if (x_fp <= SIGMOID_MIN_FP) return 0;
    if (x_fp >= SIGMOID_MAX_FP) return 4096;

    int32_t offset = (int32_t)(x_fp - SIGMOID_MIN_FP);
    int32_t index_fp = (offset * SIGMOID_RECIP_STEP_Q16);  // Q19.28
    int32_t index = index_fp >> 16;                        // Integer part
    int32_t frac = index_fp & 0xFFFF;                      // Fractional part (Q0.16)

    int16_t y0 = sigmoid_table[index];
    int16_t y1 = sigmoid_table[index + 1];
    int32_t diff = (int32_t)(y1 - y0);
    int32_t interp = y0 + ((diff * frac) >> 16);  // Linear interpolation

    return (int16_t)interp;
}
