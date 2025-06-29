# include <stdint.h>
# include "sigmoid_table_fixed.h"

// Q7.8 输入，Q7.8 输出
int16_t sigmoid_lookup_q7_8(int16_t x_fp) {
    if (x_fp <= -2048) return 0;
    if (x_fp >= 2048) return 256;

    int32_t offset = x_fp - (-2048);
    int32_t index = (offset * 6553600) >> 16;
    return sigmoid_table[index];
}

static inline int16_t sigmoid_lookup_q7_8(int16_t x_fp) {
    if (x_fp <= SIGMOID_MIN_FP) return 0;
    if (x_fp >= SIGMOID_MAX_FP) return 256;

    // index = ((x_fp - min) * (1 / step)) >> 16
    int32_t offset = (int32_t)(x_fp - SIGMOID_MIN_FP);
    int32_t index = (offset * SIGMOID_RECIP_STEP_Q16) >> 16;

    return sigmoid_table[index];
}

static inline int16_t sigmoid_lookup_q7_8_interp(int16_t x_fp) {
    if (x_fp <= SIGMOID_MIN_FP) return 0;
    if (x_fp >= SIGMOID_MAX_FP) return 256;

    int32_t offset = (int32_t)(x_fp - SIGMOID_MIN_FP);
    int32_t index_fp = (offset * SIGMOID_RECIP_STEP_Q16);  // Q16.16
    int32_t index = index_fp >> 16;        // Integer part
    int32_t frac = index_fp & 0xFFFF;      // Lower 16 bits = fractional part

    int16_t y0 = sigmoid_table[index];
    int16_t y1 = sigmoid_table[index + 1];

    // Linear interpolation: y = y0 + (y1 - y0) * frac / 65536
    int32_t diff = (int32_t)(y1 - y0);
    int32_t interp = y0 + ((diff * frac) >> 16);

    return (int16_t)interp;
}
