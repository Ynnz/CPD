# include <stdint.h>
# include "sigmoid_table_fixed_q78.h"
// Q7.8 输入，Q7.8 输出
int16_t sigmoid_lookup_q7_8(int16_t x_fp) {
    if (x_fp <= -2048) return 0;
    if (x_fp >= 2048) return 256;

    int32_t offset = x_fp - (-2048);
    int32_t index = (offset * 6553600) >> 16;
    return sigmoid_table[index];
}
