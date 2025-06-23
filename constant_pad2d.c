#include <stdio.h>
#include <stdlib.h>

// padding 方向分别是 left, right, top, bottom
void constant_pad2d(
    const float* input,        // [C][H][W] flatten
    float* output,             // [C][H+top+bottom][W+left+right] flatten
    int C, int H, int W,
    int left, int right,
    int top, int bottom,
    float pad_value
) {
    int H_out = H + top + bottom;
    int W_out = W + left + right;

    for (int c = 0; c < C; ++c) {
        for (int h = 0; h < H_out; ++h) {
            for (int w = 0; w < W_out; ++w) {
                int out_idx = c * H_out * W_out + h * W_out + w;

                int h_in = h - top;
                int w_in = w - left;

                if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                    int in_idx = c * H * W + h_in * W + w_in;
                    output[out_idx] = input[in_idx];
                } else {
                    output[out_idx] = pad_value;
                }
            }
        }
    }
}