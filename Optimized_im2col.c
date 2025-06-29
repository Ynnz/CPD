#include <stdint.h>

void im2col_int16(const int16_t* input,
                  int16_t* output,
                  int channels, int height, int width,
                  int kernel_h, int kernel_w,
                  int pad_h, int pad_w,
                  int stride_h, int stride_w,
                  int dilation_h, int dilation_w,
                  int height_col, int width_col)
{
    int channels_col = channels * kernel_h * kernel_w;

    int c_im = 0;
    int h_offset = 0;
    int w_offset = 0;

    for (int c_col = 0; c_col < channels_col; ++c_col) {
        for (int h_col = 0; h_col < height_col; ++h_col) {
            int h_im = h_col * stride_h - pad_h + h_offset * dilation_h;

            for (int w_col = 0; w_col < width_col; ++w_col) {
                int w_im = w_col * stride_w - pad_w + w_offset * dilation_w;

                int16_t val = 0;
                if (h_im >= 0 && h_im < height && w_im >= 0 && w_im < width) {
                    val = input[(c_im * height + h_im) * width + w_im];
                }

                output[(c_col * height_col + h_col) * width_col + w_col] = val;
            }
        }

        // Manually increment (c_im, h_offset, w_offset)
        ++w_offset;
        if (w_offset == kernel_w) {
            w_offset = 0;
            ++h_offset;
            if (h_offset == kernel_h) {
                h_offset = 0;
                ++c_im;
            }
        }
    }
}
