#include <stdint.h>
#include <string.h>  // for memcpy

/*
 * Split an int tensor into two equal chunks along the channel dimension.
 * 
 * Input:
 *   x : [C][H][W] (flattened as 1D array)
 * Output:
 *   a : [C/2][H][W]  → first half of channels
 *   b : [C/2][H][W]  → second half of channels
 *
 * Requirements:
 *   - Channel count C must be even
 *   - No batch dimension is handled
 */
void chunk_channel_int(
    const int* x,   // input tensor [C][H][W]
    int* a,         // output chunk A
    int* b,         // output chunk B
    int C, int H, int W)
{
    int C_half = C >> 1;         // C / 2
    int HW     = H * W;          // elements per channel

    for (int c = 0; c < C; ++c) {
        const int* src = x + c * HW;  // pointer to source channel

        int* dst = (c < C_half)
                 ? (a + c * HW)              // first half → a
                 : (b + (c - C_half) * HW);  // second half → b

        memcpy(dst, src, (size_t)HW * sizeof(int));
    }
}
