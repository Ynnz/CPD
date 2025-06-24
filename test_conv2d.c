#include <stdio.h>

int main() {
    // Input: shape [1, 1, 3, 3] → [C_in=1][H=3][W=3]
    int input[3][3] = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };

    // Weight: shape [1, 1, 2, 2] → [C_out=1][C_in=1][kH=2][kW=2]
    int weight[2][2] = {
        {1, 0},
        {0, -1}
    };

    // Bias for the output channel
    int bias = 0;

    // Output shape: [1, 1, 2, 2] → [C_out=1][H_out=2][W_out=2]
    int output[2][2] = {0};

    // Parameters
    int H = 3, W = 3;
    int kH = 2, kW = 2;
    int stride = 1;
    int pad = 0;

    // Perform convolution
    for (int i = 0; i <= H - kH; i += stride) {
        for (int j = 0; j <= W - kW; j += stride) {
            int sum = 0;
            for (int ki = 0; ki < kH; ++ki) {
                for (int kj = 0; kj < kW; ++kj) {
                    int in_val = input[i + ki][j + kj];
                    int w_val  = weight[ki][kj];
                    sum += in_val * w_val;
                }
            }
            int out_i = i / stride;
            int out_j = j / stride;
            output[out_i][out_j] = sum + bias;
        }
    }

    // Print output
    printf("Convolution result (int):\n");
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            printf("%4d ", output[i][j]);
        }
        printf("\n");
    }

    return 0;
}
/*
input = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

weight = [
    [1, 0],
    [0, -1]
]

result: 
-4  -4
-4  -4
*/