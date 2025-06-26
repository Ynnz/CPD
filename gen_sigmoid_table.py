"""
Generate a Q7.8 fixed-point sigmoid lookup table header (sigmoid_table_q78.h)
*  range :  [-8.0 , +8.0]
*  step  :   0.01              → 1601 entries
*  scale :   256  (Q7.8)       → int16_t values
"""

import numpy as np

# ---------------------- configurable parameters ----------------------
X_MIN   = -8.0               # lower bound of x
X_MAX   =  8.0               # upper bound of x
STEP    =  0.01              # resolution
SCALE   =  256               # Q7.8 => 2^8
OUTFILE = "sigmoid_table_q78.h"
# ---------------------------------------------------------------------

# 1. generate floating-point sigmoid values
x_vals       = np.arange(X_MIN, X_MAX + STEP, STEP)
sigmoid_float = 1 / (1 + np.exp(-x_vals))

# 2. convert to Q7.8 (int16_t)
sigmoid_q78 = np.round(sigmoid_float * SCALE).astype(np.int16)   # ↔  int16_t

# 3. write header file
with open(OUTFILE, "w") as f:
    f.write("#ifndef SIGMOID_TABLE_Q78_H\n#define SIGMOID_TABLE_Q78_H\n\n")
    f.write("// Q7.8 fixed-point sigmoid lookup table\n")
    f.write(f"#define SIGMOID_MIN   ({X_MIN}f)\n")
    f.write(f"#define SIGMOID_MAX   ({X_MAX}f)\n")
    f.write(f"#define SIGMOID_STEP  ({STEP}f)\n")
    f.write(f"#define SIGMOID_SCALE ({SCALE})\n")
    f.write(f"#define SIGMOID_TABLE_SIZE ({len(sigmoid_q78)})\n\n")
    f.write("static const int16_t sigmoid_table[] = {\n")

    # pretty-print: 12 values per line
    for i, val in enumerate(sigmoid_q78):
        f.write(f"  {val},")
        if (i + 1) % 12 == 0:
            f.write("\n")
    # remove trailing comma and finish
    f.seek(f.tell() - 1)     # backspace last comma
    f.write("\n};\n\n#endif // SIGMOID_TABLE_Q78_H\n")

print(f"Header generated → {OUTFILE}  (size = {len(sigmoid_q78)} entries)")
