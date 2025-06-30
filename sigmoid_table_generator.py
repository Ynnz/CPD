import numpy as np

# Parameters
x_min = -8.0
x_max = 8.0
delta = 0.01
scale = 1 << 12  # Q3.12 fixed-point scale = 4096

# Create x values and sigmoid values
x_vals = np.arange(x_min, x_max + delta, delta)
sigmoid_vals = 1 / (1 + np.exp(-x_vals))

# Convert to Q3.12 fixed-point
sigmoid_q312 = np.round(sigmoid_vals * scale).astype(np.int16)

# Compute reciprocal of delta for fixed-point (Q16.16)
reciprocal_step_q16 = int(round((1 / delta) * (1 << 16)))  # 1/delta in Q16.16

# Generate C header file content
lines = []
lines.append("#ifndef SIGMOID_TABLE_FIXED_H")
lines.append("#define SIGMOID_TABLE_FIXED_H\n")
lines.append("#include <stdint.h>")
lines.append("// Q3.12 sigmoid table using accurate fixed-point indexing\n")
lines.append(f"#define SIGMOID_MIN_FP ({int(x_min * scale)})  // Q3.12")
lines.append(f"#define SIGMOID_MAX_FP ({int(x_max * scale)})  // Q3.12")
lines.append(f"#define SIGMOID_TABLE_SIZE ({len(sigmoid_q312)})")
lines.append(f"#define SIGMOID_RECIP_STEP_Q16 ({reciprocal_step_q16})  // 1/step in Q16.16\n")

lines.append("static const int16_t sigmoid_table[SIGMOID_TABLE_SIZE] = {")
for i, val in enumerate(sigmoid_q312):
    if i % 12 == 0:
        lines.append("  ")
    lines[-1] += f"{val}, "
lines[-1] = lines[-1].rstrip(", ")
lines.append("};\n")

# Add C lookup function using Q3.12 with linear interpolation
lines.append("""
static inline int16_t sigmoid_lookup_q3_12(int16_t x_fp) {
    if (x_fp <= SIGMOID_MIN_FP) return 0;
    if (x_fp >= SIGMOID_MAX_FP) return 4096;  // 1.0 in Q3.12

    int32_t offset = (int32_t)(x_fp - SIGMOID_MIN_FP);
    int32_t index_fp = (offset * SIGMOID_RECIP_STEP_Q16);  // Q3.12 * Q16.16 = Q19.28
    int32_t index = index_fp >> 16;        // Integer part
    int32_t frac = index_fp & 0xFFFF;      // Fractional part

    int16_t y0 = sigmoid_table[index];
    int16_t y1 = sigmoid_table[index + 1];

    int32_t diff = (int32_t)(y1 - y0);
    int32_t interp = y0 + ((diff * frac) >> 16);  // Linear interpolation

    return (int16_t)interp;
}
""")

lines.append("#endif // SIGMOID_TABLE_FIXED_H")

# Write to file
with open("sigmoid_table_fixed_q3_12.h", "w") as f:
    f.write("\n".join(lines))
