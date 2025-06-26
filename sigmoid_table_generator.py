
import numpy as np

# Parameters
x_min = -8.0
x_max = 8.0
delta = 0.01
scale = 256  # Q7.8 fixed-point scale

# Create x values and sigmoid values
x_vals = np.arange(x_min, x_max + delta, delta)
sigmoid_vals = 1 / (1 + np.exp(-x_vals))

# Convert to Q7.8 fixed-point
sigmoid_q78 = np.round(sigmoid_vals * scale).astype(np.int16)

# Compute reciprocal of delta for fixed-point (Q16.16)
reciprocal_step_q16 = int(round((1 / delta) * (1 << 16)))  # 1/delta in Q16.16

# Generate C header file content
lines = []
lines.append("#ifndef SIGMOID_TABLE_FIXED_H")
lines.append("#define SIGMOID_TABLE_FIXED_H\n")
lines.append("#include <stdint.h>")
lines.append("// Q7.8 sigmoid table using accurate fixed-point indexing\n")
lines.append(f"#define SIGMOID_MIN_FP ({int(x_min * scale)})")
lines.append(f"#define SIGMOID_MAX_FP ({int(x_max * scale)})")
lines.append(f"#define SIGMOID_TABLE_SIZE ({len(sigmoid_q78)})")
lines.append(f"#define SIGMOID_RECIP_STEP_Q16 ({reciprocal_step_q16})  // 1/step in Q16.16\n")

lines.append("static const int16_t sigmoid_table[SIGMOID_TABLE_SIZE] = {")
for i, val in enumerate(sigmoid_q78):
    if i % 12 == 0:
        lines.append("  ")
    lines[-1] += f"{val}, "
lines[-1] = lines[-1].rstrip(", ")
lines.append("};\n")

lines.append("""
static inline int16_t sigmoid_lookup_q7_8(int16_t x_fp) {
    if (x_fp <= SIGMOID_MIN_FP) return 0;
    if (x_fp >= SIGMOID_MAX_FP) return 256;

    // index = ((x_fp - min) * (1 / step)) >> 16
    int32_t offset = (int32_t)(x_fp - SIGMOID_MIN_FP);
    int32_t index = (offset * SIGMOID_RECIP_STEP_Q16) >> 16;

    return sigmoid_table[index];
}
""")
lines.append("#endif // SIGMOID_TABLE_FIXED_H")

# Write to file
with open("sigmoid_table_generator.py", "w") as f:
    f.write("\n".join(lines))
