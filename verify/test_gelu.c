#include "mytypes.h"
#include "gelu_tests.h"

static inline int32_t xlutacc(int32_t acc, uint64_t table_base, uint64_t idx) {
    asm volatile(".insn r 0x0B, 4, 0, %0, %1, %2"
                 : "+r"(acc) : "r"(table_base), "r"(idx));
    return acc;
}

void main(void) {
    int fail = 0;
    uint64_t table_base = (uint64_t)GELU_LUT;

    for (int t = 0; t < NTEST; t++) {
        uint64_t idx = (uint8_t)(int8_t)TEST_X[t];
        int32_t result = xlutacc(0, table_base, idx);   /* acc=0，等於純查表 */
        if (result != TEST_EXPECT[t]) fail++;
    }

    asm volatile(
        "mv a0, %0\n"
        "slli a0, a0, 1\n"
        "ori a0, a0, 1\n"
        "la a1, tohost\n"
        "sd a0, 0(a1)\n"
        "1: j 1b\n"
        :: "r"(fail > 200 ? 200 : fail)
    );
}
