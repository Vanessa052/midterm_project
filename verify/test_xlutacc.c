#include "mytypes.h"

static inline int32_t xlutacc(int32_t acc, uint64_t table_base, uint64_t idx) {
    asm volatile(".insn r 0x0B, 4, 0, %0, %1, %2"
                 : "+r"(acc) : "r"(table_base), "r"(idx));
    return acc;
}

static int32_t LUT[256];

void main(void) {
    int fail = 0;

    for (int i = 0; i < 256; i++) {
        LUT[i] = (i * i) - 5000;   /* 任意但可預期的值，含正負數 */
    }

    uint64_t base = (uint64_t)LUT;

    /* 1. 單次查表（acc 從 0 開始，等於純 lookup，對應 GELU 用法） */
    if (xlutacc(0, base, 0)   != LUT[0])   fail++;
    if (xlutacc(0, base, 255) != LUT[255]) fail++;
    if (xlutacc(0, base, 128) != LUT[128]) fail++;

    /* 2. 累加語意（對應 softmax 累加 exp 值） */
    int32_t acc = 0;
    acc = xlutacc(acc, base, 10);
    acc = xlutacc(acc, base, 20);
    acc = xlutacc(acc, base, 30);
    int32_t expect = LUT[10] + LUT[20] + LUT[30];
    if (acc != expect) fail++;

    /* 3. idx 只取低 8 bits，高位應該被忽略 */
    if (xlutacc(0, base, 0x1234500 | 7) != LUT[7]) fail++;

    asm volatile(
        "mv a0, %0\n"
        "slli a0, a0, 1\n"
        "ori a0, a0, 1\n"
        "la a1, tohost\n"
        "sd a0, 0(a1)\n"
        "1: j 1b\n"
        :: "r"(fail)
    );
}
