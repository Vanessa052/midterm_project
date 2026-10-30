#include "mytypes.h"

static uint64_t rng_state = 0xA5A5A5A5A5A5A5A5ULL;
static uint64_t xorshift64(void) {
    uint64_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    rng_state = x;
    return x;
}

static inline int32_t xlutacc(int32_t acc, uint64_t table_base, uint64_t idx) {
    asm volatile(".insn r 0x0B, 4, 0, %0, %1, %2"
                 : "+r"(acc) : "r"(table_base), "r"(idx));
    return acc;
}

static int32_t LUT[256];

#define NTEST 2000
static int32_t accs[NTEST];
static uint64_t idxs[NTEST];

void main(void) {
    int fail = 0;

    for (int i = 0; i < 256; i++) {
        LUT[i] = (int32_t)(xorshift64() & 0xffffffff);
    }
    uint64_t base = (uint64_t)LUT;

    for (int t = 0; t < NTEST; t++) {
        accs[t] = (int32_t)(xorshift64() & 0xffffff);
        idxs[t] = xorshift64();   /* 故意不先 mask，測試指令自己只取低 8 bits */
    }

    for (int t = 0; t < NTEST; t++) {
        int32_t expect = accs[t] + LUT[idxs[t] & 0xff];
        int32_t actual = xlutacc(accs[t], base, idxs[t]);
        if (expect != actual) fail++;
    }

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
