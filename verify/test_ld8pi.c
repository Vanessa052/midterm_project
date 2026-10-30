#include "mytypes.h"

static inline uint64_t ld8pi(uint64_t **pp) {
    uint64_t val;
    uint64_t p = (uint64_t)(*pp);
    asm volatile(".insn r 0x0B, 2, 0, %0, %1, x0"
                 : "=r"(val), "+r"(p));
    *pp = (uint64_t *)p;
    return val;
}

static uint64_t data[5] = {
    0x1122334455667788ULL,
    0x0000000000000000ULL,
    0xffffffffffffffffULL,
    0xdeadbeefcafebabeULL,
    0x0102030405060708ULL,
};

void main(void) {
    uint64_t *p = data;
    int fail = 0;

    for (int i = 0; i < 5; i++) {
        uint64_t *before = p;
        uint64_t val = ld8pi(&p);

        if (val != data[i]) fail++;
        if (p != before + 1) fail++;   /* 必須正好前進 8 bytes (1 個 uint64_t) */
    }

    /* 額外測一次：確認指標真的停在陣列尾端之後，沒有多走或少走 */
    if (p != data + 5) fail++;

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
