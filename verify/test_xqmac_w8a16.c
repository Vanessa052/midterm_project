#include "mytypes.h"

static inline int32_t xqmac_w8a16(int32_t acc, uint64_t a, uint64_t w) {
    asm volatile(".insn r 0x0B, 3, 0, %0, %1, %2"
                 : "+r"(acc) : "r"(a), "r"(w));
    return acc;
}

/* 純 C 參考實作，用來算 golden value */
static int32_t golden(int32_t acc, uint64_t a, uint64_t w) {
    for (int i = 0; i < 4; i++) {
        int16_t ai = (int16_t)((a >> (16*i)) & 0xffff);
        int8_t  wi = (int8_t)((w >> (8*i)) & 0xff);
        acc += (int32_t)ai * (int32_t)wi;
    }
    return acc;
}

void main(void) {
    int fail = 0;

    /* directed cases */
    static struct { int32_t acc; uint64_t a; uint64_t w; } cases[] = {
        {0,          0x0000000000000000ULL, 0x0000000000000000ULL},
        {0,          0x0001000200030004ULL, 0x0102030400000000ULL}, /* 低32bit是weight，高32bit不用 */
        {100,        0xffffffffffffffffULL, 0xffffffffULL},          /* 全部是 -1 */
        {-50,        0x7fff7fff7fff7fffULL, 0x7f7f7f7fULL},          /* 最大正值 */
        {0,          0x8000800080008000ULL, 0x80808080ULL},          /* 最小負值 */
    };

    for (int i = 0; i < 5; i++) {
        int32_t expect = golden(cases[i].acc, cases[i].a, cases[i].w);
        int32_t actual = xqmac_w8a16(cases[i].acc, cases[i].a, cases[i].w);
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
