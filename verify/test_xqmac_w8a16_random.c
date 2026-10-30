#include "mytypes.h"

static uint64_t rng_state = 0x9E3779B97F4A7C15ULL;
static uint64_t xorshift64(void) {
    uint64_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    rng_state = x;
    return x;
}

static inline int32_t xqmac_w8a16(int32_t acc, uint64_t a, uint64_t w) {
    asm volatile(".insn r 0x0B, 3, 0, %0, %1, %2"
                 : "+r"(acc) : "r"(a), "r"(w));
    return acc;
}

static int32_t golden(int32_t acc, uint64_t a, uint64_t w) {
    for (int i = 0; i < 4; i++) {
        int16_t ai = (int16_t)((a >> (16*i)) & 0xffff);
        int8_t  wi = (int8_t)((w >> (8*i)) & 0xff);
        acc += (int32_t)ai * (int32_t)wi;
    }
    return acc;
}

#define NTEST 2000
static int32_t accs[NTEST];
static uint64_t avec[NTEST];
static uint64_t wvec[NTEST];

void main(void) {
    int fail = 0;

    for (int t = 0; t < NTEST; t++) {
        accs[t] = (int32_t)(xorshift64() & 0xffffff);   /* 避免太大，單純想測乘加邏輯 */
        avec[t] = xorshift64();                          /* 4 個 int16，任意位元組合 */
        wvec[t] = xorshift64() & 0xffffffffULL;           /* 低 32 bits 是 4 個 int8 */
    }

    for (int t = 0; t < NTEST; t++) {
        int32_t expect = golden(accs[t], avec[t], wvec[t]);
        int32_t actual = xqmac_w8a16(accs[t], avec[t], wvec[t]);
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
