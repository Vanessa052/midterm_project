#include "mytypes.h"

#define N 2000
static uint64_t data[N];

/* 簡單 xorshift64，純粹用來產生可重現的隨機資料，不需要外部 python golden */
static uint64_t rng_state = 0x243F6A8885A308D3ULL;
static uint64_t xorshift64(void) {
    uint64_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    rng_state = x;
    return x;
}

static inline uint64_t ld8pi(uint64_t **pp) {
    uint64_t val;
    uint64_t p = (uint64_t)(*pp);
    asm volatile(".insn r 0x0B, 2, 0, %0, %1, x0"
                 : "=r"(val), "+r"(p));
    *pp = (uint64_t *)p;
    return val;
}

void main(void) {
    int fail = 0;

    for (int i = 0; i < N; i++) {
        data[i] = xorshift64();
    }

    uint64_t *p = data;
    for (int i = 0; i < N; i++) {
        uint64_t *before = p;
        uint64_t val = ld8pi(&p);
        if (val != data[i]) fail++;
        if (p != before + 1) fail++;
    }
    if (p != data + N) fail++;

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
