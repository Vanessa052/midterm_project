#include "mytypes.h"

static inline int32_t requant8(int32_t acc, uint64_t packed) {
    int32_t result;
    asm volatile(".insn r 0x0B, 1, 0, %0, %1, %2"
                 : "=r"(result) : "r"(acc), "r"(packed));
    return result;
}

static uint64_t pack(uint32_t m0, uint32_t shift, uint32_t zp) {
    return ((uint64_t)m0) | ((uint64_t)shift << 32) | ((uint64_t)zp << 38);
}

int main(void) {
    int fail = 0;

    /* Case 1: acc = 0 -> should just be zero point */
    {
        int32_t r = requant8(0, pack(0x40000000u, 1, 5));
        if (r != 5) fail |= (1<<0);
    }

    /* Case 2: saturate to +127 */
    {
        int32_t r = requant8(2000000000, pack(0x7FFFFFFFu, 0, 0));
        if (r != 127) fail |= (1<<1);
    }

    /* Case 3: saturate to -128 */
    {
        int32_t r = requant8(-2000000000, pack(0x7FFFFFFFu, 0, 0));
        if (r != -128) fail |= (1<<2);
    }

    /* Case 4: shift = 0, mask becomes 0, no rounding needed */
    {
        /* M0=0x40000000 (=0.5 in Q31), acc=100 -> SRDHM ~ 50, shift 0 -> 50 */
        int32_t r = requant8(100, pack(0x40000000u, 0, 0));
        if (r != 50) fail |= (1<<3);
    }

    /* Case 5: negative acc, verify sign handling */
    {
        int32_t r = requant8(-100, pack(0x40000000u, 0, 0));
        if (r != -50) fail |= (1<<4);
    }

    /* Case 6: zero point negative (encoded as 8-bit, e.g. -5) */
    {
        uint32_t zp_neg5 = (uint32_t)(-5) & 0xff;  /* 0xFB */
        int32_t r = requant8(0, pack(0x40000000u, 1, zp_neg5));
        if (r != -5) fail |= (1<<5);
    }

    return fail;
}
