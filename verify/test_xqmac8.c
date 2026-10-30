#include "mytypes.h"

static inline int32_t xqmac8(int32_t acc, uint64_t a, uint64_t w) {
    asm volatile(".insn r 0x0B, 0, 0, %0, %1, %2"
                 : "+r"(acc) : "r"(a), "r"(w));
    return acc;
}

/* 把 8 個 int8 打包成 uint64 */
static uint64_t pack8(const int8_t v[8]) {
    uint64_t r = 0;
    for (int i = 0; i < 8; i++) r |= ((uint64_t)(uint8_t)v[i]) << (8*i);
    return r;
}

int main(void) {
    int fail = 0;

    /* Case 1: all zero */
    {
        int8_t a[8] = {0,0,0,0,0,0,0,0};
        int8_t w[8] = {0,0,0,0,0,0,0,0};
        int32_t r = xqmac8(0, pack8(a), pack8(w));
        if (r != 0) fail |= (1<<0);
    }

    /* Case 2: max negative * max negative, no overflow check */
    {
        int8_t a[8] = {-128,-128,-128,-128,-128,-128,-128,-128};
        int8_t w[8] = {-128,-128,-128,-128,-128,-128,-128,-128};
        int32_t r = xqmac8(0, pack8(a), pack8(w));
        /* 8 * (-128)*(-128) = 8*16384 = 131072 */
        if (r != 131072) fail |= (1<<1);
    }

    /* Case 3: positive/negative alternating, should cancel */
    {
        int8_t a[8] = {5,-5,5,-5,5,-5,5,-5};
        int8_t w[8] = {1,1,1,1,1,1,1,1};
        int32_t r = xqmac8(0, pack8(a), pack8(w));
        if (r != 0) fail |= (1<<2);
    }

    /* Case 4: rd carries prior accumulation */
    {
        int8_t a[8] = {1,1,1,1,1,1,1,1};
        int8_t w[8] = {2,2,2,2,2,2,2,2};
        int32_t r = xqmac8(1000, pack8(a), pack8(w));
        /* 1000 + 8*2 = 1016 */
        if (r != 1016) fail |= (1<<3);
    }

    /* Case 5: isolate byte position (only index 3 nonzero) */
    {
        int8_t a[8] = {0,0,0,7,0,0,0,0};
        int8_t w[8] = {0,0,0,9,0,0,0,0};
        int32_t r = xqmac8(0, pack8(a), pack8(w));
        if (r != 63) fail |= (1<<4);
    }

    return fail;
}
