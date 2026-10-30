#include "mytypes.h"
#include "random_tests.h"

static inline int32_t xqmac8(int32_t acc, uint64_t a, uint64_t w) {
    asm volatile(".insn r 0x0B, 0, 0, %0, %1, %2"
                 : "+r"(acc) : "r"(a), "r"(w));
    return acc;
}

static inline int32_t requant8(int32_t acc, uint64_t packed) {
    int32_t result;
    asm volatile(".insn r 0x0B, 1, 0, %0, %1, %2"
                 : "=r"(result) : "r"(acc), "r"(packed));
    return result;
}

static uint64_t pack8(const int8_t v[8]) {
    uint64_t r = 0;
    for (int i = 0; i < 8; i++) r |= ((uint64_t)(uint8_t)v[i]) << (8*i);
    return r;
}

static uint64_t pack_requant(uint32_t m0, uint32_t shift, int32_t zp) {
    uint32_t zp8 = (uint32_t)zp & 0xff;
    return ((uint64_t)m0) | ((uint64_t)shift << 32) | ((uint64_t)zp8 << 38);
}

int main(void) {
    int xqmac_fail = 0;
    int requant_fail = 0;

    for (int i = 0; i < N_XQMAC; i++) {
        int32_t r = xqmac8(xq_acc0[i], pack8(xq_a[i]), pack8(xq_w[i]));
        if (r != xq_expected[i]) xqmac_fail++;
    }

    for (int i = 0; i < N_REQUANT; i++) {
        uint64_t packed = pack_requant((uint32_t)rq_m0[i], rq_shift[i], rq_zp[i]);
        int32_t r = requant8(rq_acc[i], packed);
        if ((int8_t)r != rq_expected[i]) requant_fail++;
    }

    /* 用 exit code 回報失敗數，上限 127（exit code 範圍限制），超過就封頂 */
    int total_fail = xqmac_fail + requant_fail;
    if (total_fail > 127) total_fail = 127;
    return total_fail;
}
