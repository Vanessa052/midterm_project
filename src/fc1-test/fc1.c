#include "mytypes.h"
#include "gen_data.h"

static inline uint64_t rdinstret(void) {
    uint64_t x;
    asm volatile("csrr %0, minstret" : "=r"(x));
    return x;
}

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

static int8_t Yout[N];

__attribute__((noinline))
void fc1_b0(void) {
    for (int n = 0; n < N; n++) {
        int32_t acc = 0;
        for (int k = 0; k < K; k++) acc += X[k] * W[n][k];
        /* software requant (no custom instr) */
        int64_t ab = (int64_t)acc * (int64_t)M0[n];
        int64_t nudge = (ab >= 0) ? (1LL<<30) : (1 - (1LL<<30));
        int64_t t = (ab + nudge) / (1LL<<31);
        int64_t mask = (1LL << SHIFT_ARR[n]) - 1;
        int64_t rem = t & mask;
        int64_t thresh = (mask >> 1) + (t < 0 ? 1 : 0);
        int64_t r = (t >> SHIFT_ARR[n]) + (rem > thresh ? 1 : 0);
        int32_t qy = (int32_t)r + ZP;
        if (qy > 127) qy = 127;
        if (qy < -128) qy = -128;
        Yout[n] = (int8_t)qy;
    }
}

__attribute__((noinline))
void fc1_x2(void) {
    const uint64_t *xp = (const uint64_t *)X;
    for (int n = 0; n < N; n++) {
        int32_t acc = 0;
        const uint64_t *wp = (const uint64_t *)W[n];
        for (int i = 0; i < K/8; i++) {
            acc = xqmac8(acc, xp[i], wp[i]);
        }
        uint64_t packed = ((uint64_t)M0[n]) | ((uint64_t)SHIFT_ARR[n] << 32) | ((uint64_t)ZP << 38);
        Yout[n] = (int8_t)requant8(acc, packed);
    }
}

static int verify(void) {
    for (int n = 0; n < N; n++) {
        if (Yout[n] != Y_EXPECTED[n]) return 0;
    }
    return 1;
}

int main(void) {
    uint64_t t0 = rdinstret();
#if defined(USE_X2)
    fc1_x2();
#else
    fc1_b0();
#endif
    uint64_t t1 = rdinstret();

    if (!verify()) return 1;

    /* 把指令數編碼進 exit code 太小，改用記憶體位址讓外部可讀 */
    static volatile uint64_t result_count;
    result_count = t1 - t0;
    (void)result_count;
    return 0;
}
