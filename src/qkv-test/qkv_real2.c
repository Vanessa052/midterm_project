#include "mytypes.h"
#include "gen_data_real.h"

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

static inline uint64_t ld8pi(uint64_t **pp) {
    uint64_t val;
    uint64_t p = (uint64_t)(*pp);
    asm volatile(".insn r 0x0B, 2, 0, %0, %1, x0"
                 : "=r"(val), "+r"(p));
    *pp = (uint64_t *)p;
    return val;
}

static int8_t Yout[N];

__attribute__((noinline))
void qkv_b0(void) {
    for (int n = 0; n < N; n++) {
        int32_t acc = BIAS_FIXED[n];
        for (int k = 0; k < K; k++) acc += X[k] * W[n][k];
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
void qkv_x2(void) {
    const uint64_t *xp = (const uint64_t *)X;
    for (int n = 0; n < N; n++) {
        int32_t acc = BIAS_FIXED[n];
        const uint64_t *wp = (const uint64_t *)W[n];
        for (int i = 0; i < K/8; i++) {
            acc = xqmac8(acc, xp[i], wp[i]);
        }
        uint64_t packed = ((uint64_t)M0[n]) | ((uint64_t)SHIFT_ARR[n] << 32) | ((uint64_t)ZP << 38);
        Yout[n] = (int8_t)requant8(acc, packed);
    }
}

__attribute__((noinline))
void qkv_x2b(void) {
    const uint64_t *xp = (const uint64_t *)X;
    for (int n = 0; n < N; n++) {
        int32_t acc = BIAS_FIXED[n];
        const uint64_t *wp = (const uint64_t *)W[n];
        for (int i = 0; i < K/8; i++) {
            acc = xqmac8(acc, xp[i], wp[i]);
        }
        Yout[n] = (int8_t)requant8(acc, PACKED[n]);
    }
}

/* X2c: X2b + 外層迴圈 unroll x4 (N=576 是 4 的倍數，不需要 tail) */
__attribute__((noinline))
void qkv_x2c(void) {
    const uint64_t *xp = (const uint64_t *)X;
    for (int n = 0; n < N; n += 4) {
        int32_t acc0 = BIAS_FIXED[n];
        int32_t acc1 = BIAS_FIXED[n+1];
        int32_t acc2 = BIAS_FIXED[n+2];
        int32_t acc3 = BIAS_FIXED[n+3];
        const uint64_t *wp0 = (const uint64_t *)W[n];
        const uint64_t *wp1 = (const uint64_t *)W[n+1];
        const uint64_t *wp2 = (const uint64_t *)W[n+2];
        const uint64_t *wp3 = (const uint64_t *)W[n+3];
        for (int i = 0; i < K/8; i++) {
            uint64_t xi = xp[i];
            acc0 = xqmac8(acc0, xi, wp0[i]);
            acc1 = xqmac8(acc1, xi, wp1[i]);
            acc2 = xqmac8(acc2, xi, wp2[i]);
            acc3 = xqmac8(acc3, xi, wp3[i]);
        }
        Yout[n]   = (int8_t)requant8(acc0, PACKED[n]);
        Yout[n+1] = (int8_t)requant8(acc1, PACKED[n+1]);
        Yout[n+2] = (int8_t)requant8(acc2, PACKED[n+2]);
        Yout[n+3] = (int8_t)requant8(acc3, PACKED[n+3]);
    }
}


/* X2d_fix: X2b + unroll x8, 手寫具名變數版本 (避免陣列導致 spill) */
__attribute__((noinline))
void qkv_x2d(void) {
    const uint64_t *xp = (const uint64_t *)X;
    for (int n = 0; n < N; n += 8) {
        int32_t acc0 = BIAS_FIXED[n],   acc1 = BIAS_FIXED[n+1];
        int32_t acc2 = BIAS_FIXED[n+2], acc3 = BIAS_FIXED[n+3];
        int32_t acc4 = BIAS_FIXED[n+4], acc5 = BIAS_FIXED[n+5];
        int32_t acc6 = BIAS_FIXED[n+6], acc7 = BIAS_FIXED[n+7];
        const uint64_t *wp0 = (const uint64_t *)W[n];
        const uint64_t *wp1 = (const uint64_t *)W[n+1];
        const uint64_t *wp2 = (const uint64_t *)W[n+2];
        const uint64_t *wp3 = (const uint64_t *)W[n+3];
        const uint64_t *wp4 = (const uint64_t *)W[n+4];
        const uint64_t *wp5 = (const uint64_t *)W[n+5];
        const uint64_t *wp6 = (const uint64_t *)W[n+6];
        const uint64_t *wp7 = (const uint64_t *)W[n+7];
        for (int i = 0; i < K/8; i++) {
            uint64_t xi = xp[i];
            acc0 = xqmac8(acc0, xi, wp0[i]);
            acc1 = xqmac8(acc1, xi, wp1[i]);
            acc2 = xqmac8(acc2, xi, wp2[i]);
            acc3 = xqmac8(acc3, xi, wp3[i]);
            acc4 = xqmac8(acc4, xi, wp4[i]);
            acc5 = xqmac8(acc5, xi, wp5[i]);
            acc6 = xqmac8(acc6, xi, wp6[i]);
            acc7 = xqmac8(acc7, xi, wp7[i]);
        }
        Yout[n]   = (int8_t)requant8(acc0, PACKED[n]);
        Yout[n+1] = (int8_t)requant8(acc1, PACKED[n+1]);
        Yout[n+2] = (int8_t)requant8(acc2, PACKED[n+2]);
        Yout[n+3] = (int8_t)requant8(acc3, PACKED[n+3]);
        Yout[n+4] = (int8_t)requant8(acc4, PACKED[n+4]);
        Yout[n+5] = (int8_t)requant8(acc5, PACKED[n+5]);
        Yout[n+6] = (int8_t)requant8(acc6, PACKED[n+6]);
        Yout[n+7] = (int8_t)requant8(acc7, PACKED[n+7]);
    }
}

/* X2e: X2d + ld8pi (load+post-increment)，取代陣列索引，減少位址計算指令 */
__attribute__((noinline))
void qkv_x2e(void) {
    for (int n = 0; n < N; n += 8) {
        int32_t acc0 = BIAS_FIXED[n],   acc1 = BIAS_FIXED[n+1];
        int32_t acc2 = BIAS_FIXED[n+2], acc3 = BIAS_FIXED[n+3];
        int32_t acc4 = BIAS_FIXED[n+4], acc5 = BIAS_FIXED[n+5];
        int32_t acc6 = BIAS_FIXED[n+6], acc7 = BIAS_FIXED[n+7];
        uint64_t *xp  = (uint64_t *)X;
        uint64_t *wp0 = (uint64_t *)W[n];
        uint64_t *wp1 = (uint64_t *)W[n+1];
        uint64_t *wp2 = (uint64_t *)W[n+2];
        uint64_t *wp3 = (uint64_t *)W[n+3];
        uint64_t *wp4 = (uint64_t *)W[n+4];
        uint64_t *wp5 = (uint64_t *)W[n+5];
        uint64_t *wp6 = (uint64_t *)W[n+6];
        uint64_t *wp7 = (uint64_t *)W[n+7];
        for (int i = 0; i < K/8; i++) {
            uint64_t xi = ld8pi(&xp);
            acc0 = xqmac8(acc0, xi, ld8pi(&wp0));
            acc1 = xqmac8(acc1, xi, ld8pi(&wp1));
            acc2 = xqmac8(acc2, xi, ld8pi(&wp2));
            acc3 = xqmac8(acc3, xi, ld8pi(&wp3));
            acc4 = xqmac8(acc4, xi, ld8pi(&wp4));
            acc5 = xqmac8(acc5, xi, ld8pi(&wp5));
            acc6 = xqmac8(acc6, xi, ld8pi(&wp6));
            acc7 = xqmac8(acc7, xi, ld8pi(&wp7));
        }
        Yout[n]   = (int8_t)requant8(acc0, PACKED[n]);
        Yout[n+1] = (int8_t)requant8(acc1, PACKED[n+1]);
        Yout[n+2] = (int8_t)requant8(acc2, PACKED[n+2]);
        Yout[n+3] = (int8_t)requant8(acc3, PACKED[n+3]);
        Yout[n+4] = (int8_t)requant8(acc4, PACKED[n+4]);
        Yout[n+5] = (int8_t)requant8(acc5, PACKED[n+5]);
        Yout[n+6] = (int8_t)requant8(acc6, PACKED[n+6]);
        Yout[n+7] = (int8_t)requant8(acc7, PACKED[n+7]);
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
#if defined(USE_X2E)
    qkv_x2e();
#elif defined(USE_X2D)
    qkv_x2d();
#elif defined(USE_X2C)
    qkv_x2c();
#elif defined(USE_X2B)
    qkv_x2b();
#elif defined(USE_X2)
    qkv_x2();
#else
    qkv_b0();
#endif
    uint64_t t1 = rdinstret();
    (void)t0; (void)t1;

    if (!verify()) return 1;
    return 0;
}
