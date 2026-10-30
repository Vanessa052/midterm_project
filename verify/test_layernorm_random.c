#include "mytypes.h"
#include "layernorm_random_tests.h"

/* 標準逐位元整數開根號，保證正確，不會像牛頓法一樣在某些範圍發散 */
static uint64_t isqrt_u64(uint64_t n) {
    uint64_t res = 0;
    uint64_t bit = 1ULL << 62;
    while (bit > n) bit >>= 2;
    while (bit != 0) {
        if (n >= res + bit) {
            n -= res + bit;
            res += bit << 1;
        }
        res >>= 1;
        bit >>= 2;
    }
    return res;
}

/* 輸入 x_q16 (Q16.16 代表 variance)，輸出 1/sqrt(x) 的 Q16.16 表示 */
static int64_t inv_sqrt_q16(int64_t x_q16) {
    if (x_q16 <= 0) return 0;
    uint64_t s = isqrt_u64((uint64_t)x_q16 << 16);   /* s = sqrt(x) 的 Q16.16 */
    if (s == 0) return 0x7fffffff;                    /* x 太小，避免除0 */
    uint64_t half = s >> 1;
    return (int64_t)(((1ULL << 32) + half) / s);       /* 1/s 的 Q16.16，四捨五入 */
}

static void layernorm_int(const int64_t *x_q16, int64_t *out_q16,
                           const int64_t *gamma_q16, const int64_t *beta_q16) {
    int64_t sum = 0, sumsq = 0;
    for (int i = 0; i < D; i++) sum += x_q16[i];
    int64_t mean = (sum >= 0) ? (sum + D/2)/D : (sum - D/2)/D;
    for (int i = 0; i < D; i++) {
        int64_t d = x_q16[i] - mean;
        sumsq += (d * d) >> 16;
    }
    int64_t var = (sumsq >= 0) ? (sumsq + D/2)/D : (sumsq - D/2)/D;
    int64_t invstd = inv_sqrt_q16(var + 1);
    for (int i = 0; i < D; i++) {
        int64_t d = x_q16[i] - mean;
        int64_t normed = (d * invstd) >> 16;
        int64_t scaled = (normed * gamma_q16[i]) >> 16;
        out_q16[i] = scaled + beta_q16[i];
    }
}

static int64_t OUT[D];

void main(void) {
    int fail = 0;
    int64_t max_abs_err = 0;
    int64_t TOL = (int64_t)(0.02 * 65536.0);

    for (int t = 0; t < NTEST; t++) {
        layernorm_int(TEST_X[t], OUT, TEST_GAMMA[t], TEST_BETA[t]);
        for (int i = 0; i < D; i++) {
            int64_t err = OUT[i] - TEST_Y[t][i];
            if (err < 0) err = -err;
            if (err > max_abs_err) max_abs_err = err;
            if (err > TOL) fail++;
        }
    }

    asm volatile(
        "mv a0, %0\n"
        "slli a0, a0, 1\n"
        "ori a0, a0, 1\n"
        "la a1, tohost\n"
        "sd a0, 0(a1)\n"
        "1: j 1b\n"
        :: "r"(fail > 200 ? 200 : fail)
    );
}
