#include "mytypes.h"
#include "softmax_tests.h"

static inline int32_t xlutacc(int32_t acc, uint64_t table_base, uint64_t idx) {
    asm volatile(".insn r 0x0B, 4, 0, %0, %1, %2"
                 : "+r"(acc) : "r"(table_base), "r"(idx));
    return acc;
}

static int32_t OUT[SEQ];

static void softmax_int(const int64_t *scores, int32_t *out) {
    uint64_t table_base = (uint64_t)LUT;

    /* 1. 找最大值 */
    int64_t maxv = scores[0];
    for (int i = 1; i < SEQ; i++) {
        if (scores[i] > maxv) maxv = scores[i];
    }

    /* 2. 累加 exp 總和 */
    int32_t sum = 0;
    for (int i = 0; i < SEQ; i++) {
        int64_t diff = scores[i] - maxv;         /* <= 0 */
        if (diff < -128) diff = -128;             /* clamp，避免超出 int8 範圍 */
        uint64_t idx = (uint8_t)(int8_t)diff;
        sum = xlutacc(sum, table_base, idx);
    }

    /* 3. 每個分數 / 總和 * 127 */
    for (int i = 0; i < SEQ; i++) {
        int64_t diff = scores[i] - maxv;
        if (diff < -128) diff = -128;
        uint64_t idx = (uint8_t)(int8_t)diff;
        int32_t e = xlutacc(0, table_base, idx);  /* acc=0, 等於純查表 */
        int64_t num = (int64_t)e * 127;
        int32_t q = (int32_t)(num / sum);
        if (q > 127) q = 127;
        if (q < 0) q = 0;
        out[i] = q;
    }
}

void main(void) {
    int fail = 0;
    int64_t max_abs_err = 0;

    for (int t = 0; t < NTEST; t++) {
        softmax_int(TEST_SCORES[t], OUT);
        for (int i = 0; i < SEQ; i++) {
            int64_t err = OUT[i] - TEST_OUT[t][i];
            if (err < 0) err = -err;
            if (err > max_abs_err) max_abs_err = err;
            if (err > 1) fail++;   /* 容許 ±1 的 int8 量化誤差 */
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
