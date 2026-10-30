#include "block_data.h"

static inline int32_t xqmac8(int32_t acc, uint64_t a, uint64_t w) {
    asm volatile(".insn r 0x0B, 0, 0, %0, %1, %2" : "+r"(acc) : "r"(a), "r"(w));
    return acc;
}
static inline int32_t requant8(int32_t acc, uint64_t packed) {
    int32_t result;
    asm volatile(".insn r 0x0B, 1, 0, %0, %1, %2" : "=r"(result) : "r"(acc), "r"(packed));
    return result;
}
static inline uint64_t ld8pi(uint64_t **pp) {
    uint64_t val; uint64_t p = (uint64_t)(*pp);
    asm volatile(".insn r 0x0B, 2, 0, %0, %1, x0" : "=r"(val), "+r"(p));
    *pp = (uint64_t*)p;
    return val;
}
static inline int32_t xlutacc(int32_t acc, uint64_t table_base, uint64_t idx) {
    asm volatile(".insn r 0x0B, 4, 0, %0, %1, %2" : "+r"(acc) : "r"(table_base), "r"(idx));
    return acc;
}

static uint64_t isqrt_u64(uint64_t n) {
    uint64_t res = 0, bit = 1ULL << 62;
    while (bit > n) bit >>= 2;
    while (bit != 0) {
        if (n >= res + bit) { n -= res + bit; res += bit << 1; }
        res >>= 1; bit >>= 2;
    }
    return res;
}
static int64_t inv_sqrt_q16(int64_t x_q16) {
    if (x_q16 <= 0) return 0;
    uint64_t s = isqrt_u64((uint64_t)x_q16 << 16);
    if (s == 0) return 0x7fffffff;
    return (int64_t)(((1ULL << 32) + (s >> 1)) / s);
}

static void layernorm_opt(const int64_t *x_int, int64_t s_in_q16,
                           const int64_t *gamma_q16, const int64_t *beta_q16,
                           uint64_t packed_out, int64_t *out_int) {
    int64_t x_q16[D];
    int64_t sum = 0;
    for (int i = 0; i < D; i++) { x_q16[i] = x_int[i] * s_in_q16; sum += x_q16[i]; }
    int64_t mean = (sum >= 0) ? (sum + D/2)/D : (sum - D/2)/D;
    int64_t sumsq = 0;
    for (int i = 0; i < D; i++) { int64_t d = x_q16[i]-mean; sumsq += (d*d) >> 16; }
    int64_t var = (sumsq >= 0) ? (sumsq + D/2)/D : (sumsq - D/2)/D;
    int64_t invstd = inv_sqrt_q16(var + 1);
    for (int i = 0; i < D; i++) {
        int64_t d = x_q16[i] - mean;
        int64_t normed = (d * invstd) >> 16;
        int64_t scaled = (normed * gamma_q16[i]) >> 16;
        int64_t out_q16 = scaled + beta_q16[i];
        int32_t acc32 = (out_q16 > 2000000000) ? 2000000000 :
                         ((out_q16 < -2000000000) ? -2000000000 : (int32_t)out_q16);
        out_int[i] = requant8(acc32, packed_out);
    }
}

static void gemm_opt(const int64_t *x_int, int K, int N,
                      const uint64_t *Wflat, const int32_t *bias, const uint64_t *packed,
                      int64_t *out_int) {
    int words = K / 8;
    uint64_t xpack[96];
    for (int i = 0; i < words; i++) {
        uint64_t w = 0;
        for (int j = 0; j < 8; j++) w |= ((uint64_t)(uint8_t)x_int[i*8+j]) << (8*j);
        xpack[i] = w;
    }
    for (int n = 0; n < N; n++) {
        int32_t acc = bias[n];
        uint64_t *xp = xpack;
        uint64_t *wp = (uint64_t*)&Wflat[(uint64_t)n * words];
        for (int i = 0; i < words; i++) {
            acc = xqmac8(acc, ld8pi(&xp), ld8pi(&wp));
        }
        out_int[n] = requant8(acc, packed[n]);
    }
}

static int64_t h1_int[SEQ][D];
static int64_t qkv_int[SEQ][576];
static int64_t scores_int[NHEAD][SEQ][SEQ];
static int64_t attn_int[NHEAD][SEQ][SEQ];
static int64_t ctx_int[SEQ][D];
static int64_t proj_int[SEQ][D];
static int64_t x2_int[SEQ][D];
static int64_t h2_int[SEQ][D];
static int64_t fc1_int[SEQ][768];
static int64_t gelu_int[SEQ][768];
static int64_t fc2_int[SEQ][D];
static int64_t y_int[SEQ][D];

static int cmp2d(const int64_t *a, const int64_t *b, int n, const char *name) {
    int fail = 0;
    for (int i = 0; i < n; i++) if (a[i] != b[i]) fail++;
    return fail;
}

int main(void) {
    int fail_h1=0, fail_qkv=0, fail_scores=0, fail_attn=0, fail_ctx=0;
    int fail_proj=0, fail_x2=0, fail_h2=0, fail_fc1=0, fail_gelu=0, fail_fc2=0, fail_y=0;

    /* Stage 1: LayerNorm1 */
    for (int t = 0; t < SEQ; t++)
        layernorm_opt(X_INT[t], SX_Q16, GAMMA1_Q16, BETA1_Q16, PACKED_LN1OUT, h1_int[t]);
    fail_h1 = cmp2d((int64_t*)h1_int, (int64_t*)GOLD_H1, SEQ*D, "h1");

    /* Stage 2: QKV GEMM */
    for (int t = 0; t < SEQ; t++)
        gemm_opt(h1_int[t], D, 576, (const uint64_t*)WQKV, BIAS_QKV, PACKED_QKV, qkv_int[t]);
    fail_qkv = cmp2d((int64_t*)qkv_int, (int64_t*)GOLD_QKV, SEQ*576, "qkv");

    /* Stage 3: Attention */
    for (int h = 0; h < NHEAD; h++) {
        for (int i = 0; i < SEQ; i++) {
            uint64_t qpack[8];
            for (int w = 0; w < 8; w++) {
                uint64_t word = 0;
                for (int b = 0; b < 8; b++) {
                    int d = w*8+b;
                    int64_t qv = qkv_int[i][h*HEAD_DIM + d];
                    word |= ((uint64_t)(uint8_t)qv) << (8*b);
                }
                qpack[w] = word;
            }
            for (int j = 0; j < SEQ; j++) {
                uint64_t kpack[8];
                for (int w = 0; w < 8; w++) {
                    uint64_t word = 0;
                    for (int b = 0; b < 8; b++) {
                        int d = w*8+b;
                        int64_t kv = qkv_int[j][192 + h*HEAD_DIM + d];
                        word |= ((uint64_t)(uint8_t)kv) << (8*b);
                    }
                    kpack[w] = word;
                }
                int32_t acc = 0;
                uint64_t *qp = qpack, *kp = kpack;
                for (int w = 0; w < 8; w++) acc = xqmac8(acc, ld8pi(&qp), ld8pi(&kp));
                scores_int[h][i][j] = requant8(acc, PACKED_SCORE);
            }
        }
    }
    fail_scores = cmp2d((int64_t*)scores_int, (int64_t*)GOLD_SCORES, NHEAD*SEQ*SEQ, "scores");

    /* Stage 4: Softmax (xlutacc) */
    uint64_t exp_base = (uint64_t)EXP_LUT;
    for (int h = 0; h < NHEAD; h++) {
        for (int i = 0; i < SEQ; i++) {
            int64_t maxv = scores_int[h][i][0];
            for (int j = 1; j < SEQ; j++) if (scores_int[h][i][j] > maxv) maxv = scores_int[h][i][j];
            int32_t sum = 0;
            for (int j = 0; j < SEQ; j++) {
                int64_t diff = scores_int[h][i][j] - maxv;
                if (diff < -128) diff = -128;
                uint64_t idx = (uint8_t)(int8_t)diff;
                sum = xlutacc(sum, exp_base, idx);
            }
            for (int j = 0; j < SEQ; j++) {
                int64_t diff = scores_int[h][i][j] - maxv;
                if (diff < -128) diff = -128;
                uint64_t idx = (uint8_t)(int8_t)diff;
                int32_t e = xlutacc(0, exp_base, idx);
                int64_t num = (int64_t)e * 127;
                int32_t q = (int32_t)(num / sum);
                if (q > 127) q = 127;
                if (q < 0) q = 0;
                attn_int[h][i][j] = q;
            }
        }
    }
    fail_attn = cmp2d((int64_t*)attn_int, (int64_t*)GOLD_ATTN, NHEAD*SEQ*SEQ, "attn");

    /* Stage 5: Context = attn @ V */
    for (int h = 0; h < NHEAD; h++) {
        for (int i = 0; i < SEQ; i++) {
            uint64_t apack[4];
            for (int w = 0; w < 4; w++) {
                uint64_t word = 0;
                for (int b = 0; b < 8; b++) {
                    word |= ((uint64_t)(uint8_t)attn_int[h][i][w*8+b]) << (8*b);
                }
                apack[w] = word;
            }
            for (int d = 0; d < HEAD_DIM; d++) {
                uint64_t vpack[4];
                for (int w = 0; w < 4; w++) {
                    uint64_t word = 0;
                    for (int b = 0; b < 8; b++) {
                        int64_t vv = qkv_int[w*8+b][384 + h*HEAD_DIM + d];
                        word |= ((uint64_t)(uint8_t)vv) << (8*b);
                    }
                    vpack[w] = word;
                }
                int32_t acc = 0;
                uint64_t *ap = apack, *vp = vpack;
                for (int w = 0; w < 4; w++) acc = xqmac8(acc, ld8pi(&ap), ld8pi(&vp));
                ctx_int[i][h*HEAD_DIM + d] = requant8(acc, PACKED_CTX);
            }
        }
    }
    fail_ctx = cmp2d((int64_t*)ctx_int, (int64_t*)GOLD_CTX, SEQ*D, "ctx");

    /* Stage 6: Output projection */
    for (int t = 0; t < SEQ; t++)
        gemm_opt(ctx_int[t], D, D, (const uint64_t*)WPROJ, BIAS_PROJ, PACKED_PROJ, proj_int[t]);
    fail_proj = cmp2d((int64_t*)proj_int, (int64_t*)GOLD_PROJ, SEQ*D, "proj");

    /* Stage 7: 殘差1 */
    for (int t = 0; t < SEQ; t++)
        for (int i = 0; i < D; i++) {
            int64_t v = X_INT[t][i] + proj_int[t][i];
            if (v > 127) v = 127; if (v < -128) v = -128;
            x2_int[t][i] = v;
        }
    fail_x2 = cmp2d((int64_t*)x2_int, (int64_t*)GOLD_X2, SEQ*D, "x2");

    /* Stage 8: LayerNorm2 */
    for (int t = 0; t < SEQ; t++)
        layernorm_opt(x2_int[t], SX_Q16, GAMMA2_Q16, BETA2_Q16, PACKED_LN2OUT, h2_int[t]);
    fail_h2 = cmp2d((int64_t*)h2_int, (int64_t*)GOLD_H2, SEQ*D, "h2");

    /* Stage 9: FC1 */
    for (int t = 0; t < SEQ; t++)
        gemm_opt(h2_int[t], D, 768, (const uint64_t*)WFC1, BIAS_FC1, PACKED_FC1, fc1_int[t]);
    fail_fc1 = cmp2d((int64_t*)fc1_int, (int64_t*)GOLD_FC1, SEQ*768, "fc1");

    /* Stage 10: GELU (xlutacc) */
    uint64_t gelu_base = (uint64_t)GELU_LUT;
    for (int t = 0; t < SEQ; t++)
        for (int i = 0; i < 768; i++) {
            uint64_t idx = (uint8_t)(int8_t)fc1_int[t][i];
            gelu_int[t][i] = xlutacc(0, gelu_base, idx);
        }
    fail_gelu = cmp2d((int64_t*)gelu_int, (int64_t*)GOLD_GELU, SEQ*768, "gelu");

    /* Stage 11: FC2 */
    for (int t = 0; t < SEQ; t++)
        gemm_opt(gelu_int[t], 768, D, (const uint64_t*)WFC2, BIAS_FC2, PACKED_FC2, fc2_int[t]);
    fail_fc2 = cmp2d((int64_t*)fc2_int, (int64_t*)GOLD_FC2, SEQ*D, "fc2");

    /* Stage 12: 殘差2 (最終輸出) */
    for (int t = 0; t < SEQ; t++)
        for (int i = 0; i < D; i++) {
            int64_t v = x2_int[t][i] + fc2_int[t][i];
            if (v > 127) v = 127; if (v < -128) v = -128;
            y_int[t][i] = v;
        }
    fail_y = cmp2d((int64_t*)y_int, (int64_t*)GOLD_Y, SEQ*D, "y");

    int total = fail_h1+fail_qkv+fail_scores+fail_attn+fail_ctx+fail_proj+fail_x2+fail_h2+fail_fc1+fail_gelu+fail_fc2+fail_y;

    /* 把每個階段的 fail 編碼進回傳值：哪個非0就是哪個階段先錯 */
    int code = 0;
    if (fail_h1) code = 1;
    else if (fail_qkv) code = 2;
    else if (fail_scores) code = 3;
    else if (fail_attn) code = 4;
    else if (fail_ctx) code = 5;
    else if (fail_proj) code = 6;
    else if (fail_x2) code = 7;
    else if (fail_h2) code = 8;
    else if (fail_fc1) code = 9;
    else if (fail_gelu) code = 10;
    else if (fail_fc2) code = 11;
    else if (fail_y) code = 12;

    return code;
}
