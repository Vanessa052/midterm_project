#include "block_data.h"

/* 純軟體版 requant（跟 requant8.h 語義完全一致，但用 C 寫） */
static int32_t sw_requant8(int32_t acc, uint64_t packed) {
    int32_t m0 = (int32_t)(packed & 0xffffffffULL);
    int32_t shift = (int32_t)((packed >> 32) & 0x3f);
    int32_t zp = (int32_t)((packed >> 38) & 0xff);
    if (zp >= 128) zp -= 256;
    int64_t ab = (int64_t)acc * (int64_t)m0;
    int64_t nudge = (ab >= 0) ? (1LL<<30) : (1 - (1LL<<30));
    int64_t t = (ab + nudge) / (1LL<<31);
    int32_t qy;
    if (shift > 0) {
        int64_t mask = (1LL<<shift)-1;
        int64_t rem = t & mask;
        int64_t thr = (mask>>1) + (t<0?1:0);
        int64_t r = (t>>shift) + (rem>thr?1:0);
        qy = (int32_t)r + zp;
    } else {
        qy = (int32_t)t + zp;
    }
    if (qy>127) qy=127;
    if (qy<-128) qy=-128;
    return qy;
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

static void layernorm_b0(const int64_t *x_int, int64_t s_in_q16,
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
        out_int[i] = sw_requant8(acc32, packed_out);
    }
}

static void gemm_b0(const int64_t *x_int, int K, int N,
                     const int8_t *Wflat, const int32_t *bias, const uint64_t *packed,
                     int64_t *out_int) {
    for (int n = 0; n < N; n++) {
        int32_t acc = bias[n];
        const int8_t *wp = &Wflat[(uint64_t)n * K];
        for (int k = 0; k < K; k++) {
            acc += (int32_t)(int8_t)x_int[k] * (int32_t)wp[k];
        }
        out_int[n] = sw_requant8(acc, packed[n]);
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

static int cmp2d(const int64_t *a, const int64_t *b, int n) {
    int fail = 0;
    for (int i = 0; i < n; i++) if (a[i] != b[i]) fail++;
    return fail;
}

int main(void) {
    int code = 0;

    for (int t = 0; t < SEQ; t++)
        layernorm_b0(X_INT[t], SX_Q16, GAMMA1_Q16, BETA1_Q16, PACKED_LN1OUT, h1_int[t]);
    if (cmp2d((int64_t*)h1_int,(int64_t*)GOLD_H1,SEQ*D)) { code=1; goto done; }

    for (int t = 0; t < SEQ; t++)
        gemm_b0(h1_int[t], D, 576, (const int8_t*)WQKV_U8, BIAS_QKV, PACKED_QKV, qkv_int[t]);
    if (cmp2d((int64_t*)qkv_int,(int64_t*)GOLD_QKV,SEQ*576)) { code=2; goto done; }

    for (int h = 0; h < NHEAD; h++) {
        for (int i = 0; i < SEQ; i++) {
            for (int j = 0; j < SEQ; j++) {
                int32_t acc = 0;
                for (int d = 0; d < HEAD_DIM; d++) {
                    int32_t qv = (int32_t)(int8_t)qkv_int[i][h*HEAD_DIM+d];
                    int32_t kv = (int32_t)(int8_t)qkv_int[j][192+h*HEAD_DIM+d];
                    acc += qv*kv;
                }
                scores_int[h][i][j] = sw_requant8(acc, PACKED_SCORE);
            }
        }
    }
    if (cmp2d((int64_t*)scores_int,(int64_t*)GOLD_SCORES,NHEAD*SEQ*SEQ)) { code=3; goto done; }

    for (int h = 0; h < NHEAD; h++) {
        for (int i = 0; i < SEQ; i++) {
            int64_t maxv = scores_int[h][i][0];
            for (int j = 1; j < SEQ; j++) if (scores_int[h][i][j] > maxv) maxv = scores_int[h][i][j];
            int32_t sum = 0;
            int32_t evals[SEQ];
            for (int j = 0; j < SEQ; j++) {
                int64_t diff = scores_int[h][i][j] - maxv;
                if (diff < -128) diff = -128;
                uint64_t idx = (uint8_t)(int8_t)diff;
                int32_t *addr = EXP_LUT + idx;   /* 位址計算 + load + add 拆成三步 */
                int32_t val = *addr;
                sum += val;
                evals[j] = val;
            }
            for (int j = 0; j < SEQ; j++) {
                int64_t num = (int64_t)evals[j] * 127;
                int32_t q = (int32_t)(num / sum);
                if (q > 127) q = 127;
                if (q < 0) q = 0;
                attn_int[h][i][j] = q;
            }
        }
    }
    if (cmp2d((int64_t*)attn_int,(int64_t*)GOLD_ATTN,NHEAD*SEQ*SEQ)) { code=4; goto done; }

    for (int h = 0; h < NHEAD; h++) {
        for (int i = 0; i < SEQ; i++) {
            for (int d = 0; d < HEAD_DIM; d++) {
                int32_t acc = 0;
                for (int j = 0; j < SEQ; j++) {
                    int32_t av = (int32_t)(int8_t)attn_int[h][i][j];
                    int32_t vv = (int32_t)(int8_t)qkv_int[j][384+h*HEAD_DIM+d];
                    acc += av*vv;
                }
                ctx_int[i][h*HEAD_DIM+d] = sw_requant8(acc, PACKED_CTX);
            }
        }
    }
    if (cmp2d((int64_t*)ctx_int,(int64_t*)GOLD_CTX,SEQ*D)) { code=5; goto done; }

    for (int t = 0; t < SEQ; t++)
        gemm_b0(ctx_int[t], D, D, (const int8_t*)WPROJ_U8, BIAS_PROJ, PACKED_PROJ, proj_int[t]);
    if (cmp2d((int64_t*)proj_int,(int64_t*)GOLD_PROJ,SEQ*D)) { code=6; goto done; }

    for (int t = 0; t < SEQ; t++)
        for (int i = 0; i < D; i++) {
            int64_t v = X_INT[t][i] + proj_int[t][i];
            if (v>127) v=127; if (v<-128) v=-128;
            x2_int[t][i] = v;
        }
    if (cmp2d((int64_t*)x2_int,(int64_t*)GOLD_X2,SEQ*D)) { code=7; goto done; }

    for (int t = 0; t < SEQ; t++)
        layernorm_b0(x2_int[t], SX_Q16, GAMMA2_Q16, BETA2_Q16, PACKED_LN2OUT, h2_int[t]);
    if (cmp2d((int64_t*)h2_int,(int64_t*)GOLD_H2,SEQ*D)) { code=8; goto done; }

    for (int t = 0; t < SEQ; t++)
        gemm_b0(h2_int[t], D, 768, (const int8_t*)WFC1_U8, BIAS_FC1, PACKED_FC1, fc1_int[t]);
    if (cmp2d((int64_t*)fc1_int,(int64_t*)GOLD_FC1,SEQ*768)) { code=9; goto done; }

    for (int t = 0; t < SEQ; t++)
        for (int i = 0; i < 768; i++) {
            uint64_t idx = (uint8_t)(int8_t)fc1_int[t][i];
            int32_t *addr = GELU_LUT + idx;
            gelu_int[t][i] = *addr;
        }
    if (cmp2d((int64_t*)gelu_int,(int64_t*)GOLD_GELU,SEQ*768)) { code=10; goto done; }

    for (int t = 0; t < SEQ; t++)
        gemm_b0(gelu_int[t], 768, D, (const int8_t*)WFC2_U8, BIAS_FC2, PACKED_FC2, fc2_int[t]);
    if (cmp2d((int64_t*)fc2_int,(int64_t*)GOLD_FC2,SEQ*D)) { code=11; goto done; }

    for (int t = 0; t < SEQ; t++)
        for (int i = 0; i < D; i++) {
            int64_t v = x2_int[t][i] + fc2_int[t][i];
            if (v>127) v=127; if (v<-128) v=-128;
            y_int[t][i] = v;
        }
    if (cmp2d((int64_t*)y_int,(int64_t*)GOLD_Y,SEQ*D)) { code=12; goto done; }

done:
    return code;
}
