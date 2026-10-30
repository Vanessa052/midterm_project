#include "gen_data.h"

static inline int32_t xqmac8(int32_t acc, uint64_t a, uint64_t w) {
    asm volatile(".insn r 0x0B, 0, 0, %0, %1, %2"
                 : "+r"(acc) : "r"(a), "r"(w));
    return acc;
}

static inline int32_t xqmac_w8a16(int32_t acc, uint64_t a, uint64_t w) {
    asm volatile(".insn r 0x0B, 3, 0, %0, %1, %2"
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
void qkv_w8a8(void) {
    for (int n = 0; n < N; n++) {
        int32_t acc = BIAS8[n];
        for (int i = 0; i < K/8; i++) {
            acc = xqmac8(acc, AVEC8[i], WVEC8[n][i]);
        }
        Yout[n] = (int8_t)requant8(acc, PACKED8[n]);
    }
}

__attribute__((noinline))
void qkv_w8a16(void) {
    for (int n = 0; n < N; n++) {
        int32_t acc = BIAS16[n];
        for (int i = 0; i < K/4; i++) {
            acc = xqmac_w8a16(acc, AVEC16[i], WVEC16[n][i]);
        }
        Yout[n] = (int8_t)requant8(acc, PACKED16[n]);
    }
}

static int verify_out(int use_w8a16) {
    int fail = 0;
    for (int n = 0; n < N; n++) {
        int32_t expect = use_w8a16 ? YEXP16[n] : YEXP8[n];
        if ((int32_t)Yout[n] != expect) fail++;
    }
    return fail;
}

int main(void) {
#if defined(USE_W8A16)
    qkv_w8a16();
    int fail = verify_out(1);
#else
    qkv_w8a8();
    int fail = verify_out(0);
#endif
    return fail > 120 ? 120 : fail;
}
