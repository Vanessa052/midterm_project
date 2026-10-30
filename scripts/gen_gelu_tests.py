import numpy as np
from scipy.special import erf

np.random.seed(11)
NTEST = 200

S_in = 0.05    # 輸入 (FC1 輸出) 的量化 scale
S_out = 0.05   # 輸出 (GELU 結果) 的量化 scale

def gelu_fp(x):
    return 0.5 * x * (1 + erf(x / np.sqrt(2)))

# LUT[u]：u 當 unsigned byte，重新解讀成 signed int8，算 GELU 後再量化回 int8，存進 int32 格子
LUT = np.zeros(256, dtype=np.int64)
for u in range(256):
    s = u if u < 128 else u - 256
    x_fp = s * S_in
    y_fp = gelu_fp(x_fp)
    y_q = int(round(y_fp / S_out))
    y_q = max(-128, min(127, y_q))
    LUT[u] = y_q

cases = []
for t in range(NTEST):
    x_q8 = np.random.randint(-128, 128)
    idx = x_q8 if x_q8 >= 0 else x_q8 + 256
    expect = LUT[idx]
    cases.append((x_q8, expect))

with open("gelu_tests.h", "w") as f:
    f.write(f"#define NTEST {NTEST}\n\n")
    f.write("static int32_t GELU_LUT[256] = {" + ",".join(str(v) for v in LUT) + "};\n")
    f.write("static int64_t TEST_X[NTEST] = {" + ",".join(str(x) for x,e in cases) + "};\n")
    f.write("static int64_t TEST_EXPECT[NTEST] = {" + ",".join(str(e) for x,e in cases) + "};\n")

print("written, NTEST=%d" % NTEST)
