import numpy as np

np.random.seed(7)
SEQ = 32
NTEST = 30

Ss = 0.1          # score 量化 scale（這次單獨測試用，之後接進 block 會用實際算出來的值）
EXP_SCALE = 1 << 20

# LUT[u] = round(EXP_SCALE * exp(Ss * (int8)u))，u 當作 unsigned byte, 但解讀成 signed int8
LUT = np.zeros(256, dtype=np.int64)
for u in range(256):
    s = u if u < 128 else u - 256   # 重新解讀成 signed int8
    LUT[u] = round(EXP_SCALE * np.exp(Ss * s)) if s <= 0 else 0  # 正數分支永遠不會被查到

cases = []
for t in range(NTEST):
    scores_q8 = np.random.randint(-128, 128, size=SEQ).astype(np.int64)
    scores_fp = scores_q8 * Ss
    e = np.exp(scores_fp - scores_fp.max())
    sm = e / e.sum()
    out_q8 = np.round(sm * 127).clip(0, 127).astype(np.int64)
    cases.append((scores_q8, out_q8))

with open("softmax_tests.h", "w") as f:
    f.write(f"#define SEQ {SEQ}\n#define NTEST {NTEST}\n\n")
    f.write("static int32_t LUT[256] = {" + ",".join(str(v) for v in LUT) + "};\n")
    f.write(f"static int64_t TEST_SCORES[NTEST][SEQ] = {{\n")
    for s, o in cases:
        f.write("  {" + ",".join(str(v) for v in s) + "},\n")
    f.write("};\n")
    f.write(f"static int64_t TEST_OUT[NTEST][SEQ] = {{\n")
    for s, o in cases:
        f.write("  {" + ",".join(str(v) for v in o) + "},\n")
    f.write("};\n")

print("written, NTEST=%d SEQ=%d" % (NTEST, SEQ))
