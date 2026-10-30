import numpy as np

np.random.seed(123)
D = 192
NTEST = 50

def to_q16(x):
    return np.round(x * 65536.0).astype(np.int64)

cases = []
for t in range(NTEST):
    x = np.random.randn(D) * (1.0 + t * 0.1)   # 逐漸放大動態範圍，測試牛頓法在不同尺度下的穩定性
    gamma = np.random.uniform(0.5, 1.5, D)
    beta = np.random.uniform(-0.5, 0.5, D)

    mean = x.mean()
    var = x.var()
    normed = (x - mean) / np.sqrt(var + 1e-6)
    y = normed * gamma + beta

    cases.append((to_q16(x), to_q16(gamma), to_q16(beta), to_q16(y)))

with open("layernorm_random_tests.h", "w") as f:
    f.write("#define NTEST %d\n#define D %d\n\n" % (NTEST, D))
    f.write("static int64_t TEST_X[NTEST][D] = {\n")
    for x,g,b,y in cases:
        f.write("  {" + ",".join(str(v) for v in x) + "},\n")
    f.write("};\n")
    f.write("static int64_t TEST_GAMMA[NTEST][D] = {\n")
    for x,g,b,y in cases:
        f.write("  {" + ",".join(str(v) for v in g) + "},\n")
    f.write("};\n")
    f.write("static int64_t TEST_BETA[NTEST][D] = {\n")
    for x,g,b,y in cases:
        f.write("  {" + ",".join(str(v) for v in b) + "},\n")
    f.write("};\n")
    f.write("static int64_t TEST_Y[NTEST][D] = {\n")
    for x,g,b,y in cases:
        f.write("  {" + ",".join(str(v) for v in y) + "},\n")
    f.write("};\n")

print("written, NTEST=%d D=%d" % (NTEST, D))
