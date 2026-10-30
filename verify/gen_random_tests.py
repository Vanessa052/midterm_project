import random

random.seed(123)
N_TESTS = 1000

def srdhm(acc, m0):
    ab = acc * m0
    nudge = (1 << 30) if ab >= 0 else (1 - (1 << 30))
    t = ab + nudge
    return t // (1 << 31) if t >= 0 else -((-t) // (1 << 31))

def rshift_round(t, shift):
    mask = (1 << shift) - 1
    remainder = t & mask
    threshold = (mask >> 1) + (1 if t < 0 else 0)
    return (t >> shift) + (1 if remainder > threshold else 0)

xqmac_cases = []
for _ in range(N_TESTS):
    a = [random.randint(-128, 127) for _ in range(8)]
    w = [random.randint(-128, 127) for _ in range(8)]
    acc0 = random.randint(-1000, 1000)
    acc = acc0 + sum(a[i]*w[i] for i in range(8))
    xqmac_cases.append((a, w, acc0, acc))

requant_cases = []
for _ in range(N_TESTS):
    acc = random.randint(-2**20, 2**20)
    m0 = random.randint(0x40000000, 0x7FFFFFFF)
    shift = random.randint(0, 20)
    zp = random.randint(-20, 20)
    t = srdhm(acc, m0)
    r = rshift_round(t, shift)
    qy = max(-128, min(127, r + zp))
    requant_cases.append((acc, m0, shift, zp, qy))

with open("random_tests.h", "w") as f:
    f.write("#include <stdint.h>\n\n")
    f.write(f"#define N_XQMAC {len(xqmac_cases)}\n")
    f.write(f"#define N_REQUANT {len(requant_cases)}\n\n")

    f.write("static const int8_t xq_a[N_XQMAC][8] = {\n")
    for a,w,acc0,acc in xqmac_cases:
        f.write("{" + ",".join(map(str,a)) + "},\n")
    f.write("};\n")
    f.write("static const int8_t xq_w[N_XQMAC][8] = {\n")
    for a,w,acc0,acc in xqmac_cases:
        f.write("{" + ",".join(map(str,w)) + "},\n")
    f.write("};\n")
    f.write("static const int32_t xq_acc0[N_XQMAC] = {" + ",".join(str(c[2]) for c in xqmac_cases) + "};\n")
    f.write("static const int32_t xq_expected[N_XQMAC] = {" + ",".join(str(c[3]) for c in xqmac_cases) + "};\n\n")

    f.write("static const int32_t rq_acc[N_REQUANT] = {" + ",".join(str(c[0]) for c in requant_cases) + "};\n")
    f.write("static const uint32_t rq_m0[N_REQUANT] = {" + ",".join(str(c[1]) for c in requant_cases) + "};\n")
    f.write("static const uint32_t rq_shift[N_REQUANT] = {" + ",".join(str(c[2]) for c in requant_cases) + "};\n")
    f.write("static const int32_t rq_zp[N_REQUANT] = {" + ",".join(str(c[3]) for c in requant_cases) + "};\n")
    f.write("static const int8_t rq_expected[N_REQUANT] = {" + ",".join(str(c[4]) for c in requant_cases) + "};\n")

print(f"generated {len(xqmac_cases)} xqmac8 cases, {len(requant_cases)} requant8 cases")
