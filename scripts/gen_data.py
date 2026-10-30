import random

random.seed(42)
K, N = 192, 768

X = [random.randint(-100, 99) for _ in range(K)]
W = [[random.randint(-100, 99) for _ in range(K)] for _ in range(N)]
M0 = [random.randint(0x40000000, 0x7FFFFFFF) for _ in range(N)]
SHIFT = [random.randint(6, 11) for _ in range(N)]
ZP = 3

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

Y = []
for n in range(N):
    acc = sum(X[k] * W[n][k] for k in range(K))
    t = srdhm(acc, M0[n])
    r = rshift_round(t, SHIFT[n])
    qy = r + ZP
    qy = max(-128, min(127, qy))
    Y.append(qy)

with open("gen_data.h", "w") as f:
    f.write("#include <stdint.h>\n\n")
    f.write(f"#define K {K}\n#define N {N}\n\n")
    f.write("static const int8_t X[K] = {" + ",".join(map(str, X)) + "};\n\n")
    f.write("static const int8_t W[N][K] = {\n")
    for n in range(N):
        f.write("{" + ",".join(map(str, W[n])) + "},\n")
    f.write("};\n\n")
    f.write("static const uint32_t M0[N] = {" + ",".join(map(str, M0)) + "};\n")
    f.write("static const uint32_t SHIFT_ARR[N] = {" + ",".join(map(str, SHIFT)) + "};\n")
    f.write(f"static const uint32_t ZP = {ZP};\n\n")
    f.write("static const int8_t Y_EXPECTED[N] = {" + ",".join(map(str, Y)) + "};\n")

print("done. sample:", Y[:10])
