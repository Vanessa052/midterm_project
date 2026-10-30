import random
from safetensors.numpy import load_file
import numpy as np

random.seed(0)
np.random.seed(0)

W_PATH = "../deit_tiny/model.safetensors"
w = load_file(W_PATH)

# qkv.weight: shape (576, 192) = (N, K)
W_fp = w["blocks.0.attn.qkv.weight"].astype(np.float64)   # (N, K)
B_fp = w["blocks.0.attn.qkv.bias"].astype(np.float64)      # (N,)
N, K = W_fp.shape
assert (N, K) == (576, 192)

# --- 權重量化: per-output-channel 對稱量化 ---
w_absmax = np.abs(W_fp).max(axis=1)                 # (N,)
w_absmax = np.maximum(w_absmax, 1e-8)
Sw = w_absmax / 127.0                                # 每個 channel 的 scale
Wq = np.round(W_fp / Sw[:, None]).astype(np.int32)
Wq = np.clip(Wq, -127, 127).astype(np.int8)          # per-channel INT8 權重

# --- activation 量化: 模擬 LayerNorm 輸出 (大致 N(0,1)) ---
# 沒有真正的校正資料集，這裡用統計上合理的假設：LayerNorm 輸出近似標準常態，
# 取 4 個標準差作為動態範圍上限來定 scale。這是一個已知的近似，報告中要說明。
x_fp = np.random.randn(K)
Sx = 4.0 / 127.0
Xq = np.clip(np.round(x_fp / Sx), -127, 127).astype(np.int8)

# --- bias 折算成整數累加域 ---
# bias_fixed[n] = round(B_fp[n] / (Sx * Sw[n]))
bias_fixed = np.round(B_fp / (Sx * Sw)).astype(np.int64)

# --- 輸出 scale: 同樣沒有校正資料，假設輸出動態範圍與輸入相近 ---
Sy = Sx
ZP = 0  # 對稱量化，zero point 設 0

# --- 計算 golden INT32 累加值與 requant 結果 ---
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

def compute_m0_shift(M):
    """把 M 拆成 M0 in [0.5,1) * 2^-shift 的 Q31 定點表示"""
    if M <= 0:
        return 0, 0
    shift = 0
    while M < 0.5:
        M *= 2.0
        shift += 1
    while M >= 1.0:
        M /= 2.0
        shift -= 1
    m0 = int(round(M * (1 << 31)))
    if m0 == (1 << 31):
        m0 //= 2
        shift -= 1
    return m0, shift

M0 = np.zeros(N, dtype=np.int64)
SHIFT = np.zeros(N, dtype=np.int64)
for n in range(N):
    M = (Sx * Sw[n]) / Sy
    m0, shift = compute_m0_shift(M)
    M0[n] = m0
    SHIFT[n] = shift

Y = np.zeros(N, dtype=np.int32)
for n in range(N):
    acc = int(np.dot(Xq.astype(np.int64), Wq[n].astype(np.int64))) + int(bias_fixed[n])
    t = srdhm(acc, int(M0[n]))
    r = rshift_round(t, int(SHIFT[n]))
    qy = r + ZP
    Y[n] = max(-128, min(127, qy))

# --- 輸出成 C header，含 bias_fixed（之前版本沒有） ---
with open("gen_data_real.h", "w") as f:
    f.write("#include <stdint.h>\n\n")
    f.write(f"#define K {K}\n#define N {N}\n\n")
    f.write("static const int8_t X[K] = {" + ",".join(map(str, Xq.tolist())) + "};\n\n")
    f.write("static const int8_t W[N][K] = {\n")
    for n in range(N):
        f.write("{" + ",".join(map(str, Wq[n].tolist())) + "},\n")
    f.write("};\n\n")
    f.write("static const int32_t BIAS_FIXED[N] = {" + ",".join(map(str, bias_fixed.tolist())) + "};\n")
    f.write("static const uint32_t M0[N] = {" + ",".join(map(str, M0.tolist())) + "};\n")
    f.write("static const uint32_t SHIFT_ARR[N] = {" + ",".join(map(str, SHIFT.tolist())) + "};\n")
    f.write(f"static const uint32_t ZP = {ZP};\n\n")
    f.write("static const int8_t Y_EXPECTED[N] = {" + ",".join(map(str, Y.tolist())) + "};\n")

print(f"done. N={N} K={K}")
print("sample weight scale (first 5 channels):", Sw[:5])
print("sample output:", Y[:10])
print("saturated outputs (|y|==128 or 127):", int(np.sum((Y==127)|(Y==-128))), "/", N)

# --- 額外輸出：預先打包好的 requant 參數 ---
PACKED = [(int(M0[n]) & 0xffffffff) | (int(SHIFT[n]) << 32) | ((ZP & 0xff) << 38) for n in range(N)]
with open("gen_data_real.h", "a") as f:
    f.write("static const uint64_t PACKED[N] = {" + ",".join(f"{p}ULL" for p in PACKED) + "};\n")

print("packed array appended")
