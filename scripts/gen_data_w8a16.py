import os
from safetensors.numpy import load_file
import numpy as np

np.random.seed(7)

W_PATH = os.environ.get("DEIT_WEIGHTS", "../deit_tiny/model.safetensors")
w = load_file(W_PATH)
W_fp = w["blocks.0.attn.qkv.weight"].astype(np.float64)   # (576,192)
B_fp = w["blocks.0.attn.qkv.bias"].astype(np.float64)
K = W_fp.shape[1]   # 192
N = W_fp.shape[0]   # 576
assert K % 8 == 0 and K % 4 == 0

# ---- outlier activation（跟 accuracy 實驗的情境3一致） ----
X_fp = np.random.randn(K) * 1.0
outlier_idx = np.random.choice(K, size=int(K*0.03), replace=False)
X_fp[outlier_idx] *= 20.0

Y_fp = W_fp @ X_fp + B_fp

# ---- 權重量化（兩種方案共用，INT8 per-channel symmetric） ----
Sw = np.abs(W_fp).max(axis=1) / 127.0
Sw[Sw == 0] = 1e-8
W_q8 = np.round(W_fp / Sw[:, None]).clip(-127, 127).astype(np.int64)

# ---- activation 量化：兩種方案 ----
true_max = np.abs(X_fp).max()
Sx8  = true_max / 127.0
Sx16 = true_max / 32767.0
X_q8  = np.round(X_fp / Sx8).clip(-127, 127).astype(np.int64)
X_q16 = np.round(X_fp / Sx16).clip(-32767, 32767).astype(np.int64)

# ---- 輸出 scale：校準到真實動態範圍，避免飽和干擾 ----
Sy = np.abs(Y_fp).max() / 127.0
ZP = 0

def compute_m0_shift(M):
    assert 0 < M < 1, f"M={M} 超出預期範圍，目前的硬體設計假設 M<1（只能右移）"
    shift = 0
    m = M
    while m < 0.5 and shift < 31:
        m *= 2
        shift += 1
    M0 = int(round(m * (1 << 31)))
    if M0 >= (1 << 31):
        M0 = (1 << 31) - 1
    return M0, shift

def c_div_trunc(a, b):
    """模擬 C 整數除法：向零捨去，不是 Python `//` 的 floor"""
    q = abs(a) // abs(b)
    return -q if (a < 0) != (b < 0) else q

def requant_ref(acc, M, zp):
    M0, shift = compute_m0_shift(M)
    ab = acc * M0
    nudge = (1 << 30) if ab >= 0 else (1 - (1 << 30))
    t = c_div_trunc(ab + nudge, 1 << 31)
    if shift > 0:
        mask = (1 << shift) - 1
        remainder = t & mask
        threshold = (mask >> 1) + (1 if t < 0 else 0)
        r = (t >> shift) + (1 if remainder > threshold else 0)
    else:
        r = t
    qy = r + zp
    return max(-128, min(127, qy)), M0, shift

def pack_params(M0, shift, zp):
    zp_u8 = zp & 0xff
    return (M0 & 0xffffffff) | (shift << 32) | (zp_u8 << 38)

def gen_scheme(X_q, Sx, pack_width):
    """pack_width: 8 (xqmac8用, 8個/word) 或 4 (xqmac_w8a16用, 4個/word, 塞低32bits)"""
    bias_fixed = np.round(B_fp / (Sx * Sw)).astype(np.int64)
    acc = W_q8 @ X_q + bias_fixed
    M = Sx * Sw / Sy
    Y_expected = np.zeros(N, dtype=np.int64)
    packed = np.zeros(N, dtype=np.uint64)
    for n in range(N):
        qy, M0, shift = requant_ref(int(acc[n]), float(M[n]), ZP)
        Y_expected[n] = qy
        packed[n] = pack_params(M0, shift, ZP)

    # 打包 activation
    nwords_a = K // pack_width
    avec = np.zeros(nwords_a, dtype=np.uint64)
    for i in range(nwords_a):
        word = 0
        bits = 16 if pack_width == 4 else 8
        maskv = (1 << bits) - 1
        for j in range(pack_width):
            v = int(X_q[i*pack_width + j]) & maskv
            word |= (v << (bits*j))
        avec[i] = word

    # 打包 weight（每個 output channel 一條 row）
    nwords_w = K // pack_width
    wvec = np.zeros((N, nwords_w), dtype=np.uint64)
    for n in range(N):
        for i in range(nwords_w):
            word = 0
            for j in range(pack_width):
                v = int(W_q8[n, i*pack_width + j]) & 0xff
                word |= (v << (8*j))
            wvec[n, i] = word

    return avec, wvec, bias_fixed, packed, Y_expected

avec8,  wvec8,  bias8,  packed8,  Yexp8  = gen_scheme(X_q8,  Sx8,  8)
avec16, wvec16, bias16, packed16, Yexp16 = gen_scheme(X_q16, Sx16, 4)

def c_array(name, typ, arr, hexfmt=False):
    vals = ", ".join((f"0x{v:x}ULL" if hexfmt else str(int(v))) for v in arr.flatten())
    return f"static {typ} {name}[{arr.size}] = {{{vals}}};\n"

with open("gen_data.h", "w") as f:
    f.write("#include \"mytypes.h\"\n\n")
    f.write(f"#define K {K}\n#define N {N}\n\n")
    f.write(c_array("AVEC8",  "uint64_t", avec8,  True))
    f.write(c_array("AVEC16", "uint64_t", avec16, True))
    f.write(f"static uint64_t WVEC8[{N}][{K//8}] = {{\n")
    for n in range(N):
        f.write("  {" + ", ".join(f"0x{v:x}ULL" for v in wvec8[n]) + "},\n")
    f.write("};\n\n")
    f.write(f"static uint64_t WVEC16[{N}][{K//4}] = {{\n")
    for n in range(N):
        f.write("  {" + ", ".join(f"0x{v:x}ULL" for v in wvec16[n]) + "},\n")
    f.write("};\n\n")
    f.write(c_array("BIAS8",  "int32_t", bias8))
    f.write(c_array("BIAS16", "int32_t", bias16))
    f.write(c_array("PACKED8",  "uint64_t", packed8, True))
    f.write(c_array("PACKED16", "uint64_t", packed16, True))
    f.write(c_array("YEXP8",  "int32_t", Yexp8))
    f.write(c_array("YEXP16", "int32_t", Yexp16))

print("gen_data.h written")
print(f"K={K} N={N}")
print(f"Sx8={Sx8:.6g} Sx16={Sx16:.6g} Sy={Sy:.6g}")
