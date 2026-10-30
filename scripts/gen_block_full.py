from safetensors.numpy import load_file
import numpy as np
from scipy.special import erf

np.random.seed(42)
D, SEQ, NHEAD = 192, 32, 3
HEAD_DIM = D // NHEAD

w = load_file("../deit_tiny/model.safetensors")
def g(name): return w[name].astype(np.float64)

gamma1, beta1 = g("blocks.0.norm1.weight"), g("blocks.0.norm1.bias")
gamma2, beta2 = g("blocks.0.norm2.weight"), g("blocks.0.norm2.bias")
W_qkv, B_qkv = g("blocks.0.attn.qkv.weight"), g("blocks.0.attn.qkv.bias")
W_proj, B_proj = g("blocks.0.attn.proj.weight"), g("blocks.0.attn.proj.bias")
W_fc1, B_fc1 = g("blocks.0.mlp.fc1.weight"), g("blocks.0.mlp.fc1.bias")
W_fc2, B_fc2 = g("blocks.0.mlp.fc2.weight"), g("blocks.0.mlp.fc2.bias")

X = np.load("X.npy")
Y_fp = np.load("Y_fp.npy")

def layernorm_fp(x, gamma, beta, eps=1e-6):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * gamma + beta
def gelu_fp(x): return 0.5*x*(1+erf(x/np.sqrt(2)))
def softmax_fp(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x); return e/e.sum(axis=axis, keepdims=True)

h1_fp = layernorm_fp(X, gamma1, beta1)
qkv_fp = h1_fp @ W_qkv.T + B_qkv
q_fp,k_fp,v_fp = qkv_fp[:,:192], qkv_fp[:,192:384], qkv_fp[:,384:]
def split_heads(t): return t.reshape(SEQ,NHEAD,HEAD_DIM).transpose(1,0,2)
qh,kh,vh = split_heads(q_fp), split_heads(k_fp), split_heads(v_fp)
scores_fp = np.einsum('hqd,hkd->hqk', qh, kh)/np.sqrt(HEAD_DIM)
attn_fp = softmax_fp(scores_fp, axis=-1)
ctx_fp = np.einsum('hqk,hkd->hqd', attn_fp, vh).transpose(1,0,2).reshape(SEQ,D)
proj_fp = ctx_fp @ W_proj.T + B_proj
x2_fp = X + proj_fp
h2_fp = layernorm_fp(x2_fp, gamma2, beta2)
fc1_fp = h2_fp @ W_fc1.T + B_fc1
gelu_fp_out = gelu_fp(fc1_fp)

true_max = max(np.abs(X).max(), np.abs(Y_fp).max())
Sx = true_max/120.0
Sh1 = np.abs(h1_fp).max()/120.0
Sqkv = np.abs(qkv_fp).max()/120.0
Sscore = np.abs(scores_fp).max()/120.0
Sctx = np.abs(ctx_fp).max()/120.0
Sh2 = np.abs(h2_fp).max()/120.0
Sfc1 = np.abs(fc1_fp).max()/120.0
Sgelu = np.abs(gelu_fp_out).max()/120.0

def c_div_trunc(a,b):
    q = abs(a)//abs(b)
    return -q if (a<0)!=(b<0) else q

def compute_m0_shift(M):
    assert 0 < M < 1, f"M={M} 超出範圍"
    shift=0; m=M
    while m<0.5 and shift<31:
        m*=2; shift+=1
    M0=int(round(m*(1<<31)))
    if M0>=(1<<31): M0=(1<<31)-1
    return M0, shift

def requant_ref(acc, M, zp=0):
    M0, shift = compute_m0_shift(M)
    ab = acc*M0
    nudge = (1<<30) if ab>=0 else (1-(1<<30))
    t = c_div_trunc(ab+nudge, 1<<31)
    if shift>0:
        mask=(1<<shift)-1
        rem=t&mask
        thr=(mask>>1)+(1 if t<0 else 0)
        r=(t>>shift)+(1 if rem>thr else 0)
    else:
        r=t
    qy=r+zp
    return max(-128,min(127,qy)), M0, shift

def pack_params(M0,shift,zp=0):
    return (M0 & 0xffffffff) | (shift<<32) | ((zp&0xff)<<38)

def quant_weight(Wf):
    Sw = np.abs(Wf).max(axis=1)/127.0
    Sw[Sw==0]=1e-8
    Wq = np.round(Wf/Sw[:,None]).clip(-127,127).astype(np.int64)
    return Wq, Sw

def gemm_requant(X_int, Wf, Bf, S_in, S_out):
    """X_int: (..., K) int8 已量化的輸入；Wf,Bf: 真實權重；回傳 requant 後的 int8 輸出 + 所有要給C的資料"""
    Wq, Sw = quant_weight(Wf)
    N, K = Wq.shape
    bias_fixed = np.round(Bf/(S_in*Sw)).astype(np.int64)
    acc = X_int.astype(np.int64) @ Wq.T + bias_fixed
    M = S_in*Sw/S_out
    out = np.zeros(acc.shape, dtype=np.int64)
    packed = np.zeros(N, dtype=np.uint64)
    flat_acc = acc.reshape(-1, N)
    flat_out = np.zeros_like(flat_acc)
    for n in range(N):
        qy, M0, shift = requant_ref(0, M[n])  # 先算packed params(acc=0只是占位)
        packed[n] = pack_params(M0, shift, 0)
    for r in range(flat_acc.shape[0]):
        for n in range(N):
            qy,_,_ = requant_ref(int(flat_acc[r,n]), M[n])
            flat_out[r,n] = qy
    out = flat_out.reshape(acc.shape)
    return out, Wq, bias_fixed, packed

# ---- Stage 1: 量化 X ----
X_int = np.round(X/Sx).clip(-127,127).astype(np.int64)
print(f"X量化誤差 relErr: {np.linalg.norm(X_int*Sx-X)/np.linalg.norm(X):.4g}")

print("\n(這份腳本先跑到這裡，確認 helper function 沒問題；下一步接著做完整 pipeline)")

# ---- 整數 LayerNorm（跟 Stage B C 程式碼完全一致的演算法，確保 golden 跟硬體 bit-exact） ----
def isqrt_u64(n):
    res, bit = 0, 1 << 62
    while bit > n: bit >>= 2
    while bit != 0:
        if n >= res + bit:
            n -= res + bit
            res += bit << 1
        res >>= 1; bit >>= 2
    return res

def inv_sqrt_q16(x_q16):
    if x_q16 <= 0: return 0
    s = isqrt_u64(x_q16 << 16)
    if s == 0: return 0x7fffffff
    return ((1 << 32) + (s >> 1)) // s

def layernorm_int_q16(x_arr, gamma_arr, beta_arr):
    Dn = len(x_arr)
    s = sum(int(v) for v in x_arr)
    mean = c_div_trunc(s + (Dn//2 if s>=0 else -(Dn//2)), Dn)
    sumsq = 0
    for xi in x_arr:
        d = int(xi) - mean
        sumsq += (d*d) >> 16
    var = c_div_trunc(sumsq + (Dn//2 if sumsq>=0 else -(Dn//2)), Dn)
    invstd = inv_sqrt_q16(var + 1)
    out = []
    for i, xi in enumerate(x_arr):
        d = int(xi) - mean
        normed = (d * invstd) >> 16
        scaled = (normed * int(gamma_arr[i])) >> 16
        out.append(scaled + int(beta_arr[i]))
    return out

def layernorm_pipeline(X_int_row, S_in_q16_const, gamma_fp, beta_fp, out_M):
    """X_int_row: int8量化輸入; S_in_q16_const: round(S_in*65536)的整數常數(跟C完全一致);
       out_M: 輸出端 requant 的乘數"""
    SCALE = 65536.0
    x_q16 = [int(v) * S_in_q16_const for v in X_int_row]   # 跟C的 x_int[i]*s_in_q16 完全一致
    gamma_q16 = [int(round(v * SCALE)) for v in gamma_fp]
    beta_q16 = [int(round(v * SCALE)) for v in beta_fp]
    out_q16 = layernorm_int_q16(x_q16, gamma_q16, beta_q16)
    out_int8 = []
    for v in out_q16:
        acc32 = max(-2000000000, min(2000000000, v))
        qy,_,_ = requant_ref(acc32, out_M)
        out_int8.append(qy)
    return np.array(out_int8, dtype=np.int64)

# ---- Stage 2: LayerNorm1 (對每個token跑一次) ----
SX_Q16 = int(round(Sx * 65536.0))
M_ln1_out = 1.0/(65536.0*Sh1)
h1_int = np.array([layernorm_pipeline(X_int[t], SX_Q16, gamma1, beta1, M_ln1_out) for t in range(SEQ)])
h1_dequant = h1_int * Sh1
print(f"LayerNorm1 relErr vs float: {np.linalg.norm(h1_dequant-h1_fp)/np.linalg.norm(h1_fp):.4g}")

# ---- Stage 3: QKV GEMM ----
qkv_int, Wqkv_q, bias_qkv, packed_qkv = gemm_requant(h1_int, W_qkv, B_qkv, Sh1, Sqkv)
qkv_dequant = qkv_int * Sqkv
print(f"QKV relErr vs float: {np.linalg.norm(qkv_dequant-qkv_fp)/np.linalg.norm(qkv_fp):.4g}")

print("\n進度確認完成，接下來補 Attention + Proj + 殘差 + LayerNorm2 + FC1 + GELU + FC2 + 殘差")

# ---- Softmax: 建表 + 整數 pipeline(跟 Stage C 一致) ----
EXP_SCALE = 1 << 20
def build_exp_lut(Ss):
    LUT = np.zeros(256, dtype=np.int64)
    for u in range(256):
        s = u if u < 128 else u - 256
        LUT[u] = round(EXP_SCALE * np.exp(Ss * s)) if s <= 0 else 0
    return LUT

def softmax_int_row(scores_row, LUT):
    maxv = max(scores_row)
    exps, total = [], 0
    for s in scores_row:
        diff = s - maxv
        if diff < -128: diff = -128
        e = int(LUT[diff & 0xff])
        exps.append(e); total += e
    out = []
    for e in exps:
        q = max(0, min(127, (e*127)//total))
        out.append(q)
    return out

EXP_LUT = build_exp_lut(Sscore)

# ---- Stage 4: Attention (每個 head 獨立算) ----
def split_heads_int(t_int):
    return t_int.reshape(SEQ, NHEAD, HEAD_DIM).transpose(1,0,2)

q_int, k_int, v_int = qkv_int[:,:192], qkv_int[:,192:384], qkv_int[:,384:]
qh_i, kh_i, vh_i = split_heads_int(q_int), split_heads_int(k_int), split_heads_int(v_int)

M_score = Sqkv*Sqkv/np.sqrt(HEAD_DIM)/Sscore
scores_int = np.zeros((NHEAD, SEQ, SEQ), dtype=np.int64)
for h in range(NHEAD):
    acc32 = qh_i[h].astype(np.int64) @ kh_i[h].T.astype(np.int64)
    for i in range(SEQ):
        for j in range(SEQ):
            qy,_,_ = requant_ref(int(acc32[i,j]), M_score)
            scores_int[h,i,j] = qy

attn_int = np.zeros((NHEAD, SEQ, SEQ), dtype=np.int64)
for h in range(NHEAD):
    for i in range(SEQ):
        attn_int[h,i,:] = softmax_int_row(list(scores_int[h,i,:]), EXP_LUT)

M_ctx = (1.0/127.0)*Sqkv/Sctx
ctx_int_heads = np.zeros((NHEAD, SEQ, HEAD_DIM), dtype=np.int64)
for h in range(NHEAD):
    acc32 = attn_int[h].astype(np.int64) @ vh_i[h].astype(np.int64)
    for i in range(SEQ):
        for d in range(HEAD_DIM):
            qy,_,_ = requant_ref(int(acc32[i,d]), M_ctx)
            ctx_int_heads[h,i,d] = qy
ctx_int = ctx_int_heads.transpose(1,0,2).reshape(SEQ, D)

scores_dq = scores_int * Sscore
print(f"Attention scores relErr: {np.linalg.norm(scores_dq-scores_fp)/np.linalg.norm(scores_fp):.4g}")
attn_dq = attn_int / 127.0
print(f"Softmax (attn weights) relErr: {np.linalg.norm(attn_dq-attn_fp)/np.linalg.norm(attn_fp):.4g}")
print(f"attn_fp 統計: max={attn_fp.max():.4f} 平均每row最大值={attn_fp.max(axis=-1).mean():.4f} (越接近1代表分布越集中)")
print(f"attn_int 統計: max={attn_int.max()} 有多少是0: {(attn_int==0).sum()}/{attn_int.size}")
ctx_dq = ctx_int * Sctx
print(f"Context relErr: {np.linalg.norm(ctx_dq-ctx_fp)/np.linalg.norm(ctx_fp):.4g}")

# ---- Stage 5: Output projection (直接 requant 到 Sx) ----
proj_int, Wproj_q, bias_proj, packed_proj = gemm_requant(ctx_int, W_proj, B_proj, Sctx, Sx)

# ---- Stage 6: 殘差相加 1 (兩邊都是 Sx domain, 直接整數加飽和) ----
x2_int = np.clip(X_int + proj_int, -128, 127)
x2_dq = x2_int * Sx
print(f"x2 (殘差1後) relErr: {np.linalg.norm(x2_dq-x2_fp)/np.linalg.norm(x2_fp):.4g}")

# ---- Stage 7: LayerNorm2 ----
M_ln2_out = 1.0/(65536.0*Sh2)
h2_int = np.array([layernorm_pipeline(x2_int[t], SX_Q16, gamma2, beta2, M_ln2_out) for t in range(SEQ)])
h2_dq = h2_int * Sh2
print(f"LayerNorm2 relErr: {np.linalg.norm(h2_dq-h2_fp)/np.linalg.norm(h2_fp):.4g}")

# ---- Stage 8: FC1 ----
fc1_int, Wfc1_q, bias_fc1, packed_fc1 = gemm_requant(h2_int, W_fc1, B_fc1, Sh2, Sfc1)
fc1_dq = fc1_int * Sfc1
print(f"FC1 relErr: {np.linalg.norm(fc1_dq-fc1_fp)/np.linalg.norm(fc1_fp):.4g}")

# ---- Stage 9: GELU LUT ----
def build_gelu_lut(S_in, S_out):
    LUT = np.zeros(256, dtype=np.int64)
    for u in range(256):
        s = u if u < 128 else u - 256
        x_fp = s * S_in
        y_fp_ = gelu_fp(x_fp)
        y_q = int(round(y_fp_ / S_out))
        LUT[u] = max(-128, min(127, y_q))
    return LUT

GELU_LUT = build_gelu_lut(Sfc1, Sgelu)
gelu_int = np.zeros_like(fc1_int)
flat_in = fc1_int.flatten()
flat_out = np.array([GELU_LUT[int(v) & 0xff] for v in flat_in])
gelu_int = flat_out.reshape(fc1_int.shape)
gelu_dq = gelu_int * Sgelu
print(f"GELU relErr: {np.linalg.norm(gelu_dq-gelu_fp_out)/np.linalg.norm(gelu_fp_out):.4g}")

# ---- Stage 10: FC2 (直接 requant 到 Sx) ----
fc2_int, Wfc2_q, bias_fc2, packed_fc2 = gemm_requant(gelu_int, W_fc2, B_fc2, Sgelu, Sx)

# ---- Stage 11: 殘差相加 2 (最終輸出) ----
Y_int = np.clip(x2_int + fc2_int, -128, 127)
Y_dq = Y_int * Sx
final_relerr = np.linalg.norm(Y_dq-Y_fp)/np.linalg.norm(Y_fp)
print(f"\n=== 最終輸出 relErr vs float golden: {final_relerr:.4g} ===")
print(f"Y_int 範圍: [{Y_int.min()}, {Y_int.max()}]  (應該在[-128,127]內，不要卡在邊界)")

# ============ 匯出 C header：完整權重 + 所有 requant 參數 + 兩個 LUT + 各階段 golden ============
def pack8(int8_mat):
    """int8_mat: (N,K), K必須是8的倍數. 回傳 (N,K//8) uint64陣列，8個int8打包成1個word"""
    N, K = int8_mat.shape
    out = np.zeros((N, K//8), dtype=np.uint64)
    for n in range(N):
        for i in range(K//8):
            word = 0
            for j in range(8):
                v = int(int8_mat[n, i*8+j]) & 0xff
                word |= (v << (8*j))
            out[n,i] = word
    return out

def pack8_vec(int8_vec):
    K = len(int8_vec)
    out = np.zeros(K//8, dtype=np.uint64)
    for i in range(K//8):
        word = 0
        for j in range(8):
            v = int(int8_vec[i*8+j]) & 0xff
            word |= (v << (8*j))
        out[i] = word
    return out

WQKV_packed = pack8(Wqkv_q)   # (576,24)
WPROJ_packed = pack8(Wproj_q) # (192,24)
WFC1_packed = pack8(Wfc1_q)   # (768,24)
WFC2_packed = pack8(Wfc2_q)   # (192,96)

def c_arr(name, typ, arr, hexfmt=False):
    flat = arr.flatten()
    vals = ",".join((f"0x{int(v)&0xffffffffffffffff:x}ULL" if hexfmt else str(int(v))) for v in flat)
    shape = "][".join(str(s) for s in arr.shape)
    return f"static {typ} {name}[{shape}] = {{{vals}}};\n"

_,M0_score,shift_score = requant_ref(0, M_score)
packed_score_scalar = pack_params(M0_score, shift_score, 0)
_,M0_ctx,shift_ctx = requant_ref(0, M_ctx)
packed_ctx_scalar = pack_params(M0_ctx, shift_ctx, 0)

SCALE16 = 65536.0
gamma1_q16 = np.round(gamma1*SCALE16).astype(np.int64)
beta1_q16  = np.round(beta1*SCALE16).astype(np.int64)
gamma2_q16 = np.round(gamma2*SCALE16).astype(np.int64)
beta2_q16  = np.round(beta2*SCALE16).astype(np.int64)

with open("block_data.h", "w") as f:
    f.write("#include \"mytypes.h\"\n#define D 192\n#define SEQ 32\n#define NHEAD 3\n#define HEAD_DIM 64\n\n")
    f.write(c_arr("X_INT", "int64_t", X_int))
    f.write(c_arr("GAMMA1_Q16", "int64_t", gamma1_q16))
    f.write(c_arr("BETA1_Q16", "int64_t", beta1_q16))
    f.write(c_arr("GAMMA2_Q16", "int64_t", gamma2_q16))
    f.write(c_arr("BETA2_Q16", "int64_t", beta2_q16))
    f.write(c_arr("WQKV", "uint64_t", WQKV_packed, True))
    f.write(c_arr("BIAS_QKV", "int32_t", bias_qkv))
    f.write(c_arr("PACKED_QKV", "uint64_t", packed_qkv, True))
    f.write(c_arr("WPROJ", "uint64_t", WPROJ_packed, True))
    f.write(c_arr("BIAS_PROJ", "int32_t", bias_proj))
    f.write(c_arr("PACKED_PROJ", "uint64_t", packed_proj, True))
    f.write(c_arr("WFC1", "uint64_t", WFC1_packed, True))
    f.write(c_arr("BIAS_FC1", "int32_t", bias_fc1))
    f.write(c_arr("PACKED_FC1", "uint64_t", packed_fc1, True))
    f.write(c_arr("WFC2", "uint64_t", WFC2_packed, True))
    f.write(c_arr("BIAS_FC2", "int32_t", bias_fc2))
    f.write(c_arr("PACKED_FC2", "uint64_t", packed_fc2, True))
    f.write(c_arr("EXP_LUT", "int32_t", EXP_LUT))
    f.write(c_arr("GELU_LUT", "int32_t", GELU_LUT))
    f.write(f"static uint64_t PACKED_SCORE = 0x{packed_score_scalar:x}ULL;\n")
    f.write(f"static uint64_t PACKED_CTX = 0x{packed_ctx_scalar:x}ULL;\n")
    # 各階段 golden（debug用）
    f.write(c_arr("GOLD_H1", "int64_t", h1_int))
    f.write(c_arr("GOLD_QKV", "int64_t", qkv_int))
    f.write(c_arr("GOLD_SCORES", "int64_t", scores_int))
    f.write(c_arr("GOLD_ATTN", "int64_t", attn_int))
    f.write(c_arr("GOLD_CTX", "int64_t", ctx_int))
    f.write(c_arr("GOLD_PROJ", "int64_t", proj_int))
    f.write(c_arr("GOLD_X2", "int64_t", x2_int))
    f.write(c_arr("GOLD_H2", "int64_t", h2_int))
    f.write(c_arr("GOLD_FC1", "int64_t", fc1_int))
    f.write(c_arr("GOLD_GELU", "int64_t", gelu_int))
    f.write(c_arr("GOLD_FC2", "int64_t", fc2_int))
    f.write(c_arr("GOLD_Y", "int64_t", Y_int))

print("\nblock_data.h 匯出完成")
import os
print(f"檔案大小: {os.path.getsize('block_data.h')/1024:.1f} KB")

# ---- LayerNorm 輸入轉Q16.16的常數（純整數乘法）跟輸出量化reuse requant8 ----
SH1_Q16 = int(round(Sh1 * SCALE16))
SH2_Q16 = int(round(Sh2 * SCALE16))

_,M0_ln1,shift_ln1 = requant_ref(0, M_ln1_out)
_,M0_ln2,shift_ln2 = requant_ref(0, M_ln2_out)
packed_ln1_out = pack_params(M0_ln1, shift_ln1, 0)
packed_ln2_out = pack_params(M0_ln2, shift_ln2, 0)

with open("block_data.h", "a") as f:
    f.write(f"static int64_t SX_Q16 = {SX_Q16};\n")
    f.write(f"static int64_t SH1_Q16 = {SH1_Q16};\n")
    f.write(f"static int64_t SH2_Q16 = {SH2_Q16};\n")
    f.write(f"static uint64_t PACKED_LN1OUT = 0x{packed_ln1_out:x}ULL;\n")
    f.write(f"static uint64_t PACKED_LN2OUT = 0x{packed_ln2_out:x}ULL;\n")

print("LN常數補充完成")

# ---- B0 用：未打包的權重陣列 ----
def c_arr_i8(name, arr):
    flat = arr.flatten()
    vals = ",".join(str(int(v)) for v in flat)
    shape = "][".join(str(s) for s in arr.shape)
    return f"static int8_t {name}[{shape}] = {{{vals}}};\n"

with open("block_data.h", "a") as f:
    f.write(c_arr_i8("WQKV_U8", Wqkv_q))
    f.write(c_arr_i8("WPROJ_U8", Wproj_q))
    f.write(c_arr_i8("WFC1_U8", Wfc1_q))
    f.write(c_arr_i8("WFC2_U8", Wfc2_q))

import os
print(f"補充未打包權重完成，檔案大小: {os.path.getsize('block_data.h')/1024:.1f} KB")
