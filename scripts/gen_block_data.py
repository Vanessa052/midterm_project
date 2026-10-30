import os
from safetensors.numpy import load_file
import numpy as np

np.random.seed(42)

W_PATH = os.environ.get("DEIT_WEIGHTS", "../deit_tiny/model.safetensors")
w = load_file(W_PATH)

D = 192          # embedding dim
SEQ = 32         # 代表性子集（完整模型是 197）
NHEAD = 3
HEAD_DIM = D // NHEAD   # 64

def g(name):
    return w[name].astype(np.float64)

# 真實權重
gamma1, beta1 = g("blocks.0.norm1.weight"), g("blocks.0.norm1.bias")
gamma2, beta2 = g("blocks.0.norm2.weight"), g("blocks.0.norm2.bias")
W_qkv, B_qkv  = g("blocks.0.attn.qkv.weight"), g("blocks.0.attn.qkv.bias")      # (576,192)
W_proj, B_proj = g("blocks.0.attn.proj.weight"), g("blocks.0.attn.proj.bias")   # (192,192)
W_fc1, B_fc1  = g("blocks.0.mlp.fc1.weight"), g("blocks.0.mlp.fc1.bias")        # (768,192)
W_fc2, B_fc2  = g("blocks.0.mlp.fc2.weight"), g("blocks.0.mlp.fc2.bias")        # (192,768)

print("shapes:", W_qkv.shape, W_proj.shape, W_fc1.shape, W_fc2.shape)
print("gamma1/beta1:", gamma1.shape, beta1.shape)

# 32 個代表性 token 輸入
X = np.random.randn(SEQ, D) * 1.0

# ---- float32 完整 forward (當 golden) ----
def layernorm_fp(x, gamma, beta, eps=1e-6):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * gamma + beta

def gelu_fp(x):
    from scipy.special import erf
    return 0.5 * x * (1 + erf(x / np.sqrt(2)))

def softmax_fp(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

h1 = layernorm_fp(X, gamma1, beta1)
qkv = h1 @ W_qkv.T + B_qkv                     # (32,576)
q, k, v = qkv[:, :192], qkv[:, 192:384], qkv[:, 384:]
def split_heads(t):
    return t.reshape(SEQ, NHEAD, HEAD_DIM).transpose(1,0,2)   # (NHEAD,SEQ,HEAD_DIM)
qh, kh, vh = split_heads(q), split_heads(k), split_heads(v)
scores = np.einsum('hqd,hkd->hqk', qh, kh) / np.sqrt(HEAD_DIM)
attn = softmax_fp(scores, axis=-1)
ctx = np.einsum('hqk,hkd->hqd', attn, vh)       # (NHEAD,SEQ,HEAD_DIM)
ctx = ctx.transpose(1,0,2).reshape(SEQ, D)      # (32,192)
attn_out = ctx @ W_proj.T + B_proj
x2 = X + attn_out
h2 = layernorm_fp(x2, gamma2, beta2)
fc1_out = h2 @ W_fc1.T + B_fc1
gelu_out = gelu_fp(fc1_out)
fc2_out = gelu_out @ W_fc2.T + B_fc2
Y_fp = x2 + fc2_out

print("\nY_fp 統計: mean=%.4f std=%.4f min=%.4f max=%.4f" %
      (Y_fp.mean(), Y_fp.std(), Y_fp.min(), Y_fp.max()))
np.save("X.npy", X); np.save("Y_fp.npy", Y_fp)
print("\nStage A 資料生成完成，float32 reference 已存檔")
