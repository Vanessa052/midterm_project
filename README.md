# CV-Wally RISC-V ISA 擴充：量化 DeiT-Tiny 加速

針對 CORE-V Wally（CV-Wally）RV64IMC 核心設計並驗證 5 個自訂 ISA 擴充指令，
以 Spike 作為 ISA 模擬器，加速量化（quantized）DeiT-Tiny Transformer 推論。

本期中報告範圍：Spike 功能層級模擬 + C/Python 驗證，**不包含 RTL/PPA 評估**。

## 主要成果

| 項目 | 結果 |
|---|---|
| QKV 運算子加速 | **11.04x**（純軟體 → `xqmac8`+`requant8`+`ld8pi`） |
| 完整 Transformer block 加速 | **5.83x**（B0 純軟體 → X_opt 5 指令全開，12 階段 bit-exact 驗證） |
| W8A16 混合精度 | 1.91x 指令數代價，換得輸出端正確校正後 5.48x MSE 改善 |

詳見 [DESIGN.md](DESIGN.md)（ISA 設計）與 [EXPERIMENTS.md](EXPERIMENTS.md)（驗證與評估結果）。
AI 工具使用情況見 [AI_USAGE.md](AI_USAGE.md)。

## 目錄結構
cat > ~/cvw-project/cv-wally-isa-ext/README.md << 'EOF'
# CV-Wally RISC-V ISA 擴充：量化 DeiT-Tiny 加速

針對 CORE-V Wally（CV-Wally）RV64IMC 核心設計並驗證 5 個自訂 ISA 擴充指令，
以 Spike 作為 ISA 模擬器，加速量化（quantized）DeiT-Tiny Transformer 推論。

本期中報告範圍：Spike 功能層級模擬 + C/Python 驗證，**不包含 RTL/PPA 評估**。

## 主要成果

| 項目 | 結果 |
|---|---|
| QKV 運算子加速 | **11.04x**（純軟體 → `xqmac8`+`requant8`+`ld8pi`） |
| 完整 Transformer block 加速 | **5.83x**（B0 純軟體 → X_opt 5 指令全開，12 階段 bit-exact 驗證） |
| W8A16 混合精度 | 1.91x 指令數代價，換得輸出端正確校正後 5.48x MSE 改善 |

詳見 [DESIGN.md](DESIGN.md)（ISA 設計）與 [EXPERIMENTS.md](EXPERIMENTS.md)（驗證與評估結果）。
AI 工具使用情況見 [AI_USAGE.md](AI_USAGE.md)。

## 目錄結構
src/isa-extensions/ 5 個自訂指令的 Spike 語義檔 + encoding.h/riscv.mk.in 修改 patch
src/common/ bare-metal 共用 header（mytypes.h）
src/qkv-test/ QKV 運算子 kernel（真實 DeiT-Tiny 權重）
src/fc1-test/ FC1 運算子 kernel
src/w8a16-test/ W8A16 混合精度 kernel
src/block-test/ 完整 Transformer block kernel（B0 / X_opt）
verify/ 5 個指令 + LayerNorm/Softmax/GELU 的 directed+randomized 驗證
scripts/ Python 量化 / 測資生成 / benchmark 腳本
experiments/ 生成的測試資料與量化後權重（.h headers）
results/ benchmark 數字與 log

## 重現步驟

### 1. 套用 ISA 擴充到 Spike

```bash
cp src/isa-extensions/{ld8pi,xqmac8,requant8,xqmac_w8a16,xlutacc}.h  <riscv-isa-sim>/riscv/insns/
cd <riscv-isa-sim>
git apply <path-to>/src/isa-extensions/encoding_patch.diff
cd build && ../configure --prefix=$RISCV && make -j$(nproc) && make install
```

### 2. 編譯並驗證單一指令

```bash
cd verify
riscv64-elf-gcc -O2 -march=rv64imc -mabi=lp64 -mcmodel=medany -nostdlib -nostartfiles -fno-builtin \
    -I ../src/common -T link.ld -o test_ld8pi.elf crt0.S test_ld8pi.c
spike --isa=rv64imc test_ld8pi.elf; echo "exit: $?"
```

其餘指令（`test_xlutacc*`, `test_xqmac_w8a16*`, …）與 LayerNorm/Softmax/GELU 驗證（`test_layernorm_random.c` 等）編譯方式相同。

### 3. 重現完整 block 加速比

```bash
cd src/block-test
# 先用 scripts/gen_block_data.py + scripts/gen_block_full.py 生成 ../../experiments/block_data.h
riscv64-elf-gcc -O2 -march=rv64imc -mabi=lp64 -mcmodel=medany -nostdlib -nostartfiles -fno-builtin \
    -I ../common -I ../../experiments -T ../qkv-test/link.ld -o kernel_opt.elf ../qkv-test/crt0.S kernel_opt.c
riscv64-elf-gcc -O2 -march=rv64imc -mabi=lp64 -mcmodel=medany -nostdlib -nostartfiles -fno-builtin \
    -I ../common -I ../../experiments -T ../qkv-test/link.ld -o kernel_b0.elf ../qkv-test/crt0.S kernel_b0.c

spike --isa=rv64imc kernel_opt.elf; echo "X_opt exit: $?"
spike --isa=rv64imc kernel_b0.elf;  echo "B0 exit: $?"

spike -l --isa=rv64imc kernel_opt.elf 2>&1 | wc -l   # X_opt 指令數
spike -l --isa=rv64imc kernel_b0.elf  2>&1 | wc -l   # B0 指令數
```
