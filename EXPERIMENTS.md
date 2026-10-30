# 驗證方法與評估結果

## 驗證方法

每條新指令皆採 **directed + randomized** 兩階段驗證：

- **Directed tests**：針對邊界條件手動設計案例（例如 `xlutacc` 的索引遮罩測試、`ld8pi` 的 post-increment 正確性）。
- **Randomized tests**：`ld8pi`、`xlutacc`、`xqmac_w8a16` 各執行 2,000 筆隨機案例，與 Python 參考實作逐位元比對，皆為 **0 failures**。
- **LayerNorm / Softmax / GELU**：50 案例 × 192 元素（LayerNorm）、30 案例 × 32 元素（Softmax）、200 案例（GELU）規模的隨機測試，對照浮點 golden reference，皆通過容差檢查。
- **Complete block**：以 Python 建立 bit-exact 的量化整數 golden reference，`kernel_opt.c`（X_opt，5 指令全開）與 `kernel_b0.c`（B0，純軟體）皆對 12 個中繼階段逐一比對，兩者皆取得 **exit code 0**（全階段 bit-exact 通過）。

## Evaluation Coverage 三層成果

| 層級 | 要求 | 達成內容 |
|---|---|---|
| Microbenchmark | ≥1 | `xqmac8`、`requant8`、`ld8pi`、`xqmac_w8a16`、`xlutacc` 個別 directed + randomized 驗證 |
| Operators | ≥2 | QKV GEMM（真實 DeiT-Tiny 權重）、FC1、LayerNorm、Softmax、GELU |
| Complete block | ≥1 | 完整 DeiT-Tiny Transformer block（LayerNorm→QKV→Attention→Proj→residual→LayerNorm→FC1→GELU→FC2→residual），12 階段 bit-exact 驗證通過 |

### QKV 運算子加速

8-way unroll + 具名純量累加器 + `ld8pi` 取代陣列索引：

| 版本 | 指令數 | 加速比 |
|---|---|---|
| 純軟體基準 | 798,730（概算） | 1.00x |
| X2d（`xqmac8` + `requant8`） | 85,825 | 9.30x |
| **X2e（+ `ld8pi`）** | **72,287** | **11.04x** |

### Complete block 加速

以 SEQ=32（attention token 長度）、D=192、NHEAD=3、HEAD_DIM=64 的完整 Transformer block 做端到端比較，兩版本對同一組 Python golden reference 做逐階段 bit-exact 驗證：

| Kernel | 指令數（Spike `-l` trace 行數） | 說明 |
|---|---|---|
| B0（純軟體基準） | 107,103,277 | 無自訂指令，scalar GEMM / 軟體 requant / 顯式 LUT load+add |
| **X_opt（5 個自訂指令全開）** | **18,382,794** | `xqmac8` + `requant8` + `ld8pi` + `xlutacc`（softmax、GELU） |

**整體加速比 ≈ 5.83x**（107,103,277 ÷ 18,382,794）

此數字低於 QKV 單獨運算子的 11.04x，原因是一個典型的 **Amdahl's Law** 現象：LayerNorm 的整數開根號（`isqrt_u64` 逐位元演算法）與倒數計算，在 B0 與 X_opt 兩版本中是完全相同的純軟體序列（本專案的 5 個擴充指令並未涵蓋這一段），因此它構成一個固定不變的 overhead；當 GEMM 部分被加速 8～11x 後，這段未被加速的序列在總指令數中的佔比相對提高，進而拉低了整個 block 的端到端加速比。

## W8A16 精度 / 效能權衡分析

`xqmac_w8a16`（4×int16 activation × 4×int8 weight）相對 `xqmac8`（8×int8 × 8×int8）基準的效能與精度權衡，以真實 QKV 權重測試：

**效能：** W8A16 需要 231,598 指令，相對 W8A8 的 121,006 指令，**慢 1.91 倍**——與理論預期一致（每條指令處理的元素數因 int16 佔用空間加倍而減半）。

**精度（三組對照實驗，刻意控制變因排除混淆因子）：**

| 情境 | 設定 | MSE 改善倍數 |
|---|---|---|
| (1) 平滑 activation，無 outlier | int8 本身未飽和 | 1.42x（改善幅度有限） |
| (2) Outlier activation（3% channel ×20），輸出 scale 未重新校正 | 輸出端 22.6% 元素因 requant 飽和 | **1.00x（看不出改善）** |
| (3) 同上，輸出 scale 重新校正至真實動態範圍 | 排除飽和混淆因子 | **5.48x（真實的精度提升）** |

情境 (2) 的「零改善」結果一開始令人意外，追查後發現是輸出量化 scale（Sy）沿用了舊的、未涵蓋 outlier 分布的校正值，導致 22.6% 的輸出在 requant 階段飽和，把 W8A16 原本該有的精度優勢完全蓋掉。將 Sy 重新校正至實際輸出動態範圍後，才顯現出 W8A16 真正帶來的 5.48x MSE 改善。

**結論：** W8A16 用約 2 倍指令數的代價換取精度，但其效益高度依賴**輸出端量化是否正確校正**——若輸出 scale 沒跟著輸入精度提升而重新校正，再精細的輸入表示也會被下游的飽和吃掉。這是量化系統設計中「單一變因控制」的實際案例。
