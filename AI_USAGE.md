# AI 工具使用說明

本專案開發過程中使用 Claude（Anthropic）作為協作工具，使用方式與範圍如下：

## 使用方式

- **ISA 指令設計討論**：在設計 `ld8pi`、`xqmac_w8a16`、`xlutacc` 三個指令時，與 Claude 討論編碼方式（funct3 分配、MATCH/MASK 計算）、語義實作（Spike `insns/*.h` 的 C++ 寫法）、以及設計動機（例如 `xlutacc` 如何同時服務 softmax 與 GELU 兩種用途）。所有設計決策（例如 attention token 長度選 32、`xlutacc` 要做成硬體指令而非純軟體）由本人確認後拍板。

- **除錯協助**：在開發過程中遇到的數個 bug，由 Claude 協助定位與修復方向，包括：
  - LayerNorm 的 Newton-Raphson 收斂失敗問題（改用保證收斂的整數開根號演算法）
  - Python/C 浮點與定點數值的 double-rounding 不一致問題
  - `M_ln2_out` scale 設錯目標的 copy-paste bug
  - Python `//` 與 C `/` 除法語義不同造成的誤差
  - Homebrew riscv64-elf-gcc 缺少 newlib 標頭檔的工具鏈問題

  所有修復方案皆經過本人以隨機測試（randomized verification）實際驗證正確性後才採用，不是直接套用 AI 給的答案。

- **Benchmark 與驗證腳本撰寫**：部分 Python 量化腳本（`gen_block_full.py` 等）與 C 驗證測試的骨架由 Claude 協助撰寫，本人負責確認量化數學（scale 選取、requant 公式）正確性並執行驗證。

- **報告與文件整理**：本 README / DESIGN.md / EXPERIMENTS.md 的內容整理與撰寫由 Claude 協助完成，內容基於本人實際執行的開發與驗證結果。

## 未使用 AI 的部分

- 所有程式碼的實際編譯、執行、驗證結果皆在本人自己的環境（macOS + Spike）中跑出，Claude 沒有直接存取或執行本專案的程式碼。
- 量化 pipeline 的核心設計決策（scale 傳遞方式、哪些 GEMM 直接 requant 到殘差流 scale）由本人決定。
- 所有最終驗證結果（exit code、benchmark 數字）皆為本人實際執行所得，未經人為修改。
