# CV-Wally RISC-V ISA 擴充：量化 DeiT-Tiny 加速

針對 CORE-V Wally（CV-Wally）RV64IMC 核心設計 5 條自訂 ISA 指令，以 Spike 作為
ISA 模擬器，加速 INT8 量化 DeiT-Tiny Transformer 推論。範圍為 ISA 層級評估（retired
instruction count），不含 RTL / PPA。

## 主要成果

| 項目 | 純軟體基準 | 全部擴充 | 加速比 |
|---|---|---|---|
| QKV GEMM（真實 DeiT-Tiny 權重） | 798,097 | 72,287 | **11.04x** |
| FC1 GEMM | 1,064,350 | 164,409 | 6.47x |
| 完整 Transformer block（12 階段 bit-exact） | 107,103,277 | 18,382,794 | **5.83x** |
| W8A16 vs W8A8（QKV） | 121,006 | 231,598 | 0.52x（換取 5.48x 精度，見 EXPERIMENTS.md） |

設計細節見 [DESIGN.md](DESIGN.md)，驗證與完整實驗見 [EXPERIMENTS.md](EXPERIMENTS.md)，
AI 工具使用見 [AI_USAGE.md](AI_USAGE.md)。

## 5 條自訂指令（custom-0, opcode 0x0B）

`xqmac8`（funct3=0）、`requant8`（1）、`ld8pi`（2）、`xqmac_w8a16`（3）、`xlutacc`（4）。

## 目錄結構
src/isa-extensions/ 5 個指令的 Spike 語義檔 + encoding_patch.diff（encoding.h / riscv.mk.in）
src/common/ bare-metal 共用檔: mytypes.h, link.ld, crt0.S
src/qkv-test/ QKV kernel（真實權重） src/fc1-test/ FC1 kernel
src/w8a16-test/ W8A8 / W8A16 kernel src/block-test/ 完整 block: kernel_b0.c / kernel_opt.c
verify/ 各指令與 LayerNorm/Softmax/GELU 的 directed + randomized 測試
scripts/ run_bench.py（驗證+benchmark）、量化與測資生成腳本
experiments/ 預先生成的測資 / 量化權重 header
results/ 預期結果 expected_results.txt

## 環境需求

- `riscv64-elf-gcc`（bare-metal，不需要 newlib；本 repo 以 `src/common/mytypes.h` 取代 `<stdint.h>`）
- 套用本 repo 擴充後的 Spike
- Python 3 + `numpy`、`scipy`、`safetensors`（僅重新生成資料時需要）

## 1. 將擴充套用到 Spike

```bash
SPIKE_SRC=<riscv-isa-sim 路徑>
cp src/isa-extensions/{xqmac8,requant8,ld8pi,xqmac_w8a16,xlutacc}.h $SPIKE_SRC/riscv/insns/
cd $SPIKE_SRC && git apply <本repo>/src/isa-extensions/encoding_patch.diff
mkdir -p build && cd build && ../configure --prefix=$RISCV && make -j && make install
```

（修改 `riscv.mk.in` 後必須重新 `configure`，否則新指令不會被編入。）

## 2. 一個指令跑完驗證與 benchmark

```bash
python3 scripts/run_bench.py                    # 全部（block 約需數分鐘）
python3 scripts/run_bench.py verify qkv fc1 w8a16   # 或指定項目
```

項目：`verify`（verify/test_*.c，exit code 必須為 0）、`qkv`、`fc1`、`w8a16`、`block`。
指令數 = `spike -l` 輸出行數（每條 retired instruction 一行）。每個 kernel 都先對 golden
reference 驗證（exit code 0）才計數。可用環境變數 `RISCV_CC`、`SPIKE` 覆寫工具路徑。

預期輸出（完整版見 [results/expected_results.txt](results/expected_results.txt)）：
QKV base 798,097 | x2e 72,287 (11.04x)
FC1 base 1,064,350 | x2 164,409 (6.47x)
W8A16 base(W8A8) 121,006 | w8a16 231,598 (0.52x)
Block b0 107,103,277 | x_opt 18,382,794 (5.83x)

## 3.（選用）重新生成量化資料

測資 header 已預先生成並收錄於 `experiments/` 與各 `src/*-test/`。若要重新生成，需要
DeiT-Tiny 權重檔（`model.safetensors`）：

```bash
export DEIT_WEIGHTS=/path/to/deit_tiny/model.safetensors
mkdir -p work && cd work
python3 ../scripts/gen_block_data.py      # float golden（X.npy, Y_fp.npy）
python3 ../scripts/gen_block_full.py      # 輸出 block_data.h（複製到 experiments/）
python3 ../scripts/gen_data_w8a16.py      # W8A16 實驗資料（gen_data.h）
```
