# ISA 擴充設計（5 個自訂指令）

所有指令共用編碼模式，皆為 R-type、僅限 RV64（`require_rv64`）：
MATCH = (funct3 << 12) | 0x0B
MASK = 0x707f

| 指令 | funct3 | 功能 | 設計動機 |
|---|---|---|---|
| `xqmac8` | 0 | 8×INT8 activation × 8×INT8 weight 打包 MAC | 64-bit 暫存器一次處理 8 個 int8，取代 8 次純量乘加 |
| `requant8` | 1 | INT32 累加值 → INT8 的 SRDHM 風格 requantization | 融合「乘、捨入、位移、clamp」整段序列為一條指令 |
| `ld8pi` | 2 | 載入 8 個打包 int8（64-bit）並將來源暫存器 post-increment +8 | 省掉迴圈中每次手動 `addi` 指標的額外指令 |
| `xqmac_w8a16` | 3 | 4×INT16 activation × 4×INT8 weight 打包 MAC | 支援 W8A16（int16 activation）混合精度計算 |
| `xlutacc` | 4 | 依索引讀取 LUT 一個 int32 項目並累加進目的暫存器 | 融合「位移、加法、load、加法」為一條指令，同時服務 softmax（累加 exp 總和）與 GELU（累加初值設 0 的純查表）兩種用途 |

## 指令語義

### `ld8pi`（`src/isa-extensions/ld8pi.h`）
```c
require_rv64;
uint64_t addr = RS1;
int64_t val = MMU.load<int64_t>(addr);
WRITE_REG(insn.rs1(), addr + 8);
WRITE_RD(val);
```

### `xqmac_w8a16`
4×int16 × 4×int8 累加進 rd 的 int32。相比 `xqmac8` 一次處理 8 個元素，W8A16 因 activation 寬度加倍、單一 64-bit 暫存器只能塞 4 個 int16，理論上每條指令處理的元素數減半。

### `xlutacc`（`src/isa-extensions/xlutacc.h`）
```c
require_rv64;
uint64_t base = RS1;
uint64_t idx = RS2 & 0xff;
int32_t val = MMU.load<int32_t>(base + idx*4);
int32_t acc = (int32_t)RD;
acc += val;
WRITE_RD(sext32(acc));
```

## 編碼與建置流程

所有新指令均在 `encoding.h` 新增 `MATCH_*`/`MASK_*`/`DECLARE_INSN`，並在 `riscv.mk.in` 登記語義檔後，需重新執行 `configure && make install` 才會生效（僅修改 `.mk.in` 不會自動生效）。每次新增指令編碼前皆以 `grep -n` 驗證該編碼無重複，避免位址衝突導致的無窮迴圈等難以排查的錯誤。
