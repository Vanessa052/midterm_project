require_rv64;
uint64_t base = RS1;
uint64_t idx = RS2 & 0xff;
int32_t val = MMU.load<int32_t>(base + idx * 4);
int32_t acc = (int32_t)RD;
acc += val;
WRITE_RD(sext32(acc));
