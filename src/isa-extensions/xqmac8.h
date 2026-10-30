require_rv64;
int32_t acc = (int32_t)RD;
uint64_t a = RS1;
uint64_t w = RS2;
for (int i = 0; i < 8; i++) {
    int8_t ai = (int8_t)((a >> (8*i)) & 0xff);
    int8_t wi = (int8_t)((w >> (8*i)) & 0xff);
    acc += (int32_t)ai * (int32_t)wi;
}
WRITE_RD(sext32(acc));
