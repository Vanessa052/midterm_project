require_rv64;
int32_t acc = (int32_t)RD;
uint64_t a = RS1;   /* 4 個打包 int16 activation */
uint64_t w = RS2;   /* 4 個打包 int8 weight (低 32 bits) */
for (int i = 0; i < 4; i++) {
    int16_t ai = (int16_t)((a >> (16*i)) & 0xffff);
    int8_t wi = (int8_t)((w >> (8*i)) & 0xff);
    acc += (int32_t)ai * (int32_t)wi;
}
WRITE_RD(sext32(acc));
