require_rv64;
int64_t acc    = (int64_t)(int32_t)RS1;
int32_t m0     = (int32_t)(RS2 & 0xffffffffULL);
int32_t shift  = (int32_t)((RS2 >> 32) & 0x3f);
int32_t zp     = (int32_t)((RS2 >> 38) & 0xff);
if (zp >= 128) zp -= 256;   /* sign-extend 8-bit zero point */

/* SRDHM: saturating rounding doubling high mul */
int64_t ab = acc * (int64_t)m0;
int64_t nudge = (ab >= 0) ? (1LL << 30) : (1 - (1LL << 30));
int64_t t = (ab + nudge) / (1LL << 31);

/* RoundingShiftRight */
int64_t mask = (1LL << shift) - 1;
int64_t remainder = t & mask;
int64_t threshold = (mask >> 1) + ((t < 0) ? 1 : 0);
int64_t r = (t >> shift) + ((remainder > threshold) ? 1 : 0);

int32_t qy = (int32_t)r + zp;
if (qy > 127) qy = 127;
if (qy < -128) qy = -128;

WRITE_RD(sext32(qy));
