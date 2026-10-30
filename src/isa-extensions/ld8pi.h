require_rv64;
uint64_t addr = RS1;
int64_t val = MMU.load<int64_t>(addr);
WRITE_REG(insn.rs1(), addr + 8);
WRITE_RD(val);
