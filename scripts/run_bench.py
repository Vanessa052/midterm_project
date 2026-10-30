#!/usr/bin/env python3
"""
可重現的 benchmark 腳本：編譯 -> 執行驗證 -> 計算指令數 -> 輸出比較表
用法: python3 run_bench.py <kernel目錄> <kernel名稱>
例如: python3 run_bench.py ../fc1-test fc1
"""
import subprocess
import sys
import os

CC = "riscv64-elf-gcc"
SPIKE = "spike"
CFLAGS_BASE = "-O2 -mabi=lp64 -mcmodel=medany -ffreestanding -nostdlib -nostartfiles -I /Users/zhengshurong/cvw-project/common"

def run(cmd, timeout=60):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout, r.stderr

def compile_variant(workdir, src, march, defines, out_elf):
    def_str = " ".join(f"-D{d}" for d in defines)
    cmd = f"cd {workdir} && {CC} {CFLAGS_BASE} -march={march} {def_str} -T link.ld crt0.S {src} -o {out_elf}"
    rc, out, err = run(cmd)
    if rc != 0:
        print(f"  [COMPILE FAIL] {out_elf}")
        print(err)
        return False
    return True

def run_spike(workdir, elf, march):
    cmd = f"cd {workdir} && {SPIKE} --isa={march} {elf}"
    rc, out, err = run(cmd, timeout=90)
    return rc

def count_trace(workdir, elf, march):
    """用 -l trace 計算總指令數（適合小 kernel）"""
    trace = f"{elf}.trace.log"
    cmd = f"cd {workdir} && {SPIKE} -l --isa={march} {elf} 2> {trace}"
    run(cmd, timeout=90)
    with open(os.path.join(workdir, trace)) as f:
        n = sum(1 for _ in f)
    os.remove(os.path.join(workdir, trace))
    return n

def bench(workdir, src, variants, march="rv64imc"):
    """
    variants: list of (name, defines, exit_code_is_verification)
    例如 [("b0", [], True), ("x2", ["USE_X2"], True)]
    """
    results = {}
    for name, defines, verify in variants:
        elf = f"{src.rsplit('.',1)[0]}_{name}.elf"
        print(f"[{name}] compiling...")
        if not compile_variant(workdir, src, march, defines, elf):
            continue

        if verify:
            rc = run_spike(workdir, elf, march)
            status = "PASS" if rc == 0 else f"FAIL(exit={rc})"
            print(f"[{name}] verify: {status}")
            if rc != 0:
                continue

        print(f"[{name}] counting instructions (this may take a while)...")
        n = count_trace(workdir, elf, march)
        results[name] = n
        print(f"[{name}] instructions: {n}")

    return results

def print_table(results, baseline="b0"):
    print("\n" + "="*50)
    print(f"{'Version':<10}{'Instructions':>15}{'Speedup':>12}")
    print("-"*50)
    base = results.get(baseline)
    for name, n in results.items():
        speedup = f"{base/n:.2f}x" if base and n else "-"
        print(f"{name:<10}{n:>15}{speedup:>12}")
    print("="*50)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python3 run_bench.py <workdir> <src.c> [march]")
        sys.exit(1)

    workdir = sys.argv[1]
    src = sys.argv[2]
    march = sys.argv[3] if len(sys.argv) > 3 else "rv64imc"

    # 這裡定義要跑哪些版本，依你的 kernel 修改
    variants = [
        ("b0", [], True),
        ("x2", ["USE_X2"], True),
        ("x2b", ["USE_X2B"], True),
        ("x2c", ["USE_X2C"], True),
        ("x2d", ["USE_X2D"], True),
        ("x2e", ["USE_X2E"], True),
    ]

    results = bench(workdir, src, variants, march)
    print_table(results)
