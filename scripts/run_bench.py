#!/usr/bin/env python3
"""
可重現的驗證 + benchmark 腳本。

用法（在 repo 任何位置皆可）:
    python3 scripts/run_bench.py                 # 跑全部: verify qkv fc1 w8a16 block
    python3 scripts/run_bench.py verify qkv      # 只跑指定項目

項目:
    verify  src: verify/test_*.c，每個測試必須 exit code 0
    qkv     src/qkv-test/qkv_real2.c   各 USE_* 版本 vs base
    fc1     src/fc1-test/fc1.c         各 USE_* 版本 vs base
    w8a16   src/w8a16-test/qkv_w8a16.c base=W8A8, USE_W8A16=W8A16
    block   src/block-test/kernel_b0.c vs kernel_opt.c（完整 Transformer block）

環境變數: RISCV_CC (預設 riscv64-elf-gcc)、SPIKE (預設 spike)
指令數 = `spike -l` 輸出的行數（每個 retired instruction 一行）。
"""
import glob
import os
import re
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
COMMON = os.path.join(ROOT, "src", "common")
EXPERIMENTS = os.path.join(ROOT, "experiments")
BUILD = os.path.join(ROOT, "build")
CC = os.environ.get("RISCV_CC", "riscv64-elf-gcc")
SPIKE = os.environ.get("SPIKE", "spike")
MARCH = "rv64imc"


def compile_elf(src, defines, tag):
    os.makedirs(BUILD, exist_ok=True)
    out = os.path.join(BUILD, tag + ".elf")
    cmd = [CC, "-O2", f"-march={MARCH}_zicsr", "-mabi=lp64", "-mcmodel=medany",
           "-ffreestanding", "-nostdlib", "-nostartfiles", "-fno-builtin",
           "-I", COMMON, "-I", EXPERIMENTS, "-I", os.path.dirname(src)]
    cmd += [f"-D{d}" for d in defines]
    cmd += ["-T", os.path.join(COMMON, "link.ld"),
            os.path.join(COMMON, "crt0.S"), src, "-o", out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  [COMPILE FAIL] {tag}\n{r.stderr}")
        return None
    return out


def run_spike(elf):
    r = subprocess.run([SPIKE, f"--isa={MARCH}", elf],
                       capture_output=True, timeout=1800)
    return r.returncode


def count_instructions(elf):
    """串流計算 spike -l 的行數，不把 trace 寫到磁碟。"""
    p = subprocess.Popen([SPIKE, "-l", f"--isa={MARCH}", elf],
                         stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    n = 0
    while True:
        chunk = p.stderr.read(1 << 20)
        if not chunk:
            break
        n += chunk.count(b"\n")
    p.wait()
    return n


def detect_defines(src):
    found = []
    with open(src) as f:
        for m in re.finditer(r"\bUSE_[A-Z0-9_]+\b", f.read()):
            if m.group(0) not in found:
                found.append(m.group(0))
    return found


def auto_variants(src):
    return [("base", src, [])] + [(d[4:].lower(), src, [d])
                                  for d in detect_defines(src)]


def print_table(results, baseline):
    base = results.get(baseline)
    print("\n" + "=" * 52)
    print(f"{'Version':<14}{'Instructions':>16}{'Speedup':>14}")
    print("-" * 52)
    for name, n in results.items():
        sp = f"{base / n:.2f}x" if base else "-"
        print(f"{name:<14}{n:>16,}{sp:>14}")
    print("=" * 52)


def run_bench_suite(title, variants, baseline):
    print(f"\n### {title}")
    results, ok = {}, True
    for name, src, defines in variants:
        tag = f"{os.path.basename(src)[:-2]}_{name}"
        print(f"[{name}] compile + verify ...", flush=True)
        elf = compile_elf(src, defines, tag)
        if not elf:
            ok = False
            continue
        rc = run_spike(elf)
        if rc != 0:
            print(f"[{name}] VERIFY FAIL (exit={rc})")
            ok = False
            continue
        print(f"[{name}] verify PASS, counting instructions ...", flush=True)
        results[name] = count_instructions(elf)
    if results:
        print_table(results, baseline)
    return ok


def run_verify_suite():
    print("\n### verify (每個測試 exit code 必須為 0)")
    ok = True
    tests = sorted(glob.glob(os.path.join(ROOT, "verify", "test_*.c")))
    for src in tests:
        name = os.path.basename(src)[:-2]
        elf = compile_elf(src, [], "verify_" + name)
        if not elf:
            ok = False
            continue
        rc = run_spike(elf)
        print(f"  {name:<32}{'PASS' if rc == 0 else f'FAIL(exit={rc})'}")
        ok = ok and rc == 0
    return ok


def p(*parts):
    return os.path.join(ROOT, *parts)


SUITES = {
    "verify": lambda: run_verify_suite(),
    "qkv": lambda: run_bench_suite(
        "QKV GEMM (real DeiT-Tiny weights)",
        auto_variants(p("src", "qkv-test", "qkv_real2.c")), "base"),
    "fc1": lambda: run_bench_suite(
        "FC1 GEMM", auto_variants(p("src", "fc1-test", "fc1.c")), "base"),
    "w8a16": lambda: run_bench_suite(
        "W8A8 (base) vs W8A16",
        auto_variants(p("src", "w8a16-test", "qkv_w8a16.c")), "base"),
    "block": lambda: run_bench_suite(
        "Complete Transformer block: B0 (software) vs X_opt (5 custom insns)",
        [("b0", p("src", "block-test", "kernel_b0.c"), []),
         ("x_opt", p("src", "block-test", "kernel_opt.c"), [])], "b0"),
}

if __name__ == "__main__":
    wanted = sys.argv[1:] or list(SUITES)
    bad = [w for w in wanted if w not in SUITES]
    if bad:
        print(f"未知項目: {bad}，可用: {list(SUITES)}")
        sys.exit(2)
    all_ok = True
    for w in wanted:
        all_ok = SUITES[w]() and all_ok
    print("\nALL PASS" if all_ok else "\nSOME FAILED")
    sys.exit(0 if all_ok else 1)
