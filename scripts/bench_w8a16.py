import sys
sys.path.insert(0, ".")
from run_bench import bench, print_table

variants = [
    ("w8a8",  [], True),
    ("w8a16", ["USE_W8A16"], True),
]

results = bench("../w8a16-test", "qkv_w8a16.c", variants, march="rv64imc")
print_table(results, baseline="w8a8")
