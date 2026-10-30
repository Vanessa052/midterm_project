.section .text.init
.globl _start
_start:
    la   sp, _stack_top
    call main
    slli a0, a0, 1        # HTIF 格式：(退出碼 << 1) | 1
    ori  a0, a0, 1
    la   t0, tohost
    sd   a0, 0(t0)
1:  j    1b

.section .tohost, "aw", @progbits
.align 6
.globl tohost
tohost:   .dword 0
.align 6
.globl fromhost
fromhost: .dword 0
