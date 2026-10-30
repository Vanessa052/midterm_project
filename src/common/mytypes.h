/* riscv64-elf-gcc (Homebrew) 這個版本沒有附 newlib，<stdint.h> 會失敗。
   自己定義 bare-metal 需要的固定寬度整數型態，不依賴系統 libc。
   對應 RV64 LP64 ABI：long 是 64-bit，int 是 32-bit。 */
#ifndef MYTYPES_H
#define MYTYPES_H

typedef signed char    int8_t;
typedef unsigned char   uint8_t;
typedef short            int16_t;
typedef unsigned short   uint16_t;
typedef int              int32_t;
typedef unsigned int     uint32_t;
typedef long             int64_t;
typedef unsigned long    uint64_t;

#endif
