// а1
#pragma once

#include <cstddef>
#include <cstdint>

#if !defined(__HIP_STUB_HD)
#define __HIP_STUB_HD __attribute__((host)) __attribute__((device))
#endif

// а2
#define __HIP_DEFINE_VECTOR(T, NAME, A1, A2, A3, A4)                                        \
    struct __attribute__((aligned(A1))) NAME##1 { T x; };                                   \
    struct __attribute__((aligned(A2))) NAME##2 { T x, y; };                                \
    struct __attribute__((aligned(A3))) NAME##3 { T x, y, z; };                             \
    struct __attribute__((aligned(A4))) NAME##4 { T x, y, z, w; };                           \
    static __HIP_STUB_HD inline NAME##1 make_##NAME##1(T x) { NAME##1 r; r.x = x; return r; } \
    static __HIP_STUB_HD inline NAME##2 make_##NAME##2(T x, T y) { NAME##2 r; r.x = x; r.y = y; return r; } \
    static __HIP_STUB_HD inline NAME##3 make_##NAME##3(T x, T y, T z) { NAME##3 r; r.x = x; r.y = y; r.z = z; return r; } \
    static __HIP_STUB_HD inline NAME##4 make_##NAME##4(T x, T y, T z, T w) { NAME##4 r; r.x = x; r.y = y; r.z = z; r.w = w; return r; }

__HIP_DEFINE_VECTOR(signed char,        char,       1,  2,  1,  4)
__HIP_DEFINE_VECTOR(unsigned char,      uchar,      1,  2,  1,  4)
__HIP_DEFINE_VECTOR(short,              short,      2,  4,  2,  8)
__HIP_DEFINE_VECTOR(unsigned short,     ushort,     2,  4,  2,  8)
__HIP_DEFINE_VECTOR(int,                int,        4,  8,  4, 16)
__HIP_DEFINE_VECTOR(unsigned int,       uint,       4,  8,  4, 16)
__HIP_DEFINE_VECTOR(long,               long,       8, 16,  8, 16)
__HIP_DEFINE_VECTOR(unsigned long,      ulong,      8, 16,  8, 16)
__HIP_DEFINE_VECTOR(long long,          longlong,   8, 16,  8, 16)
__HIP_DEFINE_VECTOR(unsigned long long, ulonglong,  8, 16,  8, 16)
__HIP_DEFINE_VECTOR(float,              float,      4,  8,  4, 16)
__HIP_DEFINE_VECTOR(double,             double,     8, 16,  8, 16)

#undef __HIP_DEFINE_VECTOR

static_assert(sizeof(float4) == 16, "float4 — 16 байт");
static_assert(sizeof(ushort4) == 8, "ushort4 — 8 байт");

// к1
struct dim3 {
    uint32_t x, y, z;
    constexpr __HIP_STUB_HD dim3(uint32_t _x = 1, uint32_t _y = 1, uint32_t _z = 1)
        : x(_x), y(_y), z(_z) {}
    __HIP_STUB_HD dim3(uint3 a) : x(a.x), y(a.y), z(a.z) {}
    __HIP_STUB_HD operator uint3() const { uint3 r; r.x = x; r.y = y; r.z = z; return r; }
};
