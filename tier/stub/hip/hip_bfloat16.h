#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>
struct hip_bfloat16 {
    uint16_t data;
    hip_bfloat16()=default;
    __attribute__((host)) __attribute__((device)) hip_bfloat16(float f){ union{ float f; uint32_t u; } c; c.f=f; data=(uint16_t)(c.u>>16); }
    __attribute__((host)) __attribute__((device)) operator float() const { union{ uint32_t u; float f; } c; c.u=((uint32_t)data)<<16; return c.f; }
};
