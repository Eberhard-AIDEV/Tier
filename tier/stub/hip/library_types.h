// а1
#pragma once

typedef enum hipDataType {
    HIP_R_32F  = 0,
    HIP_R_64F  = 1,
    HIP_R_16F  = 2,
    HIP_R_8I   = 3,
    HIP_C_32F  = 4,
    HIP_C_64F  = 5,
    HIP_C_16F  = 6,
    HIP_C_8I   = 7,
    HIP_R_8U   = 8,
    HIP_C_8U   = 9,
    HIP_R_32I  = 10,
    HIP_C_32I  = 11,
    HIP_R_32U  = 12,
    HIP_C_32U  = 13,
    HIP_R_16BF = 14,
    HIP_C_16BF = 15,
    HIP_R_4I   = 16,
    HIP_R_4U   = 17,
    HIP_R_16I  = 18,
    HIP_R_16U  = 19,
    HIP_R_64I  = 20,
    HIP_R_64U  = 21,
    HIP_R_8F_E4M3 = 28,
    HIP_R_8F_E5M2 = 29
} hipDataType;

typedef enum hipLibraryPropertyType {
    HIP_LIBRARY_MAJOR_VERSION = 0,
    HIP_LIBRARY_MINOR_VERSION = 1,
    HIP_LIBRARY_PATCH_LEVEL   = 2
} hipLibraryPropertyType;
