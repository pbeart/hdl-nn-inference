#ifndef UTILS_HH
#define UTILS_HH

#include "verilated.h"

typedef union {
    SData sdata;
    _Float16 half ;
} HalfSData;

void flipHalfSData(HalfSData &data) {
    SData local = data.sdata;

    data.sdata = (local & 0xff) << 8;
    data.sdata |= (local & 0xff00) >> 8;
}

typedef struct {
    _Float16 actual;
    _Float16 expected;
} TestResult;

#endif