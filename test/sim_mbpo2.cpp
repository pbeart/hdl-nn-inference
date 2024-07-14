#include <Vfp_multiplybypowerof2.h>
#include "verilated.h"
#include <iostream>

typedef union {
    SData sdata;
    _Float16 half ;
} HalfSData;

void flipHalfSData(HalfSData &data) {
    SData local = data.sdata;

    data.sdata = (local & 0xff) << 8;
    data.sdata |= (local & 0xff00) >> 8;
}

int main(int argc, char** argv) {
    VerilatedContext* contextp = new VerilatedContext;
    contextp->commandArgs(argc, argv);
    Vfp_multiplybypowerof2* top = new Vfp_multiplybypowerof2{contextp};


    for (int i=0; i<1; i++) {
        HalfSData p;
        p.half = (_Float16)0.4;
        top->argumenta = p.sdata;
        
        top->eval();

        HalfSData out;
        out.sdata = top->out;

        printf("Tried %f, got %f. Equiv. to tried %x got %x\n", (double)p.half, (double)out.half, p.sdata, out.sdata);

    }
    delete top;
    delete contextp;
    return 0;
}