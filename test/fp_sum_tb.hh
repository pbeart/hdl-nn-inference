#include <Vfp_sum.h>
#include "verilated.h"
#include "verilated_vcd_c.h"
#include <iostream>
#include "utils.hh"


TestResult test_fp_sum(_Float16 inputs[], int length) {
    VerilatedContext* contextp = new VerilatedContext;
    //contextp->commandArgs(argc, argv);
    Vfp_sum* top = new Vfp_sum{contextp};
    Verilated::traceEverOn(true);

    VerilatedVcdC* tfp = new VerilatedVcdC;

    top->trace(tfp, 100);
    tfp->open("trace.vcd");


    
    HalfSData p;

    int model_length = sizeof(top->argument_array)/sizeof(top->argument_array[0]);


    if (model_length != length) {
        throw std::runtime_error(std::string("Wrong number of sum parameters supplied, got ") + std::to_string(length) + std::string(", expected ") + std::to_string(model_length));
    };
    
    //top->eval();
    //printf("Summing: ");

    _Float16 pred_result = (_Float16)0.0;
    
    for (int i=0; i<sizeof(top->argument_array)/sizeof(top->argument_array[0]); i++) {
        p.half = (_Float16)(inputs[i]);
        pred_result += p.half;
        //printf("%f,", (double)p.half);
        top->argument_array[i] = p.sdata;
    }
    //printf("\n");
    top->eval();
    tfp->dump(0);

    HalfSData out;
    out.sdata = top->out;
    tfp->flush();
    
    delete top;
    delete contextp;
    delete tfp;

    return {out.half, pred_result};
};
