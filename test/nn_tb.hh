#include <Vnn.h>
#include "verilated.h"
#include "verilated_vcd_c.h"
#include <iostream>
#include "utils.hh"
#include <vector>

std::vector<_Float16> test_nn(_Float16 inputs[], int length) {
    VerilatedContext* contextp = new VerilatedContext;

    Vnn* top = new  Vnn{contextp};
    Verilated::traceEverOn(true);

    VerilatedVcdC* tfp = new VerilatedVcdC;

    top->trace(tfp, 100);
    tfp->open("trace.vcd");

    
    
    HalfSData p;

    int model_inputs = sizeof(top->input_array)/sizeof(top->input_array[0]);
    int model_outputs = sizeof(top->output_array)/sizeof(top->output_array[0]);

    if (model_inputs != length) {
        throw std::runtime_error(std::string("Wrong number of NN input parameters supplied, got ") + std::to_string(length) + std::string(", expected ") + std::to_string(model_inputs));
    };
    
    //top->eval();
    //printf("Summing: ");

    //_Float16 pred_result = (_Float16)0.0;
    
    for (int i=0; i<sizeof(top->input_array)/sizeof(top->input_array[0]); i++) {
        p.half = (_Float16)(inputs[i]);
        //pred_result += p.half;
        //printf("%f,", (double)p.half);
        top->input_array[i] = p.sdata;
    }
    //printf("\n");
    top->eval();
    tfp->dump(0);

    std::vector<_Float16> out;
    for (int i=0; i<model_outputs; i++) {
        p.sdata = top->output_array[i];
        out.push_back(p.half);
    };
    tfp->flush();
    
    delete top;
    delete contextp;
    delete tfp;

    return out;
};
