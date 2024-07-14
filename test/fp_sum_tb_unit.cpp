#include <cassert>
#include "fp_sum_tb.hh"
#include <stdlib.h>

int main(int argc, char** argv) {
    
    
    assert(argc >= 2);

    int set_inputs = argc - 1; // argv[0] is just program name

    _Float16 inputs[set_inputs];
    for (int i=0; i<set_inputs; i++) {
        inputs[i] = (_Float16)atof(argv[i+1]);
    };
    
   
    TestResult result = test_fp_sum(inputs, set_inputs);
    

    printf("Got %f, predicted %f\n",  (double)result.actual, (double)result.expected);


    return 0;
}