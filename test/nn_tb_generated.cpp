#include <cassert>
#include "nn_tb.hh"
#include <stdlib.h>
#include <vector>

int main(int argc, char** argv) {
    
    
    assert(argc >= 2);

    int set_inputs = argc - 1; // argv[0] is just program name

    _Float16 inputs[set_inputs];
    for (int i=0; i<set_inputs; i++) {
        inputs[i] = (_Float16)atof(argv[i+1]);
    };
    
   
    std::vector<_Float16> result = test_nn(inputs, set_inputs);
    

    printf("Got values: ");
    for (const _Float16 &val : result) {
        printf("%f, ", (float)val);
    };
    printf("\n");


    return 0;
}