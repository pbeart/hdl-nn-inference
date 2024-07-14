#include <cassert>
#include "fp_sum_tb.hh"
#include <stdlib.h>


int main() {
    printf("Inputs, Expected, Actual\n");

    // from https://stackoverflow.com/questions/12911299/read-csv-file-in-c
    char line[1024];
    while (fgets(line, 1024, stdin))
    {
        char* tmp = strdup(line);
        
        _Float16 inputs[512]; // nasty hack, limit to 512 inputs for now

        const char* tok;
        int num=0;
        for (tok = strtok(line, ","); tok && *tok; tok = strtok(NULL, ",\n"))
        {
            assert(num < 512); 
            inputs[num++] = (_Float16)atof(tok);
            printf("%f ", (double)((_Float16)atof(tok))); // log actual (nearly) correct precision
        };
    
        TestResult result = test_fp_sum(inputs, num);

        printf(", %f, %f\n",  (double)result.expected, (double)result.actual);

        // NOTE strtok clobbers tmp
        free(tmp);
    }


    return 0;
}