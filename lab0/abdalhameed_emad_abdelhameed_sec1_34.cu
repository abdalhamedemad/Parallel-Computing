// abdalhameed_emad_abdelhameed_sec1_34.c
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    if (argc < 4) {
        printf("please enter with these format : %s n_rows n_cols n_rows*n_cols numbers\n", argv[0]);
        return 1;
    }
    // get the numbers of rows
    int n_rows = atoi(argv[1]);
    // get the numbers of columns
    int n_cols = atoi(argv[2]);
    // calculate the total numbers of elements in order to alocate the memory
    int total_elements = n_rows * n_cols;
    // allocate the array
    int *numbers = (int*)malloc(total_elements * sizeof(int));

    // read the array
    for (int i = 0; i < total_elements; ++i) {
        numbers[i] = atoi(argv[i + 3]);
        //printf("%d \n",numbers[i]);
    }

    int result = 0;
    // calc the summation
    for (int j = 0; j < n_cols; ++j) {
        int sum = 0;
        for (int i = 0; i < n_rows; ++i) {
              int  n = numbers[i * n_cols + j];
              int mul = 1;
              if (n == 0) 
                  mul = 10;
              else { 
                    while (n != 0) { 
                        n = n / 10; 
                        mul = mul*10; 
                        }
                    }
                sum = sum * mul +  numbers[i * n_cols + j] ;

            }
            //printf("%d \n",sum);
            result += sum;
    }

    // printing the result
    printf("%d\n", result);
    
    
    // free the memory
    free(numbers);


    return 0;
}