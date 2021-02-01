#include <stdlib.h>
#include <iostream>
#include <math.h>
#include "../include/dcpagerank_c_interface.h"
#include "../include/readData.hpp"

using namespace std;

/**
 * Print the elements of a vector to stdout.
 * @param vec - the vector to be printed
 * @param vec_length - the length of the vector
 * @return
 */
template <typename T>
void print_vector(T* vec, int64_t vec_length){
    for(size_t i = 0; i < (size_t) vec_length; ++i){
        cout << vec[i] << " ";
    }
    cout << endl;
}

int main()
{
    cout << "test dcpagerank with K_(3, 3)" << endl;

    // Set up inputs to the dcpagerank functions
    // First, set up the graph matrices. This is K_(3, 3) with an extra edge (0-1).
    int64_t n = 6;
    int64_t ai[7] = {0, 4, 7, 10, 13, 16, 19};
    int64_t aj[19] = {1, 3, 4, 5, 3, 4, 5, 3, 4, 5, 0, 1, 2, 0, 1, 2, 0, 1, 2};

    // Set up the pagerank parameters
    double alpha = 0.01;
    double eps = 0.001;
    int64_t seedids[1] = {0};
    int64_t nseedids = 1;
    int64_t maxsteps = 100;

    // Set up the output vectors
    int64_t xids_1[6];
    int64_t xids_2[6];
    double values_1[6];
    double values_2[6];
    int64_t xlength = 6;

    // Call the pagerank function itself.
#ifdef DEBUG
    cout << "Running pagerank code..." << endl;
#endif
    int64_t actual_length = dcpagerank64(n, ai, aj, 0,
                                         alpha,eps,seedids, nseedids, maxsteps,
                                         xids_1, xids_2, xlength, values_1, values_2, true);

    // Display the results.
    cout << endl << "actual length: " << actual_length << endl;
    cout << "nonzero pagerank sets and values" << endl;
    cout << "xids_1: ";
    print_vector(xids_1, actual_length);
    cout << "xids_2: ";
    print_vector(xids_2, actual_length);
    cout << "values_1: ";
    print_vector(values_1, actual_length);
    cout << "values_2: ";
    print_vector(values_2, actual_length);

    return EXIT_SUCCESS;
}
