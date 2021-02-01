/**
 * aclpagerank with C interface. It takes an unweighted and undirected graph with CSR representation
 * and some seed vetrices as input and output the approximate pagerank vector. Choose different C interface 
 * based on the data type of your input.
 *
 * INPUT:
 *     n        - the number of vertices in the graph
 *     ai,aj    - Compressed sparse row representation
 *     offset   - offset for zero based arrays (matlab) or one based arrays (julia)
 *     alpha    - value of alpha
 *     eps      - value of epsilon
 *     seedids  - the set of indices for seeds
 *     nseedids - the number of indices in the seeds
 *     maxsteps - the max number of steps
 *     xlength  - the max number of ids in the solution vector
 *     xids     - the solution vector, i.e. the vertices with nonzero pagerank value
 *     values   - the pagerank value vector for xids (already sorted in decreasing order)
 *
 * OUTPUT:
 *     actual_length - the number of nonzero entries in the solution vector
 *
 * COMPILE:
 *     make aclpagerank
 *
 * EXAMPLE:
 *     Use functions from readData.hpp to read a graph and seed from files.
 *     int64_t xlength = 100;
 *     double alpha = 0.99;
 *     double eps = pow(10,-7);
 *     int64_t maxstep = (size_t)1/(eps*(1-alpha));
 *     int64_t* xids = (int64_t*)malloc(sizeof(int64_t)*m);
 *     double* values = (double*)malloc(sizeof(double)*m);
 *     int64_t actual_length =  aclpagerank64(m,ai,aj,0,alpha,eps,seedids,
 *                                            nseedids,maxstep,xids,xlength,values);
 */

#include <stdio.h>
#include <stdlib.h>
#include <unordered_map>
#include <queue>
#include <iostream>
#include <algorithm>
#include <stdint.h>

#include "include/aclpagerank_c_interface.h"
#include "include/routines.hpp"

using namespace std;

template<typename vtype>
bool myobject (pair <vtype, double> i, pair <vtype, double> j) { return (i.second>j.second);}

uint32_t aclpagerank32(
        uint32_t n, uint32_t* ai, uint32_t* aj, uint32_t offset,
        double alpha,
        double eps, 
        uint32_t* seedids, uint32_t nseedids,
        uint32_t maxsteps,
        uint32_t* xids, uint32_t xlength, double* values)
{
    // Construct a graph object from the sparse adjacency matrix. Notice that for the unweighted calculation, the actual
    // data vector a is not needed - all of the edge connectivity information is given by ai and aj only.
    graph<uint32_t,uint32_t> g(ai[n],n,ai,aj,NULL,offset,NULL);

    // Call the pagerank method on the graph to compute the pagerank itself and return the result.
    uint32_t actual_length = g.aclpagerank(alpha, eps, seedids, nseedids, maxsteps, xids, xlength, values);
    return actual_length;
}

int64_t aclpagerank64(
        int64_t n, int64_t* ai, int64_t* aj, int64_t offset, 
        double alpha, 
        double eps, 
        int64_t* seedids, int64_t nseedids,
        int64_t maxsteps,
        int64_t* xids, int64_t xlength, double* values)
{
    // Construct a graph object from the sparse adjacency matrix. Notice that for the unweighted calculation, the actual
    // data vector a is not needed - all of the edge connectivity information is given by ai and aj only.
    graph<int64_t,int64_t> g(ai[n],n,ai,aj,NULL,offset,NULL);

    // Call the pagerank method on the graph to compute the pagerank itself and return the result.
    int64_t actual_length = g.aclpagerank(alpha, eps, seedids, nseedids, maxsteps, xids, xlength, values);
    return actual_length;
}

uint32_t aclpagerank32_64(
        uint32_t n, int64_t* ai, uint32_t* aj, uint32_t offset, 
        double alpha, 
        double eps, 
        uint32_t* seedids, uint32_t nseedids,
        uint32_t maxsteps,
        uint32_t* xids, uint32_t xlength, double* values)
{
    // Construct a graph object from the sparse adjacency matrix. Notice that for the unweighted calculation, the actual
    // data vector a is not needed - all of the edge connectivity information is given by ai and aj only.
    graph<uint32_t,int64_t> g(ai[n],n,ai,aj,NULL,offset,NULL);

    // Call the pagerank method on the graph to compute the pagerank itself and return the result.
    uint32_t actual_length = g.aclpagerank(alpha, eps, seedids, nseedids, maxsteps, xids, xlength, values);
    return actual_length;
}

template<typename vtype, typename itype>
vtype graph<vtype,itype>::aclpagerank(double alpha, double eps, vtype* seedids, vtype nseedids,
                                      vtype maxsteps, vtype* xids, vtype xlength, double* values)
{
    // Call the internal pprgrow method to compute the pagerank.
    vtype actual_length;
    actual_length=pprgrow(alpha, eps, seedids, nseedids, maxsteps, xids, xlength, values);
    return actual_length;
}

/** 
 * pprgrow compute the approximate pagerank vector locally. It may be useful to refer to [ACL08] for the formal
 * description of this algorithm for computing the approximate personalised pagerank vector.
 *
 * INUPUT:
 *     rows     - a self defined struct which contains all the info of a CSR based graph
 *     alpha    - value of alpha
 *     eps      - value of epsilon
 *     seedids  - the set of indices for seeds
 *     nseedids - the number of indices in the seeds
 *     maxsteps - the max number of steps
 *     xlength  - the max number of ids in the solution vector
 *     xids     - the solution vector, i.e. the vertices with nonzero pagerank value
 *     values   - the pagerank value vector for xids (already sorted in decreasing order)
 *
 * OUTPUT:
 *     actual_length - the number of nonzero entries in the solution vector
 */
template<typename vtype, typename itype>
vtype graph<vtype,itype>::pprgrow(double alpha, double eps,vtype* seedids, vtype nseedids,
                                  vtype maxsteps, vtype* xids, vtype xlength, double* values)
{
    // The x_map variable will store the approximate personalised pagerank vector at the end of the algorithm.
    // The r_map variable stores the 'residuals' throughout the algorithm. The algorithm guarantees that the residual
    // for some vertex v will be less than (eps * deg(v)).
    // Both of these are stored as sparse arrays represented by unordered_maps.
    unordered_map<vtype, double> x_map;
    unordered_map<vtype, double> r_map;
    typename unordered_map<vtype, double>::const_iterator x_iter, r_iter;

    // This queue will store all vertices whose value on the residual vector r is greater than the epsilon error
    // parameter.
    queue<vtype> Q;

    // Initialise the vectors x and r according to the seed nodes. At the beginning of the algorithm, all of the
    // 'weight' is on the seed nodes in the residual vector.
    for(size_t i = 0; i < (size_t)nseedids; i ++){
        r_map[seedids[i] - offset] = 1;
        x_map[seedids[i] - offset] = 0;
    }

    // Initialise the queue. A vertex is included in the queue if its residual value is greater than (eps * deg)
    for(r_iter = r_map.begin(); r_iter != r_map.end(); ++r_iter){
        if(r_iter->second >= eps * get_degree_unweighted(r_iter->first)){
            Q.push(r_iter->first);
        }
    }

    // Begin the main loop of the algorithm. This loop corresponds to the PUSH algorithm in [ACL08].
    // j will be the index of the vertex we are performing the current 'push' operation on.
    vtype j;
    double xj, rj, delta;
    vtype steps_count = 0;
    while(Q.size()>0 && steps_count<maxsteps){
        // Pull a vertex off the front of the queue and find the corresponding residual value, rj
        j = Q.front();
        Q.pop();
        x_iter = x_map.find(j);
        r_iter = r_map.find(j);
        rj = r_iter->second;

        // Update x[j] - step 1(a) of push algorithm in [ACL08]
        if(x_iter == x_map.end()){
            // If the vertex j is not already in the sparse x matrix, then add it, with the value (alpha * rj)
            xj = alpha * rj;
            x_map[j] = xj;
        }
        else{
            // If the vertex is already in x, then add (alpha * rj)
            xj = x_iter->second + alpha*rj;
            x_map.at(j) = xj;
        }

        // Update r[j] - step 1(b) of push algorithm in [ACL08]
        r_map.at(j) = ((1-alpha)/2)*rj;

        // Check whether r[j] still exceeds the threshold (eps * deg(j)) and so still needs to be on the queue to be
        // pushed again.
        if (r_map[j] >= eps * get_degree_unweighted(j)) {
            Q.push(j);
        }

        // Update x[u] for u in neighbours(j) - step 1(c) of push algorithm in [ACL08]
        // delta is the amount that each x[u] should be increased
        delta = ((1-alpha)/2)*(rj/get_degree_unweighted(j));
        vtype u;
        double ru_new, ru_old;
        for(itype i = ai[j] - offset; i < ai[j+1] - offset; i++){
            // u is a neighbour of j
            u = aj[i] - offset;

            // Find the vertex u in r if it has a value so far
            r_iter = r_map.find(u);
            if(r_iter == r_map.end()){
                // If u was not already present in r, then add it with the value delta.
                ru_old = 0;
                ru_new = delta;
                r_map[u] = ru_new;
            }
            else{
                // If us was already present in r, increase its value by delta
                ru_old = r_iter->second;
                ru_new = delta + ru_old;
                r_map.at(u) = ru_new;
            }

            // If r[u] exceeds the error threshold (eps * deg(u)) and it didn't previously, then add it to the queue
            // of vertices to be processed.
            if(ru_new > eps * get_degree_unweighted(u) && ru_old <= eps * get_degree_unweighted(u)){
                Q.push(u);
            }
        }

        // Make sure to count the number of steps so we stop if we hit the threshold.
        steps_count ++;
    }

    // The remainder of this method is simply choosing which vertices to return in the final xids vector since the
    // caller may have specified a limit on the number of vertices in the xlength parameter.
    // map_size is the number of non-zero elements in the approximate pagerank vector.
    vtype map_size = x_map.size();

    // Create an array object possible_nodes from the unordered_map x_map
    // They contain the same data, just as different structures.
    pair <vtype, double>* possible_nodes = new pair <vtype, double>[map_size];
    int i = 0;
    for(x_iter = x_map.begin(); x_iter != x_map.end(); ++x_iter){
        possible_nodes[i].first = x_iter->first;
        possible_nodes[i].second = x_iter->second;
        i++;
    }

    // Add the vertices with the largest pagerank values to the output vectors.
    sort(possible_nodes, possible_nodes + map_size, myobject<vtype>);
    for(j = 0; j < min(map_size, xlength); j ++){
        xids[j] = possible_nodes[j].first + offset;
        values[j] = possible_nodes[j].second;
    }
    delete [] possible_nodes;

    // Return the number of non-zero entries in the original solution vector.
    return map_size;
}


