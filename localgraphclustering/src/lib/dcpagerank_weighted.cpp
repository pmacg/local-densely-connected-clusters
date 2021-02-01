/**
 * Weighted dcpagerank with C interface. It takes a weighted and undirected graph with CSR representation
 * and some seed vetrices as input and computes the approximate pagerank vector on the double cover of the graph.
 * Choose the appropriate C interface for the data type of your input.
 *
 * INPUT:
 *     n        - the number of vertices in the graph
 *     ai,aj,a  - Compressed sparse row representation
 *     offset   - offset for zero based arrays (matlab) or one based arrays (julia)
 *     alpha    - value of alpha
 *     eps      - value of epsilon
 *     seedids  - the set of indices for seeds
 *     nseedids - the number of indices in the seeds
 *     maxsteps - the max number of steps
 *     xids_1   - the first solution vector, i.e. the vertices with nonzero pagerank value
 *     xids_2   - the second solution vector, i.e. the vertices with nonzero pagerank value
 *     xlength  - the maximum length of the vector we can output
 *     values_1 - the first pagerank value vector on the double cover for xids
 *     values_2 - the second pagerank value vector on the double cover for xids
 *
 * OUTPUT:
 *     actual_length - the number of nonzero entries in the longer solution vector, before limiting by xlength.
 *                     the calling code needs to check that this is not longer than xlength, otherwise, they should
 *                     allocate more memory for the ouput and call the function again.
 *
 * COMPILE:
 *     make dcpagerank_weighted
 *
 */

#include <cstdio>
#include <cstdlib>
#include <unordered_map>
#include <queue>
#include <iostream>
#include <algorithm>
#include <cstdint>

#include "include/dcpagerank_weighted_c_interface.h"
#include "include/routines.hpp"

using namespace std;

template<typename vtype>
bool compare(pair<vtype, double> i, pair<vtype, double> j) {
    return (i.second > j.second);
}

uint32_t dcpagerank_weighted32(
        uint32_t n, uint32_t* ai, uint32_t* aj, double* a, uint32_t offset,
        double alpha,
        double eps, 
        uint32_t* seedids, uint32_t nseedids,
        uint32_t maxsteps,
        uint32_t* xids_1,
        uint32_t* xids_2,
        uint32_t xlength,
        double* values_1, double* values_2,
        bool simplify)
{
    // Construct a graph object from the sparse adjacency matrix.
    graph<uint32_t,uint32_t> g(ai[n],n,ai,aj,a, offset,nullptr);

    // Call the pagerank method on the graph to compute the pagerank itself and return the result.
    uint32_t actual_length = g.dcpagerank_weighted(alpha, eps, seedids, nseedids, maxsteps, xids_1, xids_2, xlength,
                                                   values_1, values_2, simplify);
    return actual_length;
}

int64_t dcpagerank_weighted64(
        int64_t n, int64_t* ai, int64_t* aj, double* a, int64_t offset,
        double alpha, 
        double eps, 
        int64_t* seedids, int64_t nseedids,
        int64_t maxsteps,
        int64_t* xids_1,
        int64_t* xids_2,
        int64_t xlength,
        double* values_1, double* values_2,
        bool simplify)
{
    // Construct a graph object from the sparse adjacency matrix.
    graph<int64_t,int64_t> g(ai[n],n,ai,aj,a,offset,nullptr);

    // Call the pagerank method on the graph to compute the pagerank itself and return the result.
    int64_t actual_length = g.dcpagerank_weighted(alpha, eps, seedids, nseedids, maxsteps, xids_1, xids_2, xlength,
                                                  values_1, values_2, simplify);
    return actual_length;
}

uint32_t dcpagerank_weighted32_64(
        uint32_t n, int64_t* ai, uint32_t* aj, double* a, uint32_t offset,
        double alpha, 
        double eps, 
        uint32_t* seedids, uint32_t nseedids,
        uint32_t maxsteps,
        uint32_t* xids_1,
        uint32_t* xids_2,
        uint32_t xlength, double* values_1, double* values_2,
        bool simplify)
{
    // Construct a graph object from the sparse adjacency matrix.
    graph<uint32_t,int64_t> g(ai[n],n,ai,aj,a,offset,nullptr);

    // Call the pagerank method on the graph to compute the pagerank itself and return the result.
    uint32_t actual_length = g.dcpagerank_weighted(alpha, eps, seedids, nseedids, maxsteps, xids_1, xids_2, xlength,
                                                   values_1, values_2, simplify);
    return actual_length;
}

template<typename vtype, typename itype>
vtype graph<vtype,itype>::dcpagerank_weighted(double alpha, double eps, vtype* seedids, vtype nseedids,
                                              vtype maxsteps, vtype* xids_1, vtype* xids_2, vtype xlength,
                                              double* values_1, double* values_2, bool simplify)
{
    // Call the internal dc_pprgrow method to compute the pagerank.
    vtype actual_length;
    actual_length = dc_pprgrow_weighted(
            alpha, eps, seedids, nseedids, maxsteps, xids_1, xids_2, xlength, values_1, values_2, simplify);
    return actual_length;
}

/** 
 * dc_pprgrow_weighted compute the approximate pagerank vector locally on the double cover of the weighted graph.
 *
 * INUPUT:
 *     rows     - a self defined struct which contains all the info of a CSR based graph
 *     alpha    - value of alpha
 *     eps      - value of epsilon
 *     seedids  - the set of indices for seeds
 *     nseedids - the number of indices in the seeds
 *     maxsteps - the max number of steps
 *     xlength  - the max number of ids in the solution vector
 *     xids_1   - the first solution vector, i.e. the vertices with nonzero pagerank value
 *     xids_2   - the second solution vector, i.e. the vertices with nonzero pagerank value
 *     xlength  - the maximum possible length of an ouput vector
 *     values_1 - the first vector of pagerank values on the double cover for xids
 *     values_2 - the second vector of pagerank values on the double cover for xids
 *     simplify - whether to apply the simplify operator before returning the result
 *
 * OUTPUT:
 *     actual_length - the number of nonzero entries in the solution vector, before limiting by xlength.
 */
template<typename vtype, typename itype>
vtype graph<vtype,itype>::dc_pprgrow_weighted(double alpha, double eps,vtype* seedids, vtype nseedids,
                                              vtype maxsteps, vtype* xids_1, vtype* xids_2, vtype xlength,
                                              double* values_1, double* values_2, bool simplify)
{
#ifdef DEBUG
    cout << endl << "Entering dc_ppgrow_weighted." << endl;
#endif
    // The x_1_map and x_2_map variables will store the approximate personalised pagerank vector on the double cover
    // at the end of the algorithm.
    // The r_1_map and r_2_map variables store the 'residuals' throughout the algorithm. The algorithm guarantees that
    // the residual for some vertex v will be less than (eps * deg(v)).
    // Both of these are stored as sparse arrays represented by unordered_maps.
#ifdef DEBUG
    cout << "Declaring maps and iterators." << endl;
#endif
    unordered_map<vtype, double> x_1_map, x_2_map, r_1_map, r_2_map;
    unordered_map<vtype, double> *this_x_map, *this_r_map, *that_r_map;
    typename unordered_map<vtype, double>::const_iterator r_iter, x_iter;

    // This queue will store all vertices whose value on the residual vector r is greater than the epsilon error
    // parameter. Stored as a pair - the first element indicates the vertex and the second element indicates whether
    // it refers to the first or second set of vertices in the double cover.
#ifdef DEBUG
    cout << "Declaring queue." << endl;
#endif
    queue<pair<vtype, uint8_t>> Q;

    // Initialise the vectors x and r according to the seed nodes. At the beginning of the algorithm, all of the
    // 'weight' is on the seed nodes in the first residual vector.
#ifdef DEBUG
    cout << "Innitialising maps with seed nodes." << endl;
#endif
    for(size_t i = 0; i < (size_t)nseedids; i ++){
        r_1_map[seedids[i] - offset] = 1;
    }

    // Initialise the queue. A vertex is included in the queue if its residual value is greater than (eps * deg)
#ifdef DEBUG
    cout << "Initialising queue." << endl;
#endif
    pair<vtype, uint8_t> p;
    for(r_iter = r_1_map.begin(); r_iter != r_1_map.end(); ++r_iter){
        if(r_iter->second >= eps * get_degree_weighted(r_iter->first)){
            p = make_pair(r_iter->first, 1);
            Q.push(p);
        }
    }

    // Begin the main loop of the algorithm.
    // j will be the index of the vertex we are performing the current 'push' operation on.
    // dcset will be 1 or 2 depending on which set of double cover vertices we are interested in.
    vtype j;
    uint8_t dcset;
    double xj, rj, delta;
    vtype steps_count = 0;
    while(Q.size() > 0 && steps_count < maxsteps){
#ifdef DEBUG
        cout << endl << "Iteration " << (steps_count + 1) << endl;
        cout << "There are " << Q.size() << " items in the queue." << endl;
#endif
        // Pull a vertex off the front of the queue
#ifdef DEBUG
        cout << "Pop from the front of the queue." << endl;
#endif
        p = Q.front();
        j = p.first;
        dcset = p.second;
        Q.pop();
#ifdef DEBUG
        cout << "Operating on vertex " << j << "_" << unsigned(dcset) << endl;
#endif

        // Set the 'this' and 'that' pointers to point to the appropriate maps given which of the double cover
        // sets we are operating on.
#ifdef DEBUG
        cout << "Specify this and that." << endl;
#endif
        if (dcset == 1){
            this_x_map = &x_1_map;
            this_r_map = &r_1_map;
            that_r_map = &r_2_map;
        } else {
            this_x_map = &x_2_map;
            this_r_map = &r_2_map;
            that_r_map = &r_1_map;
        }
        x_iter = this_x_map->find(j);
        r_iter = this_r_map->find(j);

        // Get the r_i[j] value
        rj = r_iter->second;
#ifdef DEBUG
        cout << "rj = " << rj << endl;
#endif

        // Update x_i[j] - line 2 of dcpush algorithm
        if(x_iter == this_x_map->end()){
            // If the vertex j is not already in the sparse x matrix, then add it, with the value (alpha * rj)
            xj = alpha * rj;
            this_x_map->insert({j, xj});
        }
        else{
            // If the vertex is already in x, then add (alpha * rj)
            xj = x_iter->second + alpha*rj;
            this_x_map->at(j) = xj;
        }

        // Update r_i[j] - line 3 of dcpush algorithm
        this_r_map->at(j) = ((1-alpha)/2)*rj;
#ifdef DEBUG
        cout << "Set r_j = " << this_r_map->at(j) << endl;
#endif

        // Check whether r_i[j] still exceeds the threshold (eps * deg(j)) and so still needs to be on the queue to be
        // pushed again.
        if (this_r_map->at(j) >= eps * get_degree_weighted(j)) {
#ifdef DEBUG
            cout << "r_j still exceeds threshold" << endl;
#endif
            Q.push(p);
        }

        // Update x_(3-i)[u] for u in neighbours(j) - line 5 of dcpush algorithm
        // delta is the amount that each x_(3-i)[u] should be increased
        delta = ((1-alpha)/2)*(rj/get_degree_weighted(j));
#ifdef DEBUG
        cout << "delta = " << delta << endl;
#endif
        vtype u;
        double ru_new, ru_old;
        for(itype i = ai[j] - offset; i < ai[j+1] - offset; i++){
            // u is a neighbour of j
            u = aj[i] - offset;
#ifdef DEBUG
            cout << "Pushing to neighbour u = " << u << endl;
#endif

            // Find the vertex u in r if it has a value so far
            r_iter = that_r_map->find(u);
            if(r_iter == that_r_map->end()){
                // If u was not already present in r, then add it with the value delta.
                ru_old = 0;
                ru_new = a[i] * delta;
                that_r_map->insert({u, ru_new});
            }
            else{
                // If us was already present in r, increase its value by delta
                ru_old = r_iter->second;
                ru_new = a[i] * delta + ru_old;
                that_r_map->at(u) = ru_new;
            }

            // If r[u] exceeds the error threshold (eps * deg(u)) and it didn't previously, then add it to the queue
            // of vertices to be processed.
            if(ru_new > eps * get_degree_weighted(u) && ru_old <= eps * get_degree_weighted(u)){
                p = make_pair(u, 3 - dcset);
                Q.push(p);
            }
        }

        // Make sure to count the number of steps so we stop if we hit the threshold.
        steps_count ++;
    }

    // If we are to simplify the output, do this now before applying the maximum size of the output
    if (simplify){
        for (auto iter_1 = x_1_map.begin(), next_iter_1 = iter_1; iter_1 != x_1_map.end(); iter_1 = next_iter_1){
            // Increment the iterator. We write the for loop like this since we are modifying the x_1_map as we go.
            // See e.g. https://stackoverflow.com/questions/8234779/how-to-remove-from-a-map-while-iterating-it
            ++next_iter_1;

            // Check whether there is a corresponding value in x_2_map
            vtype this_vertex = iter_1->first;

            auto iter_2 = x_2_map.find(this_vertex);
            if (iter_2 != x_2_map.end()){
                // This vertex is present in both maps
                // Perform the simplification
                if (iter_1->second > iter_2->second){
                    x_1_map[this_vertex] = iter_1->second - iter_2->second;
                    x_2_map.erase(this_vertex);
                }
                else {
                    x_2_map[this_vertex] = iter_2->second - iter_1->second;
                    x_1_map.erase(this_vertex);
                }
            }
        }
    }

    // The remainder of this method is simply choosing which vertices to return in the final xids vector since the
    // caller may have specified a limit on the number of vertices in the xlength parameter.
    // map_size is the number of non-zero elements in the approximate pagerank vector.
    vtype map_1_size = x_1_map.size();
    vtype map_2_size = x_2_map.size();

    // Create an array object possible_nodes from the unordered_maps x_1_map and x_2_map
    // They contain the same data, just as different structures.
    auto* possible_nodes_1 = new pair<vtype, double>[map_1_size];
    auto* possible_nodes_2 = new pair<vtype, double>[map_2_size];
    int i = 0;
    for(x_iter = x_1_map.begin(); x_iter != x_1_map.end(); ++x_iter){
        possible_nodes_1[i].first = x_iter->first;
        possible_nodes_1[i].second = x_iter->second;
        i++;
    }
    i = 0;
    for(x_iter = x_2_map.begin(); x_iter != x_2_map.end(); ++x_iter){
        possible_nodes_2[i].first = x_iter->first;
        possible_nodes_2[i].second = x_iter->second;
        i++;
    }

    // Add the vertices with the largest pagerank values to the output vectors.
    sort(possible_nodes_1, possible_nodes_1 + map_1_size, compare<vtype>);
    sort(possible_nodes_2, possible_nodes_2 + map_2_size, compare<vtype>);
    for(j = 0; j < min(map_1_size, xlength); j++){
        xids_1[j] = possible_nodes_1[j].first + offset;
        values_1[j] = possible_nodes_1[j].second;

        // If x_1 is longer than x_2, then set the extra x_2 parameters to 0
        if (j >= map_2_size){
            values_2[j] = 0;
        }
    }
    for(j = 0; j < min(map_2_size, xlength); j++){
        xids_2[j] = possible_nodes_2[j].first + offset;
        values_2[j] = possible_nodes_2[j].second;

        // If x_2 is longer than x_1, then set the extra x_1 parameters to 0
        if (j >= map_1_size){
            values_1[j] = 0;
        }
    }
    delete [] possible_nodes_1;
    delete [] possible_nodes_2;

    // Return the number of non-zero entries in the original solution vector.
    return max(map_1_size, map_2_size);
}


