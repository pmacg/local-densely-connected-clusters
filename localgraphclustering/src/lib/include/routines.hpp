#ifndef ROUTINES_HPP
#define ROUTINES_HPP

#include <unordered_map>
#include "ppr_path.hpp"
#include <vector>
#include "sparseheap.hpp" // include our heap functions
#include "sparserank.hpp" // include our sorted-list functions
#include "sparsevec.hpp" // include our sparse hashtable functions
#include <tuple>

using namespace std;

// vtype and itype are the types used to represent vertices and edges respectively.
// additionally, they are used to represent the number of vertices and so they must be some numerical type.
template<typename vtype,typename itype>
struct Edge
{
    vtype v ; 
    double flow ; 
    double C; 
    itype rev; 
};

template <class T>
inline void hash_combine(std::size_t& seed, const T& v)
{
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed<<6) + (seed>>2);
}

// Only for pairs of std::hash-able types for simplicity.
// You can of course template this struct to allow other hash functions
struct pair_hash {
    template <class T1, class T2>
    std::size_t operator () (const std::pair<T1,T2> &p) const {
        size_t seed = 0;
        hash_combine(seed,p.first);
        hash_combine(seed,p.second);
        
        //cout << p.first << " " << p.second << " " << seed << endl;
        
        // Mainly for demonstration purposes, i.e. works but is overly simple
        // In the real world, use sth. like boost.hash_combine
        return seed;
    }
};

/**
 * This class represents a (weighted) graph.
 *
 * @tparam vtype - the type used for entries in the row indices for the compressed row format adjacency matrix.
 * @tparam itype - the type used for entries in the column indices for the compressed row format adjacency matrix.
 */
template<typename vtype, typename itype>
class graph{
    itype m; // number of edges (for an undirected graph, edges are counted twice - once in each direction)
    vtype n; // number of vertices

    // In the language of scipy.csr_sparse:
    //   - ai is the indptr array
    //   - aj is the indices array
    //   - a is the data array
    // Together, these make up a Compresses Sparse Row representation of the adjacency matrix of the graph.
    // It might be useful to refer to:
    //   - https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.sparse.csr_matrix.html
    //   - https://stackoverflow.com/questions/52299420/scipy-csr-matrix-understand-indptr
    // Though obviously here, we are working in C++, not python!
    itype* ai;
    vtype* aj;
    double* a;

    vtype offset; // offset for zero based arrays (matlab) or one based arrays (julia)
    double* degrees; // weighted degrees of vertices
    double volume; // the volume of graph, 2m for an unweighted undirected graph
public:
    // Constructors
    /**
     * Basic graph constructor.
     */
    graph<vtype,itype>(itype, vtype, itype*, vtype*, double*, vtype, double*);
    
    // Common functions - implemented in this header file.
    /**
     * Return the weighted degree of a given vertex.
     * @param id - the index of the vertex of interest.
     * @return the weighted degree
     */
    double get_degree_weighted(vtype id);

    /**
     * Return the unweighted degree of a given vertex.
     * @param id - the index of the vertex of interest
     * @return the unweighted degree
     */
    vtype get_degree_unweighted(vtype id);

    /**
     * Compute the unweighted volume and cut value for a set of vertices defined by the map R_map.
     * @param R_map - a map with keys corresponding to vertices (corrected for the offset) and unique values
     *                corresponding just to the count of vertices in the set.
     * @param nR - the number of vertices in the set.
     * @return a pair containing the set volume and cut size. (unweighted)
     */
    pair<itype, itype> get_stats(unordered_map<vtype, vtype>& R_map, vtype nR);

    /**
     * Compute the weighted volume and cut value for a set of vertices.
     * @param R_map - a map with keys corresponding to vertices in set target set. (corrected for the offset)
     * @param nR - the number of vertices in the set.
     * @return a pair containing the set volume and cut size
     */
    pair<double, double> get_stats_weighted(unordered_map<vtype, vtype>& R_map, vtype nR);

    /**
     * Returns the unweighted 'relative volume' of S and R, along with the normal unweighted cut size of S.
     * Letting X = (S intersection R) and Y = (S \ R), the relative volume is given by
     *   vol_rel(S, R) = vol(Y) - delta * vol(X)
     * @param S_map - a map with keys corresponding to vertices in target set S (corrected for offset)
     * @param R_map - a map with keys corresponding to vertices in set R (corrected for offset)
     * @param nR - the number of vertices in R (unused)
     * @param delta - the delta to be used in the relative volume calculation.
     * @return a pair containing the relative volume and cut size
     */
    pair<itype, itype> get_stats_rel(unordered_map<vtype, vtype>& S_map, unordered_map<vtype, vtype>& R_map, vtype nR,
                                     double delta);

    /**
     * Returns the weighted 'relative volume' of S and R, along with the normal weighted cut size of S.
     * Letting X = (S intersection R) and Y = (S \ R), the relative volume is given by
     *   vol_rel(S, R) = vol(Y) - delta * vol(X)
     * @param S_map - a map with keys corresponding to vertices in target set S (corrected for offset)
     * @param R_map - a map with keys corresponding to vertices in set R (corrected for offset)
     * @param nR - the number of vertices in R (unused)
     * @param delta - the delta to be used in the relative volume calculation.
     * @return a pair containing the relative volume and cut size
     */
    pair<double, double> get_stats_rel_weighted(unordered_map<vtype, vtype>& S_map, unordered_map<vtype, vtype>& R_map,
                                                vtype nR, double delta);

    // Methods used for MaxFlow computations
    /**
     * Add an edge with capacity C to the list of edges maintained in adj.
     * @param u - the first vertex in the edge
     * @param v - the second vertex in the edge
     * @param C - the capacity of the edge
     */
    void addEdge(vtype u, vtype v, double C);

    /**
     * Check whether more flow can be sent from the source s to the sink t using a BFS.
     * Also assigns levels to nodes using the level variable.
     * @param s - the source vertex
     * @param t - the sink (target) vertex
     * @param V - the number of vertices
     * @return
     */
    bool BFS(vtype s, vtype t, vtype V);

    double sendFlow(vtype u, double flow, vtype t, vector<vtype>& start, vector<pair<int,double>>& SnapShots);
    pair<double,vtype> DinicMaxflow(vtype s, vtype t, vtype V, vector<bool>& mincut);
    void find_cut(vtype u, vector<bool>& mincut, vtype& length);

    // Data used for MaxFlow computations
    int* level;
    vector< Edge<vtype,itype> > *adj;
    
    // Functions in aclpagerank.cpp
    /**
     * Computes the personalised pagerank vector as required by the ACL algorithm.
     *
     * @param alpha - number in [0, 1]. Values closer to 1 give a smaller cluster.
     * @param eps - accuracy of approximate pagerank. Smaller values slower, but more accurate.
     * @param seedids - the starting vertices for the personalised pagerank
     * @param nseedids - the number of starting vertices
     * @param maxsteps - the maximum number of steps when compute the personalised pagerank.
     * @param xids - will be populated with a vector giving the vertices with non-zero pagerank value.
     * @param xlength - the maximum number of vertices allowed in xids.
     * @param values - the actual pagerank vector itself. Values are given only for vertices in xids.
     * @return
     */
    vtype aclpagerank(double alpha, double eps, vtype* seedids, vtype nseedids,
                      vtype maxsteps, vtype* xids, vtype xlength, double* values);
    /**
     * The internal algorithm computing the personalised pagerank for ACL. External callers should use aclpagerank
     * instead.
     */
    vtype pprgrow(double alpha, double eps, vtype* seedids, vtype nseedids,
                  vtype maxsteps, vtype* xids, vtype xlength, double* values);


    // Functions in aclpagerank_weighted.cpp
    /**
     * Computes the weighted personalised pagerank vector as required by the ACL algorithm.
     *
     * @param alpha - number in [0, 1]. Values closer to 1 give a smaller cluster.
     * @param eps - accuracy of approximate pagerank. Smaller values slower, but more accurate.
     * @param seedids - the starting vertices for the personalised pagerank
     * @param nseedids - the number of starting vertices
     * @param maxsteps - the maximum number of steps when compute the personalised pagerank.
     * @param xids - will be populated with a vector giving the vertices with non-zero pagerank value.
     * @param xlength - the maximum number of vertices allowed in xids.
     * @param values - the actual pagerank vector itself. Values are given only for vertices in xids.
     * @return
     */
    vtype aclpagerank_weighted(double alpha, double eps, vtype* seedids, vtype nseedids,
                      vtype maxsteps, vtype* xids, vtype xlength, double* values);

    /**
     * The internal algorithm computing the weighted personalised pagerank for ACL. External callers should use
     * aclpagerank_weighted instead.
     */
    vtype pprgrow_weighted(double alpha, double eps, vtype* seedids, vtype nseedids,
                  vtype maxsteps, vtype* xids, vtype xlength, double* values);

    // Functions in dcpagerank.cpp
    /**
     * Computes the personalised pagerank vector on the double cover of the graph.
     *
     * @param alpha - number in [0, 1]. Values closer to 1 give a smaller cluster.
     * @param eps - accuracy of approximate pagerank. Smaller values slower, but more accurate.
     * @param seedids - the starting vertices for the personalised pagerank
     * @param nseedids - the number of starting vertices
     * @param maxsteps - the maximum number of steps when compute the personalised pagerank.
     * @param xids_1, xids_2 - will be populated with a vector giving the vertices with non-zero pagerank value.
     * @param xlength - the maximum allowed length of an output vector
     * @param values_1, values_2 - the actual pagerank vector itself. Values are given only for vertices in xids.
     * @param simplify - whether to simplify the pagerank vector before returning it
     * @return
     */
    vtype dcpagerank(double alpha, double eps, vtype* seedids, vtype nseedids,
                      vtype maxsteps, vtype* xids_1, vtype* xids_2, vtype xlength, double* values_1, double* values_2,
                      bool simplify);

    /**
     * The internal algorithm computing the personalised pagerank on the double cover. External callers should use
     * dcpagerank instead.
     */
    vtype dc_pprgrow(double alpha, double eps, vtype* seedids, vtype nseedids,
                     vtype maxsteps, vtype* xids_1, vtype* xids_2, vtype xlength, double* values_1, double* values_2,
                     bool simplify);

    /**
     * Computes the personalised pagerank vector on the double cover of a weighted graph.
     *
     * @param alpha - number in [0, 1]. Values closer to 1 give a smaller cluster.
     * @param eps - accuracy of approximate pagerank. Smaller values slower, but more accurate.
     * @param seedids - the starting vertices for the personalised pagerank
     * @param nseedids - the number of starting vertices
     * @param maxsteps - the maximum number of steps when compute the personalised pagerank.
     * @param xids_1, xids_2 - will be populated with a vector giving the vertices with non-zero pagerank value.
     * @param xlength - the maximum number of vertices allowed in xids.
     * @param values - the actual pagerank vector itself. Values are given only for vertices in xids.
     * @param simplify - whether to simplify the pagerank vector before returning it
     * @return
     */
    vtype dcpagerank_weighted(double alpha, double eps, vtype* seedids, vtype nseedids,
                              vtype maxsteps, vtype* xids_1, vtype* xids_2, vtype xlength, double* values_1,
                              double* values_2, bool simplify);

    /**
     * The internal algorithm computing the weighted personalised pagerank on the double cover. External callers should
     * use dcpagerank instead.
     */
    vtype dc_pprgrow_weighted(double alpha, double eps, vtype* seedids, vtype nseedids,
                              vtype maxsteps, vtype* xids_1, vtype* xids_2, vtype xlength, double* values_1,
                              double* values_2, bool simplify);


    //functions in sweepcut.cpp
    vtype sweepcut_with_sorting(double* value, vtype* ids, vtype* results,
                                vtype num, double* ret_cond);
    vtype sweepcut_without_sorting(vtype* ids, vtype* results, vtype num,
                                   double* ret_cond);
    vtype sweep_cut(vtype* ids, vtype* results, vtype num, double* ret_cond);


    //functions in ppr_path.hpp
    vtype ppr_path(double alpha, double eps, double rho, vtype* seedids, vtype nseedids, vtype* xids,
                   vtype xlength, struct path_info ret_path_results, struct rank_info ret_rank_results);
    void hypercluster_graphdiff_multiple(const vector<vtype>& set, double t, double eps, double rho,
                                         eps_info<vtype>& ep_stats, rank_record<vtype>& rkrecord, vector<vtype>& cluster);
    void graphdiffseed(sparsevec& set, const double t, const double eps_min, const double rho, const vtype max_push_count,
                       eps_info<vtype>& ep_stats, rank_record<vtype>& rkrecord, vector<vtype>& cluster);
    bool resweep(vtype r_end, vtype r_start, sparse_max_rank<vtype,double,size_t>& rankinfo, sweep_info<vtype>& swinfo);
    vtype rank_permute(vector<vtype> &cluster, vtype r_end, vtype r_start);
    void copy_array_to_index_vector(const vtype* v, vector<vtype>& vec, vtype num);


    //functions in MQI.cpp
    vtype MQI(vtype nR, vtype* R, vtype* ret_set);
    void build_map(unordered_map<vtype, vtype>& R_map,unordered_map<vtype, vtype>& degree_map,
                   vtype* R, vtype nR);
    void build_list(unordered_map<vtype, vtype>& R_map, unordered_map<vtype, vtype>& degree_map, vtype src, vtype dest, itype a, itype c);

    
    //functions in MQI_weighted.cpp
    vtype MQI_weighted(vtype nR, vtype* R, vtype* ret_set);
    void build_map_weighted(unordered_map<vtype, vtype>& R_map,unordered_map<vtype, double>& degree_map,
                   vtype* R, vtype nR, double* degrees);
    void build_list_weighted(unordered_map<vtype, vtype>& R_map, unordered_map<vtype, double>& degree_map, vtype src, vtype dest, 
                   double a, double c, double* degrees);


    //functions in proxl1PRaccel.cpp
    vtype proxl1PRaccel(double alpha, double rho, vtype* v, vtype v_nums, double* d,
                        double* ds, double* dsinv, double epsilon, double* grad, double* p, double* y,
                        vtype maxiter,double max_time, bool use_distribution, double* distribution);
                        
    // functions in proxl1PRrand.cpp
    vtype proxl1PRrand(vtype num_nodes, vtype* seed, vtype num_seeds, double epsilon, double alpha, double rho, double* q, double* d, double* ds, double* dsinv, vtype maxiter, vtype* candidates);
//     // functions in proxl1PRrand.cpp
//     vtype proxl1PRrand_unnormalized(vtype num_nodes, vtype* seed, vtype num_seeds, double epsilon, double alpha, double rho, double* q, double* d, double* ds, double* dsinv, double* grad, vtype maxiter);
//     //functions in proxl1PRaccel.cpp
    vtype proxl1PRaccel_unnormalized(double alpha, double rho, vtype* v, vtype v_nums, double* d,
                        double* ds, double* dsinv, double epsilon, double* grad, double* p, double* y,
                        vtype maxiter, double max_time, bool use_distribution, double* distribution);


    //functions in densest_subgraph.cpp
    double densest_subgraph(vtype *ret_set, vtype *actual_length);
    void build_list_DS(double g, vtype src, vtype dest);


    //functions in SimpleLocal.cpp
    void STAGEFLOW(double delta, double alpha, double beta, unordered_map<vtype,vtype>& fullyvisited, unordered_map<vtype,vtype>& R_map, unordered_map<vtype,vtype>& S);
    vtype SimpleLocal(vtype nR, vtype* R, vtype* ret_set, double delta, bool relcondflag);
    void init_VL(unordered_map<vtype,vtype>& VL, unordered_map<vtype,vtype>& VL_rev,unordered_map<vtype,vtype>& R_map);
    void init_EL(vector< tuple<vtype,vtype,double> >& EL, unordered_map<vtype,vtype>& R_map, unordered_map<vtype,vtype>& VL, vtype s, vtype t, double alpha, double beta);
    void update_VL(unordered_map<vtype,vtype>& VL, unordered_map<vtype,vtype>& VL_rev, vector<vtype>& E);
    void update_EL(vector< tuple<vtype,vtype,double> >& EL, unordered_map<vtype,vtype>& VL, unordered_map<vtype,vtype>& R_map, unordered_map<vtype,vtype>& W_map,
                   vtype s, vtype t, double alpha, double beta);
    void assemble_graph(vector<bool>& mincut, vtype nverts, itype nedges, vector<tuple<vtype,vtype,double>>& EL);

    //functions in SimpleLocal_weighted.cpp
    void STAGEFLOW_weighted(double delta, double alpha, double beta, unordered_map<vtype,vtype>& fullyvisited, unordered_map<vtype,vtype>& R_map, unordered_map<vtype,vtype>& S);
    vtype SimpleLocal_weighted(vtype nR, vtype* R, vtype* ret_set, double delta, bool relcondflag);
    void init_VL_weighted(unordered_map<vtype,vtype>& VL, unordered_map<vtype,vtype>& VL_rev,unordered_map<vtype,vtype>& R_map);
    void init_EL_weighted(vector< tuple<vtype,vtype,double> >& EL, unordered_map<vtype,vtype>& R_map, unordered_map<vtype,vtype>& VL, vtype s, vtype t, double alpha, double beta);
    void update_VL_weighted(unordered_map<vtype,vtype>& VL, unordered_map<vtype,vtype>& VL_rev, vector<vtype>& E);
    void update_EL_weighted(vector< tuple<vtype,vtype,double> >& EL, unordered_map<vtype,vtype>& VL, unordered_map<vtype,vtype>& R_map, unordered_map<vtype,vtype>& W_map,
                   vtype s, vtype t, double alpha, double beta);
    void assemble_graph_weighted(vector<bool>& mincut, vtype nverts, itype nedges, vector<tuple<vtype,vtype,double>>& EL);
    

    //functions for capacity releasing diffusion
    vtype capacity_releasing_diffusion(vector<vtype>& ref_node, vtype U,vtype h,vtype w,vtype iterations,vtype* cut);
    void unit_flow(unordered_map<vtype,double>& Delta, vtype U, vtype h, vtype w, unordered_map<vtype,double>& f_v, 
        unordered_map<vtype,double>& ex, unordered_map<vtype,vtype>& l);
    void round_unit_flow(unordered_map<vtype,vtype>& l, unordered_map<vtype,double>& cond,unordered_map<vtype,vector<vtype>>& labels);

    //functions in triangleclusters.cpp
    void triangleclusters(double* cond, double* cut, double* vol, double* cc, double* t);
};

template<typename vtype, typename itype>
graph<vtype,itype>::graph(itype _m, vtype _n, itype* _ai, vtype* _aj, double* _a,
                          vtype _offset, double* _degrees)
{
    m = _m;
    n = _n;
    ai = _ai;
    aj = _aj;
    a = _a;
    offset = _offset;
    degrees = _degrees;
    volume = 0;
    if (ai != NULL) {
        volume = (double)ai[n];
    }
    adj = NULL;
    level = NULL;

}

template<typename vtype, typename itype>
void graph<vtype,itype>::addEdge(vtype u, vtype v, double C)
{
    // Forward edge : 0 flow and C capacity
    Edge<vtype,itype> p{v, 0, C, (itype)adj[v].size()};
 
    // Back edge : 0 flow and C capacity
    Edge<vtype,itype> q{u, 0, C, (itype)adj[u].size()};
 
    adj[u].push_back(p);
    adj[v].push_back(q); // reverse edge
}

template<typename vtype,typename itype>
double graph<vtype,itype>::get_degree_weighted(vtype id)
{
    double d = 0;

    // Loop through the neighbors of the given vertex, and add the weights of the edges.
    // In the adjacency matrix data matrix, a, the entries for the neighbors of j are from index ai[j] to (ai[j + 1]-1)
    // (and the offset is either 0 or 1 to correct for when the wrapping code uses 1-indexed arrays).
    for(vtype j = ai[id] - offset; j < ai[id+1] - offset; j ++){
        d += a[j];
    }
    return d;
}

template<typename vtype,typename itype>
vtype graph<vtype,itype>::get_degree_unweighted(vtype id)
{
    // The unweighted degree is simply the number of entries in the sparse matrix representation corresponding to this
    // vertex. If it is not clear to you why this line works, see the comment above the declaration of ai, aj and a in
    // the graph class declaration, along with this stackoverflow answer:
    //  - https://stackoverflow.com/questions/52299420/scipy-csr-matrix-understand-indptr
    return ai[id + 1] - ai[id];
}

template<typename vtype, typename itype>
pair<itype, itype> graph<vtype,itype>::get_stats(unordered_map<vtype, vtype>& R_map, vtype nR)
{
    itype curvol = 0;
    itype curcutsize = 0;

    // Iterate through the vertices in the set (stored as keys in R_map)
    for(auto R_iter = R_map.begin(); R_iter != R_map.end(); ++ R_iter){
        vtype v = R_iter->first;
        itype deg = get_degree_unweighted(v);
        curvol += deg;

        // Iterate through the neighbours of v
        // j is the index of the neighbour of v in the data array a. aj[j] gives the neighboring vertex.
        for(itype j = ai[v] - offset; j < ai[v + 1] - offset; j ++){
            if(R_map.count(aj[j] - offset) == 0){
                curcutsize ++;
            }
        }
    }
    
    pair<itype, itype> set_stats (curvol, curcutsize);
    return set_stats;
}

template<typename vtype, typename itype>
pair<double, double> graph<vtype,itype>::get_stats_weighted(unordered_map<vtype, vtype>& R_map, vtype nR)
{
    double curvol = 0;
    double curcutsize = 0;

    // Iterate through the vertices in the set (stored as keys in R_map)
    for(auto R_iter = R_map.begin(); R_iter != R_map.end(); ++ R_iter){
        vtype v = R_iter->first;
        double deg = degrees[v];
        curvol += deg;

        // Iterate through the neighbours of v
        // j is the index of the neighbour of v in the data array a. aj[j] gives the neighboring vertex.
        for(itype j = ai[v] - offset; j < ai[v + 1] - offset; j ++){
            if(R_map.count(aj[j] - offset) == 0){
                // Add the entry in the adjacency matrix corresponding to this edge
                // This ensures that the cut size is weighted correctly
                curcutsize += a[j];
            }
        }
    }
    
    pair<double, double> set_stats (curvol, curcutsize);
    return set_stats;
}

template<typename vtype, typename itype>
pair<itype, itype> graph<vtype,itype>::get_stats_rel(unordered_map<vtype, vtype>& S_map, unordered_map<vtype,
                                                     vtype>& R_map, vtype nR, double delta)
{
    itype curvol = 0;
    itype curcutsize = 0;
    itype deg;

    // Iterate through the vertices in the set S
    for(auto S_iter = S_map.begin(); S_iter != S_map.end(); ++ S_iter){
        vtype v = S_iter->first;
        if (R_map.count(v) != 0) {
            // If the vertex is also in R, add the degree to the volume
            deg = get_degree_unweighted(v);
            curvol += deg;
        }
        else {
            // If the vertex is not in R, subtract delta times the degree
            deg = get_degree_unweighted(v);
            curvol -= delta*deg;
        }

        // Iterate through neighbours of vertex v, and increment the cut size if the neighbour is not in S.
        for(itype j = ai[v] - offset; j < ai[v + 1] - offset; j ++){
            if(S_map.count(aj[j] - offset) == 0){
                curcutsize ++;
            }
        }
    }
    
    pair<itype, itype> set_stats (curvol, curcutsize);
    return set_stats;
}


template<typename vtype, typename itype>
pair<double, double> graph<vtype,itype>::get_stats_rel_weighted(unordered_map<vtype, vtype>& S_map, unordered_map<vtype, vtype>& R_map, vtype nR, double delta)
{
    double curvol = 0;
    double curcutsize = 0;
    double deg;

    // Iterate through the vertices in S
    for(auto S_iter = S_map.begin(); S_iter != S_map.end(); ++ S_iter){
        vtype v = S_iter->first;
        if (R_map.count(v) != 0) {
            // If the vertex is not in R, add the degree to the volume
            deg = degrees[v];
            curvol += deg;
        }
        else {
            // If the vertex is also in R, subtract delta times the degree to the volume
            deg = degrees[v];
            curvol -= delta*deg;
        }

        // Iterate through neighbours of v, adding the weight of the edge to the cut size if the neighbour is not also
        // in S.
        for(itype j = ai[v] - offset; j < ai[v + 1] - offset; j ++){
            if(S_map.count(aj[j] - offset) == 0){
                curcutsize += a[j];
            }
        }
    }
    
    pair<double, double> set_stats (curvol, curcutsize);
    return set_stats;
}

#include "maxflow.cpp"

#endif