#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif
    /**
     * A C interface for aclpagerank.
     * @param n - the number of vertices in the graph
     * @param ai - the indptr array for the compressed sparse row adjacency matrix
     * @param aj - the index array for the compressed sparse row adjacency matrix
     * @param offset - 0 or 1 depending on whether the calling code uses 0 or 1 based indexing
     * @param alpha - the alpha parameter for the approximate pagerank calculation
     * @param eps - the epsilon parameter for the approximate pagerank calculation
     * @param seedids - an array of vertex indices for seeding the personalised pagerank
     * @param nseedids - the number of vertices in seedids
     * @param maxsteps - the maximum number of steps to take when computeing the personalised pagerank
     * @param xids - output vector containing the ids of vertices with non-zero pagerank value
     * @param xlength - the maximum allowed length of xids
     * @param values - output vector with the pagerank vector values corresponding to the vertices in xids
     * @return the actual length of xids
     */
    uint32_t aclpagerank32(
        uint32_t n, uint32_t* ai, uint32_t* aj, uint32_t offset,
            //Compressed sparse row representation, with offset for
            //zero based (matlab) or one based arrays (julia)
        double alpha,    //value of alpha
        double eps,    //value of epsilon
        uint32_t* seedids, uint32_t nseedids,    //the set indices for seeds
        uint32_t maxsteps,    //the maximum number of steps
        uint32_t* xids, uint32_t xlength, double* values); //solution vectors


    /**
     * A C interface for aclpagerank.
     * @param n - the number of vertices in the graph
     * @param ai - the indptr array for the compressed sparse row adjacency matrix
     * @param aj - the index array for the compressed sparse row adjacency matrix
     * @param offset - 0 or 1 depending on whether the calling code uses 0 or 1 based indexing
     * @param alpha - the alpha parameter for the approximate pagerank calculation
     * @param eps - the epsilon parameter for the approximate pagerank calculation
     * @param seedids - an array of vertex indices for seeding the personalised pagerank
     * @param nseedids - the number of vertices in seedids
     * @param maxsteps - the maximum number of steps to take when computeing the personalised pagerank
     * @param xids - output vector containing the ids of vertices with non-zero pagerank value
     * @param xlength - the maximum allowed length of xids
     * @param values - output vector with the pagerank vector values corresponding to the vertices in xids
     * @return the actual length of xids
     */
    int64_t aclpagerank64(
        int64_t n, int64_t* ai, int64_t* aj, int64_t offset,
            //Compressed sparse row representation, with offset for
            //zero based (matlab) or one based arrays (julia)
        double alpha,    //value of alpha
        double eps,    //value of epsilon
        int64_t* seedids, int64_t nseedids,    //the set indices for seeds
        int64_t maxsteps,    //the maximum number of steps
        int64_t* xids, int64_t xlength, double* values); //solution vectors

    /**
     * A C interface for aclpagerank.
     * @param n - the number of vertices in the graph
     * @param ai - the indptr array for the compressed sparse row adjacency matrix
     * @param aj - the index array for the compressed sparse row adjacency matrix
     * @param offset - 0 or 1 depending on whether the calling code uses 0 or 1 based indexing
     * @param alpha - the alpha parameter for the approximate pagerank calculation
     * @param eps - the epsilon parameter for the approximate pagerank calculation
     * @param seedids - an array of vertex indices for seeding the personalised pagerank
     * @param nseedids - the number of vertices in seedids
     * @param maxsteps - the maximum number of steps to take when computeing the personalised pagerank
     * @param xids - output vector containing the ids of vertices with non-zero pagerank value
     * @param xlength - the maximum allowed length of xids
     * @param values - output vector with the pagerank vector values corresponding to the vertices in xids
     * @return the actual length of xids
     */
    uint32_t aclpagerank32_64(
        uint32_t n, int64_t* ai, uint32_t* aj, uint32_t offset, 
            //Compressed sparse row representation, with offset for
            //zero based (matlab) or one based arrays (julia)
        double alpha,   //value of alpha
        double eps,    //value of epsilon
        uint32_t* seedids, uint32_t nseedids,     //the set indices for seeds
        uint32_t maxsteps,   //the maximum number of steps
        uint32_t* xids, uint32_t xlength, double* values);  //solution vectors

#ifdef __cplusplus
}
#endif



