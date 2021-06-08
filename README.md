# Code for Local Graph Clustering for Learning Higher Order Structure
Here you will find the code implementing the algorithms introduced in the paper.

Our implementation extends the open source library available here: https://github.com/kfoynt/LocalGraphClustering

We add an implementation of 3 algorithms:
- `LocBipartDC`: our new algorithm for finding two sets in an undirected graph with a small bipartiteness
- `EvoCutDC`: our new algorithm for finding two sets in a directed graph with a small flow ratio
- `LPAlmosBipartite`: the algorithm by Li & Peng (2013) to which we compare `LocBipartDC`

## Installing and running the code
To install the dependencies and compile the code, run
- ```python3 -m pip install -r requirements.txt```
- ```bin/createGraphLibFile.sh```

Examples showing how to use our new algorithms are provided in the notebooks folder.