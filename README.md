# Local Algorithms for Finding Densely Connected Clusters
This repository contains code to accompany the paper "Local Algorithms for Finding Densely Connected Clusters",
published at ICML 2021.

Please see [the demo page](https://pmacg.io/conflict/) to experiment with the MID conflict dataset for yourself.

The implementation extends the open source library available [here](https://github.com/kfoynt/LocalGraphClustering)
with the new algorithms introduced in the paper.

We add an implementation of 3 algorithms:
- `LocBipartDC`: our new algorithm for finding two sets in an undirected graph with a small bipartiteness ratio
- `EvoCutDC`: our new algorithm for finding two sets in a directed graph with a small flow ratio
- `LPAlmosBipartite`: the algorithm by Li & Peng (2013) to which we compare `LocBipartDC`

## Installing and running the code
To install the dependencies and compile the code, run
- ```python3 -m pip install -r requirements.txt```
- ```bin/createGraphLibFile.sh```

We provide some simple examples showing how to use our new algorithms in the `notebooks` folder.

If you are interested in digging into the code, the entry point for the new algorithms
is in the `localgraphclustering/find_bipartite_clusters.py` file.

## Contact
If you have any questions, or would like help with getting up and running, please contact
[peter.macgregor@ed.ac.uk](mailto:peter.macgregor@ed.ac.uk).
