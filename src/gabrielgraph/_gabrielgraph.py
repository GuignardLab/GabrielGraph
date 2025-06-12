"""
A simple function to create a Gabriel Graph from a set of point and there indices
For info: leo.guignard _at_ univ-amu.fr
"""

import numpy as np
from scipy.spatial import Delaunay
from itertools import combinations
import scipy as sp
from typing import Literal, Sequence
from numpy.typing import ArrayLike


def build_gabriel_graph(
    node_ids: Sequence,
    pos: ArrayLike,
    data_struct: Literal["adj-dict", "adj-mat"] = "adj-dict",
    dist: bool = False,
) -> dict[int, set[int]] | sp.sparse.coo_array:
    """
    Build the gabriel graph of a set of nodes with
    associtated positions.

    Parameters
    ----------
    node_ids : Sequence of int of length N the number of points
        list of node ids
    pos : ArrayLike of size N x D
        ndarray of the positions where N is the number of points
        and D is the number of spatial dimensions
    data_struct : {"adj-dict", "adj-mat"}
        data structure in which type of data structure will
        the graph will be returned.
        'adj-dict': Adjacency dictionary
        'adj-mat' : Adjacency matrix
    dist : bool
        in the case of adjacency matrix, put the L2 norm
        between the points if they are connected rather than True.
        /!\ Note that if the distance is asked, for pratical reason
        (because of sparse matrices), no connection is coded as 0. /!\
    
    Returns
    -------
    dict maps int to set of ints
        the gabriel graph as an adjacency list, a dictionary that maps node ids
        to the list of neighboring node ids
    OR
    coo_array array of size N x N
        the gabriel graph as an adjacency matrix where
        `m[i, j]` is not `False` or `0` if `m[i, j]` are connected
        and `m[i, j]` is the distance between the nodes `i` and `j`
        if dist is True. Otherwise `m[i, j]` is `True`
    """
    if data_struct not in ["adj-dict", "adj-mat"]:
        raise ValueError("Data structure for the Gabriel graph not understood")
    tmp = Delaunay(pos)
    delaunay_graph = {}

    for N in tmp.simplices:
        for e1, e2 in combinations(np.sort(N), 2):
            delaunay_graph.setdefault(e1, set()).add(e2)
            delaunay_graph.setdefault(e2, set()).add(e1)

    if data_struct.lower() == "adj-dict":
        Gabriel_graph = {}
        for e1, neighbs in delaunay_graph.items():
            for ni in neighbs:
                if not any(
                    np.linalg.norm((pos[ni] + pos[e1]) / 2 - pos[i])
                    < np.linalg.norm(pos[ni] - pos[e1]) / 2
                    for i in neighbs.intersection(delaunay_graph[ni])
                ):
                    Gabriel_graph.setdefault(e1, set()).add(ni)
                    Gabriel_graph.setdefault(ni, set()).add(e1)

        final_GG = {}
        for e1, neighbs in Gabriel_graph.items():
            final_GG[node_ids[e1]] = {node_ids[ni] for ni in neighbs}

    elif data_struct.lower() == "adj-mat":
        X, Y, val = [], [], []
        for e1, neighbs in delaunay_graph.items():
            for ni in [n for n in neighbs if e1 < n]:
                D = np.linalg.norm(pos[e1] - pos[ni])
                if not any(
                    np.linalg.norm((pos[ni] + pos[e1]) / 2 - pos[i]) < D / 2
                    for i in neighbs.intersection(delaunay_graph[ni])
                ):
                    X.append(node_ids[e1])
                    Y.append(node_ids[ni])
                    X.append(node_ids[ni])
                    Y.append(node_ids[e1])
                    if dist:
                        val.append(D)
                        val.append(D)
                    else:
                        val.append(True)
                        val.append(True)
        final_GG = sp.sparse.coo_array(
            (val, (X, Y)), shape=(max(node_ids) + 1, max(node_ids) + 1)
        )

    return final_GG
