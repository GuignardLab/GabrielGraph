"""
You should write your module/set of function/class(es) here
"""
import numpy as np
from scipy.spatial import Delaunay
from itertools import combinations
import scipy as sp

def build_gabriel_graph(node_ids, pos, data_struct="adj-dict", dist=False):
    """
    Build the gabriel graph of a set of nodes with
    associtated positions.

    Args:
        node_ids ([int, ] (size n)): list of node ids
        pos (n x m ndarray): ndarray of the positions where n is
            the number of nodes and m is the spatial dimension
        data_struct (str): in which type of data structure will
            the graph be saved, currently either 'adj-dict' and
            'adj-mat' are supported.
            'adj-dict': Adjacency dictionary
            'adj-mat' : Adjacency matrix
        dist (bool)
    Returns:
        final_GG (dict id: set([ids, ])): the gabriel graph as
            an adjacency list, a dictionary that maps node ids
            to the list of neighboring node ids
    """
    if not data_struct in ["adj-dict", "adj-mat"]:
        raise ValueError(
            "Data structure for the Gabriel graph not understood"
        )
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
            neighbs = np.array(list(neighbs))
            distances = np.linalg.norm(
                pos[e1] - [pos[ni] for ni in neighbs], axis=1
            )
            final_GG[node_ids[e1]] = {
                node_ids[ni]
                for ni in neighbs[distances <= 5 * np.median(distances)]
            }

    elif data_struct.lower() == "adj-mat":
        X, Y, val = [], [], []
        for e1, neighbs in delaunay_graph.items():
            for ni in [n for n in neighbs if e1 < n]:
                D = np.linalg.norm(pos[e1] - pos[ni])
                if not any(
                    np.linalg.norm((pos[ni] + pos[e1]) / 2 - pos[i])
                    < D / 2
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
        final_GG = final_GG.tocsr()

    return final_GG