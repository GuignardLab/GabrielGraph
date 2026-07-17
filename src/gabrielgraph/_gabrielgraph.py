"""
A simple function to create a Gabriel Graph from a set of points and
their indices
For info: leo.guignard _at_ univ-amu.fr
"""

import warnings
from itertools import combinations
from typing import Literal, Sequence

import numpy as np
import scipy as sp
from numpy.typing import ArrayLike
from scipy.spatial import Delaunay, KDTree


def build_gabriel_graph(
    node_ids: Sequence,
    pos: ArrayLike,
    data_struct: Literal["adj-dict", "adj-mat"] = "adj-dict",
    dist: bool = False,
) -> dict[int, set[int]] | sp.sparse.coo_array:
    """
    Build the gabriel graph of a set of nodes with
    associated positions.

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
        Only used when `data_struct` is 'adj-mat': store the L2 norm
        between connected points instead of `True`.
        Note that if the distance is asked, for practical reasons
        (because of sparse matrices), no connection is coded as 0.
        Ignored when `data_struct` is 'adj-dict'; passing `dist=True`
        in that case raises a `UserWarning`.

    Returns
    -------
    dict maps int to set of ints
        the gabriel graph as an adjacency list, a dictionary that maps
        node ids to the set of neighboring node ids
    OR
    coo_array array of size N x N
        the gabriel graph as an adjacency matrix where
        `m[i, j]` is not `False` or `0` if the nodes `i` and `j` are
        connected and `m[i, j]` is the distance between the nodes `i`
        and `j` if dist is True. Otherwise `m[i, j]` is `True`
    """
    if data_struct.lower() not in ["adj-dict", "adj-mat"]:
        raise ValueError("Data structure for the Gabriel graph not understood")
    if dist and data_struct.lower() == "adj-dict":
        warnings.warn(
            "`dist=True` is only supported with data_struct='adj-mat'; "
            "it is ignored for 'adj-dict' output.",
            UserWarning,
            stacklevel=2,
        )
    pos = np.asarray(pos, dtype=float)
    tri = Delaunay(pos, qhull_options="QJ")

    # All unique Delaunay edges as an (E, 2) array with u < v.
    vertex_pairs = list(combinations(range(tri.simplices.shape[1]), 2))
    pairs = tri.simplices[:, vertex_pairs].reshape(-1, 2)
    edges = np.unique(np.sort(pairs, axis=1), axis=0)

    # A Delaunay edge is a Gabriel edge iff no site lies strictly inside
    # its diametral ball. The endpoints sit exactly at distance r from
    # the midpoint, so the edge is kept iff the nearest site to the
    # midpoint is at distance >= r (closed-ball convention: points
    # exactly on the sphere, up to a relative tolerance, do not
    # disqualify the edge).
    mid = pos[edges].mean(axis=1)
    r = np.linalg.norm(pos[edges[:, 0]] - pos[edges[:, 1]], axis=1) / 2
    nearest_d, _ = KDTree(pos).query(mid, k=1)
    gabriel = nearest_d >= r * (1 - 1e-12)
    gg_edges = edges[gabriel]

    if data_struct.lower() == "adj-dict":
        final_GG = {}
        for u, v in gg_edges:
            final_GG.setdefault(node_ids[u], set()).add(node_ids[v])
            final_GG.setdefault(node_ids[v], set()).add(node_ids[u])

    elif data_struct.lower() == "adj-mat":
        ids = np.asarray(node_ids)
        u, v = gg_edges[:, 0], gg_edges[:, 1]
        X = np.concatenate([ids[u], ids[v]])
        Y = np.concatenate([ids[v], ids[u]])
        if dist:
            val = np.tile(2 * r[gabriel], 2)
        else:
            val = np.ones(2 * len(gg_edges), dtype=bool)
        final_GG = sp.sparse.coo_array(
            (val, (X, Y)), shape=(max(node_ids) + 1,) * 2
        )

    return final_GG
