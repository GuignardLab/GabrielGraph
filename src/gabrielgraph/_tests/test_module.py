from itertools import combinations

import numpy as np
import pytest

from gabrielgraph import build_gabriel_graph


def gabriel_brute_force(pos):
    """O(N^3) reference implementation, straight from the definition:
    an edge (i, j) belongs to the Gabriel graph iff no other point lies
    strictly inside the closed diametral ball of the segment [i, j].
    """
    n = len(pos)
    graph = {}
    for i, j in combinations(range(n), 2):
        mid = (pos[i] + pos[j]) / 2
        r2 = ((pos[i] - pos[j]) ** 2).sum() / 4
        if all(
            ((pos[k] - mid) ** 2).sum() >= r2 * (1 - 1e-12)
            for k in range(n)
            if k not in (i, j)
        ):
            graph.setdefault(i, set()).add(j)
            graph.setdefault(j, set()).add(i)
    return graph


def test_module():
    """Regression test: fixed input, known output."""
    np.random.seed(0)
    ids = np.arange(10)
    pos = np.random.rand(20).reshape(-1, 2)
    test_obj = build_gabriel_graph(ids, pos)
    assert test_obj == {
        2: {0, 1, 3, 7, 8},
        0: {1, 2, 3, 6, 9},
        1: {0, 2, 5},
        3: {0, 2, 6, 8},
        7: {2, 8},
        8: {2, 3, 7},
        5: {1, 4, 9},
        4: {5},
        9: {0, 5, 6},
        6: {0, 3, 9},
    }


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("dim", [2, 3])
def test_against_brute_force(seed, dim):
    rng = np.random.default_rng(seed)
    pos = rng.random((30, dim))
    ids = np.arange(len(pos))
    assert build_gabriel_graph(ids, pos) == gabriel_brute_force(pos)


def test_adj_dict_symmetry():
    rng = np.random.default_rng(1)
    pos = rng.random((50, 3))
    gg = build_gabriel_graph(np.arange(len(pos)), pos)
    for node, neighbors in gg.items():
        for neighbor in neighbors:
            assert node in gg[neighbor]


def test_adj_mat_matches_adj_dict():
    rng = np.random.default_rng(2)
    pos = rng.random((50, 3))
    ids = np.arange(len(pos))
    gg_dict = build_gabriel_graph(ids, pos)
    gg_mat = build_gabriel_graph(ids, pos, "adj-mat").todok()
    mat_dict = {}
    for (i, j), value in gg_mat.items():
        assert value
        mat_dict.setdefault(i, set()).add(j)
    assert mat_dict == gg_dict


def test_adj_mat_dist():
    rng = np.random.default_rng(3)
    pos = rng.random((30, 2))
    ids = np.arange(len(pos))
    gg_mat = build_gabriel_graph(ids, pos, "adj-mat", dist=True).todok()
    assert len(gg_mat) > 0
    for (i, j), value in gg_mat.items():
        assert value == pytest.approx(np.linalg.norm(pos[i] - pos[j]))


def test_non_contiguous_node_ids():
    rng = np.random.default_rng(4)
    pos = rng.random((10, 2))
    ids = np.arange(10) * 3 + 1
    gg = build_gabriel_graph(ids, pos)
    assert set(gg) <= set(ids)
    reference = build_gabriel_graph(np.arange(10), pos)
    assert gg == {
        ids[n]: {ids[v] for v in neighbs} for n, neighbs in reference.items()
    }


def test_dist_warns_for_adj_dict():
    np.random.seed(0)
    ids = np.arange(10)
    pos = np.random.rand(20).reshape(-1, 2)
    with pytest.warns(UserWarning, match="adj-mat"):
        build_gabriel_graph(ids, pos, "adj-dict", dist=True)


def test_bad_data_struct_raises():
    with pytest.raises(ValueError):
        build_gabriel_graph([0, 1, 2], np.eye(3)[:, :2], "adj-list")
