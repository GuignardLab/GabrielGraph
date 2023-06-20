from gabrielgraph import build_gabriel_graph
import numpy as np


def test_module():
    """You should write tests here!"""
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
