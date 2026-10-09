"""Check the circuit's prefix network arithmetic, work and dependency depth."""
import ast
from pathlib import Path

import numpy as np
import pytest


class Vector:
    products = 0

    def __init__(self, values, depth=0):
        self.values = np.asarray(values)
        self.depth = np.broadcast_to(depth, self.values.shape)

    def __mul__(self, other):
        Vector.products += self.values.size
        return Vector(self.values * other.values, np.maximum(self.depth, other.depth) + 1)

    def __rsub__(self, other):
        return Vector(other - self.values, self.depth)

    def get_vector(self, base, size):
        return Vector(self.values[base:base + size], self.depth[base:base + size])


class Secret:
    def __new__(cls, value, size):
        return Vector(np.full(size, value))

    @staticmethod
    def concat(parts):
        return Vector(np.concatenate([x.values for x in parts]),
                      np.concatenate([x.depth for x in parts]))


@pytest.mark.parametrize('n', [1, 2, 3, 5, 10, 17, 50])
def test_parallel_prefix_selects_first_match_with_linear_work_and_log_depth(n):
    source = Path(__file__).resolve().parents[1] / 'mpc' / 'shared_graph_group.mpc'
    functions = [node for node in ast.parse(source.read_text()).body
                 if isinstance(node, ast.FunctionDef) and node.name == 'first_match_columns']
    assert len(functions) == 1
    scope = {'sint': Secret}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'), scope)
    width = 31
    bits = np.random.default_rng(n).integers(0, 2, (width, n))
    bits[0] = 0
    expected = bits * (np.cumsum(bits, axis=1) == 1)
    Vector.products = 0
    columns = scope['first_match_columns']([Vector(bits[:, j]) for j in range(n)], width, 'parallel')
    assert np.array_equal(np.array([v.values for v in columns]).T, expected)
    assert Vector.products <= 3 * n * width
    assert max(int(v.depth.max()) for v in columns) <= 2 * (n - 1).bit_length() + 1
