# MIT License
#
# Copyright (c) 2024 Tskit Developers
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Test cases for matrix-vector product stats
"""
import msprime
import numpy as np
import pytest

import tskit
from tests.test_highlevel import get_example_tree_sequences

# ↑ See https://github.com/tskit-dev/tskit/issues/1804 for when
# we can remove this.


# Implementation note: the class structure here, where we pass in all the
# needed arrays through the constructor was determined by an older version
# in which we used numba acceleration. We could just pass in a reference to
# the tree sequence now, but it is useful to keep track of exactly what we
# require, so leaving it as it is for now.
class RelatednessVector:
    def __init__(
        self,
        sample_weights,
        num_nodes,
        samples,
        nodes_time,
        edges_left,
        edges_right,
        edges_parent,
        edges_child,
        edge_insertion_order,
        edge_removal_order,
        sequence_length,
        verbosity=0,
        internal_checks=False,
    ):
        self.sample_weights = np.asarray(sample_weights, dtype=np.float64)
        # virtual root is at num_nodes; virtual samples are beyond that
        N = num_nodes + 1 + len(samples)
        # Quintuply linked tree
        self.parent = np.full(N, -1, dtype=np.int32)
        self.left_sib = np.full(N, -1, dtype=np.int32)
        self.right_sib = np.full(N, -1, dtype=np.int32)
        self.left_child = np.full(N, -1, dtype=np.int32)
        self.right_child = np.full(N, -1, dtype=np.int32)
        # Sample lists refer to sample *index*
        self.num_samples = np.full(N, 0, dtype=np.int32)
        # Edges and indexes
        self.edges_left = edges_left
        self.edges_right = edges_right
        self.edges_parent = edges_parent
        self.edges_child = edges_child
        self.edge_insertion_order = edge_insertion_order
        self.edge_removal_order = edge_removal_order
        self.sequence_length = sequence_length
        self.nodes_time = nodes_time
        self.samples = samples
        self.position = 0
        self.virtual_root = num_nodes
        self.x = np.zeros(N, dtype=np.float64)
        self.w = np.zeros(N, dtype=np.float64)
        self.stack = np.zeros(N, dtype=np.float64)
        self.verbosity = verbosity
        self.internal_checks = internal_checks

        for j, u in enumerate(samples):
            self.num_samples[u] = 1
            self.w[u] = self.sample_weights[j]
            # Add branch to the virtual sample
            v = num_nodes + 1 + j
            self.insert_branch(u, v)
            self.num_samples[v] = 1

    def print_state(self, msg=""):
        num_nodes = len(self.parent)
        print(f"..........{msg}................")
        print(f"position = {self.position}")
        for j in range(num_nodes):
            if j <= self.virtual_root:
                st = "NaN" if j >= self.virtual_root else f"{self.nodes_time[j]}"
                pt = (
                    "NaN"
                    if self.parent[j] == tskit.NULL
                    else f"{self.nodes_time[self.parent[j]]}"
                )
                print(
                    f"node {j} -> {self.parent[j]}: "
                    f"ns = {self.num_samples[j]}, "
                    f"z = ({pt} - {st})"
                    f" * ({self.position} - {self.x[j]})"
                    f" * {self.w[j]}"
                    f" = {self.get_z(j)}"
                )
            else:
                sample = self.samples[j - self.virtual_root - 1]
                print(f"node {j} -> virtual sample for : {sample}")
            print(f"         stack: {self.stack[j]}")
        roots = []
        u = self.left_child[self.virtual_root]
        while u != tskit.NULL:
            roots.append(u)
            u = self.right_sib[u]
        fmt = "{:<6}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}{:>12.6}{:>12.6}{:>12.6}"
        s = f"roots = {roots}\n"
        s += (
            fmt.format(
                "node",
                "parent",
                "lsib",
                "rsib",
                "lchild",
                "rchild",
                "nsamp",
                "stack",
                "weight",
                "z",
            )
            + "\n"
        )
        for u in range(num_nodes):
            u_str = f"{u}"
            if u == self.virtual_root:
                u_str = f"{u}(VR)"
            elif u > self.virtual_root:
                sample = self.parent[u]
                u_str = f"{u}({sample})"
            s += (
                fmt.format(
                    u_str,
                    self.parent[u],
                    self.left_sib[u],
                    self.right_sib[u],
                    self.left_child[u],
                    self.right_child[u],
                    self.num_samples[u],
                    self.stack[u],
                    self.w[u],
                    self.get_z(u),
                )
                + "\n"
            )
        print(s)

        print("Current state:")
        state = self.current_state()
        for j, x in enumerate(state):
            print(f"   {j}: {x}")
        print("..........................")

    def remove_branch(self, p, c):
        lsib = self.left_sib[c]
        rsib = self.right_sib[c]
        if lsib == -1:
            self.left_child[p] = rsib
        else:
            self.right_sib[lsib] = rsib
        if rsib == -1:
            self.right_child[p] = lsib
        else:
            self.left_sib[rsib] = lsib
        self.parent[c] = -1
        self.left_sib[c] = -1
        self.right_sib[c] = -1

    def insert_branch(self, p, c):
        self.parent[c] = p
        u = self.right_child[p]
        if u == -1:
            self.left_child[p] = c
            self.left_sib[c] = -1
            self.right_sib[c] = -1
        else:
            self.right_sib[u] = c
            self.left_sib[c] = u
            self.right_sib[c] = -1
        self.right_child[p] = c

    def remove_edge(self, p, c):
        if self.verbosity > 0:
            self.print_state(f"remove {int(p), int(c)}")
        assert p != -1
        self.stack[c] += self.get_z(c)
        self.remove_branch(p, c)
        # update weights
        u = p
        while u != tskit.NULL:
            self.w[u] -= self.w[c]
            u = self.parent[u]

    def insert_edge(self, p, c):
        if self.verbosity > 0:
            self.print_state(f"insert {int(p), int(c)}")
        assert p != -1
        assert self.parent[c] == -1, "contradictory edges"
        # update weights
        u = p
        while u != tskit.NULL:
            self.w[u] += self.w[c]
            u = self.parent[u]
        self.insert_branch(p, c)

    def get_z(self, u):
        p = self.parent[u]
        if p == tskit.NULL or u >= self.virtual_root:
            return 0.0
        time = self.nodes_time[p] - self.nodes_time[u]
        span = self.position - self.x[u]
        return time * span * self.w[u]

    def mrca(self, a, b):
        # just used for `current_state`
        aa = [a]
        while a != tskit.NULL:
            a = self.parent[a]
            aa.append(a)
        while b not in aa:
            b = self.parent[b]
        return b

    def current_state(self):
        """
        Compute the current output, for debugging.
        """
        if self.verbosity > 2:
            print("---------------")
        n = len(self.samples)
        virtual_samples = [j + self.virtual_root + 1 for j in range(n)]
        out = np.zeros(n)
        for j, a in enumerate(virtual_samples):
            # edges on the path up from a
            pa = a
            while pa != tskit.NULL:
                if self.verbosity > 2:
                    print("edge:", pa, self.get_z(pa))
                out[j] += self.get_z(pa) + self.stack[pa]
                pa = self.parent[pa]
        if self.verbosity > 2:
            print("---------------")
        return out

    def get_root_path(self, u):
        """
        Returns the list of nodes back to the virtual root.
        """
        root_path = []
        p = u
        while p != tskit.NULL:
            root_path.append(p)
            p = self.parent[p]
        return root_path

    def push_down(self, u):
        """
        Add the edge above u to its stack, and then move u's stack to its
        children.
        """
        if self.verbosity > 0:
            print(f"push_down({u})")
        if self.internal_checks:
            # this operation should not change the current output
            before_state = self.current_state()
            if self.verbosity > 1:
                self.print_state("before push down")

        if self.verbosity > 1:
            print(f"  adding {self.get_z(u)} to the stack of {u}")
        self.stack[u] += self.get_z(u)
        self.x[u] = self.position

        c = self.left_child[u]
        while c != tskit.NULL:
            if self.verbosity > 1:
                print(f"  pushing down {self.stack[u]} from {u} to {c}")
            self.stack[c] += self.stack[u]
            c = self.right_sib[c]

        self.stack[u] = 0
        if self.internal_checks:
            if self.verbosity > 1:
                self.print_state("after push down")
            after_state = self.current_state()
            np.testing.assert_allclose(before_state, after_state)

    def flush_root_path(self, root_path):
        """
        Clears all nodes on the path from the virtual root down to u
        by pushing the contributions of all their branches to the stack
        and pushing their stacks to their children.
        """
        if self.verbosity > 0:
            print(f"flush_root_path({root_path})")
        if self.internal_checks:
            # this operation should not change the current output
            before_state = self.current_state()

        j = len(root_path) - 1
        while j >= 0:
            p = root_path[j]
            self.push_down(p)
            j -= 1

        if self.internal_checks:
            after_state = self.current_state()
            np.testing.assert_allclose(before_state, after_state)

    def flush_edge(self, p, c):
        """
        TODO: UNUSED?
        Adds to stack[c] all contributions above c, and moves the contribution
        of the edge above c into the stack.
        """
        if self.verbosity > 0:
            print(f"flush_edge({c})")
        if self.internal_checks:
            # this operation should not change the current output
            before_state = self.current_state()

        self.stack[c] += self.get_z(c)
        self.x[c] = self.position

        if self.internal_checks:
            after_state = self.current_state()
            np.testing.assert_allclose(before_state, after_state)

        u = p
        while u != tskit.NULL:
            self.stack[c] += self.stack[u]
            self.stack[c] += self.get_z(u)
            u = self.parent[u]

        if self.internal_checks:
            after_state = self.current_state()
            np.testing.assert_allclose(before_state, after_state)

    def run(self):
        sequence_length = self.sequence_length
        M = self.edges_left.shape[0]
        in_order = self.edge_insertion_order
        out_order = self.edge_removal_order
        edges_left = self.edges_left
        edges_right = self.edges_right
        edges_parent = self.edges_parent
        edges_child = self.edges_child

        j = 0
        k = 0
        # TODO: self.position is redundant with left
        left = 0
        self.position = left

        while k < M and left <= self.sequence_length:
            while k < M and edges_right[out_order[k]] == left:
                p = edges_parent[out_order[k]]
                c = edges_child[out_order[k]]
                root_path = self.get_root_path(p)
                self.flush_root_path(root_path)
                self.remove_edge(p, c)
                k += 1
            while j < M and edges_left[in_order[j]] == left:
                p = edges_parent[in_order[j]]
                c = edges_child[in_order[j]]
                if self.position > 0:
                    root_path = self.get_root_path(p)
                    self.flush_root_path(root_path)
                assert self.parent[p] == tskit.NULL or self.x[p] == self.position
                self.insert_edge(p, c)
                self.x[c] = self.position
                j += 1
            right = sequence_length
            if j < M:
                right = min(right, edges_left[in_order[j]])
            if k < M:
                right = min(right, edges_right[out_order[k]])
            left = right
            self.position = left

        # self.print_state()

        # clear remaining things down to virtual samples
        for j, u in enumerate(self.samples):
            self.push_down(u)
            v = self.virtual_root + 1 + j
            self.remove_edge(u, v)

        if self.verbosity > 1:
            self.print_state()

        out = np.zeros(len(self.samples))
        for out_i in range(len(self.samples)):
            i = out_i + self.virtual_root + 1
            out[out_i] = self.stack[i]
        return out


def relatedness_vector(ts, sample_weights, **kwargs):
    rv = RelatednessVector(
        sample_weights,
        ts.num_nodes,
        samples=ts.samples(),
        nodes_time=ts.nodes_time,
        edges_left=ts.edges_left,
        edges_right=ts.edges_right,
        edges_parent=ts.edges_parent,
        edges_child=ts.edges_child,
        edge_insertion_order=ts.indexes_edge_insertion_order,
        edge_removal_order=ts.indexes_edge_removal_order,
        sequence_length=ts.sequence_length,
        **kwargs,
    )
    return rv.run()


def relatedness_matrix(ts):
    # TODO: this is the CENTRED matrix;
    # once #1623 goes in, make sure to add `centre=False`
    Sigma = ts.genetic_relatedness(
        sample_sets=[[i] for i in ts.samples()],
        indexes=[(i, j) for i in range(ts.num_samples) for j in range(ts.num_samples)],
        mode="branch",
        span_normalise=False,
        proportion=False,
    ).reshape((ts.num_samples, ts.num_samples))
    return Sigma


def verify_relatedness_vector(ts, w, *, internal_checks=False, verbosity=0):
    # TODO: genetic_relatedness is currently only centered,
    # so we can only check vectors with mean zero.
    w = np.round(len(w) * w)
    w = w - np.mean(w)
    R1 = relatedness_vector(
        ts, sample_weights=w, internal_checks=internal_checks, verbosity=verbosity
    )
    # TODO: Also we need to center the result.
    R1 -= np.mean(R1)
    Sigma = relatedness_matrix(ts)
    R2 = Sigma.dot(w)
    if verbosity > 0:
        print(ts.draw_text())
        print("weights:", w)
        print("here:", R1)
        print("with ts:", R2)
        print("Sigma:", Sigma)
    np.testing.assert_allclose(R1, R2)
    return R2


def check_relatedness_vector(ts, n=5, *, internal_checks=False, verbosity=0, seed=123):
    rng = np.random.default_rng(seed=seed)
    for _ in range(n):
        w = rng.normal(size=ts.num_samples)
        R = verify_relatedness_vector(
            ts, w, internal_checks=internal_checks, verbosity=verbosity
        )
    return R


class TestExamples:
    @pytest.mark.parametrize("n", [2, 3, 5])
    @pytest.mark.parametrize("seed", range(1, 4))
    def test_small_internal_checks(self, n, seed):
        ts = msprime.sim_ancestry(
            n,
            ploidy=1,
            sequence_length=1000,
            recombination_rate=0.01,
            random_seed=seed,
        )
        assert ts.num_trees >= 2
        check_relatedness_vector(ts, verbosity=1, internal_checks=True)

    @pytest.mark.parametrize("n", [2, 3, 5, 15])
    @pytest.mark.parametrize("seed", range(1, 5))
    def test_simple_sims(self, n, seed):
        ts = msprime.sim_ancestry(
            n,
            ploidy=1,
            population_size=20,
            sequence_length=100,
            recombination_rate=0.01,
            random_seed=seed,
        )
        assert ts.num_trees >= 2
        check_relatedness_vector(ts)

    @pytest.mark.parametrize("n", [2, 3, 5, 15])
    def test_single_balanced_tree(self, n):
        ts = tskit.Tree.generate_balanced(n).tree_sequence
        check_relatedness_vector(ts, internal_checks=True, verbosity=1)

    def test_internal_sample(self):
        tables = tskit.Tree.generate_balanced(4).tree_sequence.dump_tables()
        flags = tables.nodes.flags
        flags[3] = 0
        flags[5] = tskit.NODE_IS_SAMPLE
        tables.nodes.flags = flags
        ts = tables.tree_sequence()
        check_relatedness_vector(ts, verbosity=0)

    # @pytest.mark.skip()
    @pytest.mark.parametrize("seed", range(1, 5))
    def test_one_internal_sample_sims(self, seed):
        ts = msprime.sim_ancestry(
            10,
            ploidy=1,
            population_size=20,
            sequence_length=100,
            recombination_rate=0.01,
            random_seed=seed,
        )
        t = ts.dump_tables()
        # Add a new sample directly below another sample
        u = t.nodes.add_row(time=-1, flags=tskit.NODE_IS_SAMPLE)
        t.edges.add_row(parent=0, child=u, left=0, right=ts.sequence_length)
        t.sort()
        t.build_index()
        ts = t.tree_sequence()
        check_relatedness_vector(ts)

    def test_missing_flanks(self):
        ts = msprime.sim_ancestry(
            20,
            ploidy=1,
            population_size=20,
            sequence_length=100,
            recombination_rate=0.01,
            random_seed=1234,
        )
        assert ts.num_trees >= 2
        ts = ts.keep_intervals([[20, 80]])
        assert ts.first().interval == (0, 20)
        check_relatedness_vector(ts)

    @pytest.mark.parametrize("ts", get_example_tree_sequences())
    def test_suite_examples(self, ts):
        if ts.num_samples > 0:
            check_relatedness_vector(ts)

    @pytest.mark.parametrize("n", [2, 3, 10])
    def test_dangling_on_samples(self, n):
        # Adding non sample branches below the samples does not alter
        # the overall divergence *between* the samples
        ts1 = tskit.Tree.generate_balanced(n).tree_sequence
        D1 = check_relatedness_vector(ts1)
        tables = ts1.dump_tables()
        for u in ts1.samples():
            v = tables.nodes.add_row(time=-1)
            tables.edges.add_row(left=0, right=ts1.sequence_length, parent=u, child=v)
        tables.sort()
        tables.build_index()
        ts2 = tables.tree_sequence()
        D2 = check_relatedness_vector(ts2, internal_checks=True)
        np.testing.assert_array_almost_equal(D1, D2)

    @pytest.mark.parametrize("n", [2, 3, 10])
    def test_dangling_on_all(self, n):
        # Adding non sample branches below the samples does not alter
        # the overall divergence *between* the samples
        ts1 = tskit.Tree.generate_balanced(n).tree_sequence
        D1 = check_relatedness_vector(ts1)
        tables = ts1.dump_tables()
        for u in range(ts1.num_nodes):
            v = tables.nodes.add_row(time=-1)
            tables.edges.add_row(left=0, right=ts1.sequence_length, parent=u, child=v)
        tables.sort()
        tables.build_index()
        ts2 = tables.tree_sequence()
        D2 = check_relatedness_vector(ts2, internal_checks=True)
        np.testing.assert_array_almost_equal(D1, D2)

    def test_disconnected_non_sample_topology(self):
        # Adding non sample branches below the samples does not alter
        # the overall divergence *between* the samples
        ts1 = tskit.Tree.generate_balanced(5).tree_sequence
        D1 = check_relatedness_vector(ts1)
        tables = ts1.dump_tables()
        # Add an extra bit of disconnected non-sample topology
        u = tables.nodes.add_row(time=0)
        v = tables.nodes.add_row(time=1)
        tables.edges.add_row(left=0, right=ts1.sequence_length, parent=v, child=u)
        tables.sort()
        tables.build_index()
        ts2 = tables.tree_sequence()
        D2 = check_relatedness_vector(ts2, internal_checks=True)
        np.testing.assert_array_almost_equal(D1, D2)
