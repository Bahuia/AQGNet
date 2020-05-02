# -*- coding: utf-8 -*-
# !/usr/bin/python

"""
# @Time    : 2020/5/1
# @Author  : Yongrui Chen
# @File    : grammar.py
# @Software: PyCharm
"""

import re
import sys
import copy
import numpy as np
import itertools
from collections import deque, namedtuple

sys.path.append("..")
from utils.query_interface import DBpedia_query
from utils.utils import check_relation

Edge = namedtuple('Edge', 'start, end, label')

end_id = 4


class AbstractQueryGraph:
    """
    AQG data structure
    """
    def __init__(self):
        self.vertices = set()
        self.edges = []

        self.v_labels = dict()

    def get_vertex_pairs(self, v1, v2, both_ends=True):
        if both_ends:
            vertex_pairs = [[v1, v2], [v2, v1]]
        else:
            vertex_pairs = [[v1, v2]]
        return vertex_pairs

    def remove_edge(self, v1, v2, both_ends=True):
        vertex_pairs = self.get_vertex_pairs(v1, v2, both_ends)
        edges = self.edges[:]
        for edge in edges:
            if [edge.start, edge.end] in vertex_pairs:
                self.edges.remove(edge)

    def add_vertex(self, v, v_label):
        self.vertices.add(v)
        self.v_labels[v] = v_label

    def add_edge(self, v1, v2, e_label, both_ends=True):
        self.edges.append(Edge(start=v1, end=v2, label=e_label))
        if both_ends:
            self.edges.append(Edge(start=v2, end=v1, label=e_label))

    @property
    def neighbours(self):
        neighbours = {vertex: set() for vertex in self.vertices}
        for edge in self.edges:
            neighbours[edge.start].add((edge.end, edge.label))
        return neighbours

    @property
    def cur_operation(self):
        if self.op_idx == 0:
            return 'av'
        elif self.op_idx % 3 == 1:
            return 'av'
        elif self.op_idx % 3 == 2:
            return 'sv'
        else:
            return 'ae'

    def init_state(self):
        self.vertices = set()
        self.edges = []

        self.v_labels = dict()
        self.pred_obj_labels = []
        self.cur_v_add = -1
        self.cur_v_slc = -1
        self.op_idx = 0

    def update_state(self, op, obj):
        """
        :param op:  { "av", "sv", "ae" }
        :param obj: object
        """
        if op == "av":
            if obj != end_id:
                self.cur_v_add = len(self.vertices)
                self.add_vertex(len(self.vertices), obj)

        elif op == "sv":
            self.cur_v_slc = obj

        elif op == "ae":
            self.add_edge(self.cur_v_slc, self.cur_v_add, obj)

        else:
            raise ValueError("Operation \"{}\" is wrong !".format(op))
        self.pred_obj_labels.append(obj)
        self.op_idx += 1

    def get_state(self):
        """
        get current state of the aqg.
        :return:
        """
        vertices = [x for x in self.vertices]
        v_labels = {k:v for k, v in self.v_labels.items()}
        edges = [x for x in self.edges]
        return vertices, v_labels, edges

    def is_equal(self, another_aqg):
        """
        check whether two aqg are identical
        """
        assert type(another_aqg) == AbstractQueryGraph

        if len(self.vertices) != len(another_aqg.vertices):
            return False

        v_labels1 = [self.v_labels[x] for x in self.vertices]
        v_labels2 = [another_aqg.v_labels[x] for x in another_aqg.vertices]

        v_labels1 = " ".join([str(x) for x in sorted(v_labels1)])
        v_labels2 = " ".join([str(x) for x in sorted(v_labels2)])

        if v_labels1 != v_labels2:
            return False

        edges1 = [[self.v_labels[e[0]], self.v_labels[e[1]], e[2]] for e in self.edges]
        edges2 = [[another_aqg.v_labels[e[0]], another_aqg.v_labels[e[1]], e[2]] for e in another_aqg.edges]
        edges1 = ";".join(sorted([" ".join([str(x) for x in e]) for e in edges1]))
        edges2 = ";".join(sorted([" ".join([str(x) for x in e]) for e in edges2]))

        if edges1 != edges2:
            return False

        vertices1 = [v for v in self.vertices]
        vertex_labels1 = [self.v_labels[v] for v in self.vertices]
        vertex_idx1 = {v: i for i, v in enumerate(vertices1)}
        adj1 = np.full((len(vertices1), len(vertices1)), -1)
        for v1, v2, e in self.edges:
            adj1[vertex_idx1[v1]][vertex_idx1[v2]] = e
        adj_flat1 = " ".join([str(x) for x in adj1.flatten()])

        vertices2 = [v for v in another_aqg.vertices]

        for perm in itertools.permutations([i for i in range(len(vertices2))], len(vertices2)):
            vertex_idx2 = {v: perm[i] for i, v in enumerate(vertices2)}
            vertex_labels2 = [0 for _ in range(len(vertices2))]
            for v in vertices2:
                vertex_labels2[vertex_idx2[v]] = another_aqg.v_labels[v]

            if " ".join([str(x) for x in vertex_labels1]) != " ".join([str(x) for x in vertex_labels2]):
                continue

            adj2 = np.full((len(vertices2), len(vertices2)), -1)
            for v1, v2, e in another_aqg.edges:
                adj2[vertex_idx2[v1]][vertex_idx2[v2]] = e

            adj_flat2 = " ".join([str(x) for x in adj2.flatten()])

            if adj_flat1 == adj_flat2:
                return True
        return False

    def mk_patterns(self, index):
        """
        make SPARQL patterns from AQG by dfs, take the direction of edges in account
        """
        if index >= len(self.edges):
            return [[]]
        new_patterns = []
        v1, v2, e_label = self.edges[index]
        patterns = self.mk_patterns(index + 2)
        # Rel
        if e_label == 3:
            for p in patterns:
                new_patterns.append([(v1, v2, e_label)] + p)
                new_patterns.append([(v2, v1, e_label)] + p)
        # Isa
        elif e_label == 2:
            # v1 == Type
            if self.v_labels[v1] == 3:
                for p in patterns:
                    new_patterns.append([(v2, v1, e_label)] + p)
            else:
                for p in patterns:
                    new_patterns.append([(v1, v2, e_label)] + p)
        else:
            for p in patterns:
                new_patterns.append(p)
        return new_patterns

    def grounding(self, cand_vertices, kb_endpoint):
        """
        grounding the aqg to generate candidate queries
        :param cand_vertices:
        :return:
        """

        aqg_vertices = {}
        for v, v_class in self.v_labels.items():
            # do not consider variables and answers
            if v_class in [0, 1]:
                continue
            if v_class not in aqg_vertices:
                aqg_vertices[v_class] = []
            aqg_vertices[v_class].append(v)

        # check is aqg match candidate vertices
        flag = True
        for v_class, vertices in aqg_vertices.items():
            if v_class not in cand_vertices or len(cand_vertices[v_class]) < len(vertices):
                flag = False
                break
        if not flag:
            return []

        query_intention = "NONE"
        for v1, v2, e_label in self.edges:
            if e_label == 0:
                query_intention = "COUNT"
            elif e_label == 1:
                query_intention = "ASK"

        tmps = {}
        for v_class, vertices in aqg_vertices.items():
            tmps[v_class] = []
            for perm in itertools.permutations(cand_vertices[v_class], len(vertices)):
                v_map = {vertices[i]: perm[i] for i in range(len(vertices))}
                tmps[v_class].append(v_map)
        vertex_maps = [[]]
        for v_class, v_map in tmps.items():
            vertex_maps = [x for x in itertools.product(vertex_maps, v_map)]
            vertex_maps = [x + list(y.items()) for x, y in vertex_maps]

        vertex_maps = [{k:v for k, v in x} for x in vertex_maps]

        sparql_patterns = self.mk_patterns(0)

        # first grounding: fill vertices into patterns
        conds_set = set()
        first_grounded_sparqls = []
        for v_map in vertex_maps:
            for pattern in sparql_patterns:
                condition = []
                for i, (v1, v2, e_label) in enumerate(pattern):
                    if self.v_labels[v1] == 0:
                        g_v1 = "?uri"
                    elif self.v_labels[v1] == 1:
                        g_v1 = "?x_" + str(v1)
                    else:
                        g_v1 = v_map[v1]

                    if self.v_labels[v2] == 0:
                        g_v2 = "?uri"
                    elif self.v_labels[v2] == 1:
                        g_v2 = "?x_" + str(v2)
                    else:
                        g_v2 = v_map[v2]

                    # e_label != "ASK" and "COUNT"
                    assert e_label != 0 and e_label != 1
                    if e_label == 2:    # Isa
                        g_e = "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>"
                    else:               # Rel
                        g_e = "?rel"
                    condition.append(" ".join((g_v1, g_e, g_v2)))

                # avoid duplicate conditions
                cond_sort = ";".join(sorted(condition))
                if cond_sort not in conds_set:
                    conds_set.add(cond_sort)
                    need_grounded_rels = []
                    for i, cond in enumerate(condition):
                        g_v1, g_e, g_v2 = cond.split(" ")
                        if g_e == "?rel":
                            g_e = g_e + "_" + str(i)
                            need_grounded_rels.append(g_e)
                        condition[i] = " ".join((g_v1, g_e, g_v2))
                    condition = "{ " + " . ".join(condition) + " }"
                    need_grounded_rels = " ".join(need_grounded_rels)
                    tmp_sparql = "SELECT " + need_grounded_rels + " WHERE " + condition
                    first_grounded_sparqls.append(tmp_sparql)

        p_where = re.compile(r'[{](.*?)[}]', re.S)

        # second grounding for "Rel" edges.
        grounded_sparqls = []
        for s in first_grounded_sparqls:
            results = DBpedia_query(s, kb_endpoint)
            for res in results:
                flag = True
                for k, v in res.items():
                    if not check_relation(v):
                        flag = False
                        break
                if not flag:
                    continue
                if len(res) > 0:
                    where_clauses = re.findall(p_where, s)
                    assert len(where_clauses) == 1
                    if query_intention == "COUNT":
                        new_s = "SELECT DISTINCT COUNT(?uri) WHERE {" + where_clauses[0] + "}"
                    elif query_intention == "ASK":
                        new_s = "ASK WHERE {" + where_clauses[0] + "}"
                    else:
                        new_s = "SELECT DISTINCT ?uri WHERE {" + where_clauses[0] + "}"
                    for k, v in res.items():
                        new_s = new_s.replace("?" + k, "<" + v + ">")
                    grounded_sparqls.append(new_s)
        grounded_sparqls = list(set(grounded_sparqls))

        return grounded_sparqls

