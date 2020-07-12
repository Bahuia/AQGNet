# -*- coding: utf-8 -*-
# !/usr/bin/python

"""
# @Time    : 2020/5/1
# @Author  : Yongrui Chen
# @File    : sparql.py
# @Software: PyCharm
"""

import re
import copy
import random
import sys
from collections import deque, namedtuple

sys.path.append("..")
from rules.grammar import AbstractQueryGraph

end_id = 4

class SPARQLParser:
    """
    a parser for transform SPARQL to AQG operation label ground truth.
    """
    def __init__(self):
        pass

    def _parse_where(self, clause):
        """
        parse where clause of the sparql.
        """
        clause = clause.strip(' ')
        tmp = [x.strip('.') for x in clause.split(' ') if x not in ['.', '', ' ']]
        assert len(tmp) % 3 == 0
        t_id = -1
        triples = []
        for i, x in enumerate(tmp):
            if i % 3 == 0:
                t_id += 1
                triples.append([x])
            else:
                triples[t_id].append(x)
        return triples

    def parse(self, sparql):

        is_ask = sparql.find('ASK') != -1
        is_count = sparql.find('COUNT') != -1

        p_where = re.compile(r'[{](.*?)[}]', re.S)

        where_clauses = re.findall(p_where, sparql)
        assert len(where_clauses) == 1
        edges = self._parse_where(where_clauses[0])

        if is_count:
            edges = [[x.replace('?uri', '?z') for x in y] for y in edges]
            edges.append(["?uri", "count", "?z"])

        if is_ask:
            edges.append(["?uri", "ask", edges[0][0]])

        vertices = list(set(sum(([t[0], t[2]] for t in edges), [])))

        random.shuffle(vertices)

        v_indexs = {v: i for i, v in enumerate(vertices)}
        v_labels = {v : self.check_type(v, is_v=True) for i, v in enumerate(vertices)}

        # rule ?uri 's index -> 0
        if v_indexs["?uri"] != 0:
            swap_name = None
            for v, v_id in v_indexs.items():
                if v_id == 0:
                    swap_name = v
                    break
            assert swap_name is not None
            v_indexs[swap_name] = v_indexs["?uri"]
            v_indexs["?uri"] = 0

        aqg = AbstractQueryGraph()
        for v, v_id in v_indexs.items():
            aqg.add_vertex(v_id, v_labels[v])

        for v1, e, v2 in edges:
            # v1 --> v2
            aqg.add_edge(v1=v_indexs[v1], v2=v_indexs[v2],
                         e_label=self.check_type(e, inv=False),
                          both_ends=False)

            # v2 --> v1
            aqg.add_edge(v1=v_indexs[v2], v2=v_indexs[v1],
                         e_label=self.check_type(e, inv=True),
                         both_ends=False)

        rule_labels = self.build_rule_obj_labels(aqg)
        return rule_labels, aqg

    def build_rule_obj_labels(self, aqg):

        def dfs(current_vertex):
            for next_vertex, edge_label in aqg.neighbours[current_vertex]:
                if next_vertex not in visit:
                    visit[next_vertex] = len(visit)
                    rule_labels.append(aqg.v_labels[next_vertex])
                    rule_labels.append(visit[current_vertex])
                    rule_labels.append(edge_label)
                    dfs(next_vertex)

        rule_labels = []
        visit = dict()
        start_vertex = 0
        visit[start_vertex] = 0
        rule_labels.append(aqg.v_labels[start_vertex])

        dfs(start_vertex)
        # terminal
        rule_labels.append(end_id)
        return rule_labels

    def check_type(self, x, is_v=False, inv=False):
        if is_v:
            if x == '?uri':
                return 0 # Ans

            if x[0] == '?':
                return 1 # Var

            if x.find("http://dbpedia.org/resource/") != -1:
                return 2 # Ent

            if x.find("http://dbpedia.org/ontology/") != -1:
                return 3 # Type

            raise ValueError('Wrong vertex type: {}'.format(x))
        else:
            if x == 'count':
                return 0 # Count

            if x == 'ask':
                return 1 # ASK

            if x == '<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>':
                return 2 # Isa

            if x.find("http://dbpedia.org/ontology/") != -1 or x.find("http://dbpedia.org/property/") != -1:
                if not inv:
                    return 3 #  direction "+" Rel
                else:
                    return 4 #  direction "-" Rel

            raise ValueError('Wrong edge type: {}'.format(x))


if __name__ == '__main__':
    p = SPARQLParser()
    labels1, aqg1 = p.parse("SELECT DISTINCT ?uri WHERE { "
                          "?x <http://dbpedia.org/ontology/chairman> <http://dbpedia.org/resource/Ronaldo> . "
                          "?x <http://dbpedia.org/ontology/ground> ?uri  . }")

    labels2, aqg2 = p.parse("SELECT DISTINCT ?uri WHERE { "
                          "?x <http://dbpedia.org/ontology/chairman> <http://dbpedia.org/resource/Ronaldo> . "
                          "?x <http://dbpedia.org/ontology/ground> ?uri  . }")

    print(aqg1.is_equal(aqg2))

    # cand_vertices = {
    #     2: ["<http://dbpedia.org/resource/Cleopatra_V_of_Egypt>",
    #         "<http://dbpedia.org/resource/Ptolemy_XIII_Theos_Philopator>"],
    #     3: ["<http://dbpedia.org/ontology/Royalty>",]
    # }
    #
    # res = aqg.grounding(cand_vertices)
    # print(res)