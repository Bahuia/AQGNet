# -*- coding: utf-8 -*-
# !/usr/bin/python

"""
# @Time    : 2020/5/1
# @Author  : Yongrui Chen
# @File    : query_interface.py
# @Software: PyCharm
"""

import json
import re
from datetime import datetime

from SPARQLWrapper import SPARQLWrapper, JSON

# DBpedia_endpoint = "https://dbpedia.org/sparql"


def DBpedia_query(_query, kb_endpoint):
    """
    :param _query: sparql query statement
    :return:
    """
    sparql = SPARQLWrapper(kb_endpoint)
    sparql.setQuery(_query)
    sparql.setReturnFormat(JSON)
    # sparql.setTimeout(5)
    response = sparql.query().convert()
    results = parse_query_results(response, _query)
    return results

def parse_query_results(response, query):
    if 'ASK' in query:  # ask
        result = response['boolean']
    elif 'COUNT' in query:  # count
        result = int(response['results']['bindings'][0]['callret-0']['value'])
    else:
        result = []
        for res in response['results']['bindings']:
            res = {k: v["value"] for k, v in res.items()}
            result.append(res)
    return result

def formalize(query):
    p_where = re.compile(r'[{](.*?)[}]', re.S)
    select_clause = query[:query.find("{")].strip(" ")
    select_clause = [x.strip(" ") for x in select_clause.split(" ")]
    select_clause = " ".join([x for x in select_clause if x != ""])
    select_clause = select_clause.replace("DISTINCT COUNT(?uri)", "COUNT(?uri)")

    where_clauses = re.findall(p_where, query)[0]
    where_clauses = where_clauses.strip(" ").strip(".").strip(" ")
    triples = [[y.strip(" ") for y in x.strip(" ").split(" ") if y != ""]
               for x in where_clauses.split(". ")]
    triples = [" ".join(["?x" if y[0] == "?" and y[1] == "x" else y for y in x]) for x in triples]
    where_clause = " . ".join(triples)
    query = select_clause + "{ " + where_clause + " }"
    return query

def query_answers(query, kb_endpoint):
    query = formalize(query)
    sparql = SPARQLWrapper(kb_endpoint)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    # sparql.setTimeout(5)
    response = sparql.query().convert()

    if "ASK" in query:
        results = [str(response["boolean"])]
    elif "COUNT" in query:
        tmp = response["results"]["bindings"]
        assert len(tmp) == 1 and ".1" in tmp[0]
        results = [tmp[0][".1"]["value"]]
    else:
        tmp = response["results"]["bindings"]
        results = [x["uri"]["value"] for x in tmp]
    return results


if __name__ == '__main__':

    start_time = datetime.now()

    query = "SELECT DISTINCT ?uri WHERE { <http://dbpedia.org/resource/Dave_Bing> <http://dbpedia.org/property/draftTeam> ?uri. " \
            "<http://dbpedia.org/resource/Ron_Reed> <http://dbpedia.org/property/draftTeam> ?uri . }"
    results = DBpedia_query(query)
    print(results)

    print('used time ', datetime.now() - start_time)