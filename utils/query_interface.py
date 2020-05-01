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
DBpedia_endpoint = "http://10.201.180.179:3030/dbpedia/sparql"


def DBpedia_query(_query):
    """
    :param _query: sparql query statement
    :return:
    """
    sparql = SPARQLWrapper(DBpedia_endpoint)
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


if __name__ == '__main__':

    start_time = datetime.now()

    query = "SELECT DISTINCT ?uri WHERE { <http://dbpedia.org/resource/Dave_Bing> <http://dbpedia.org/property/draftTeam> ?uri. " \
            "<http://dbpedia.org/resource/Ron_Reed> <http://dbpedia.org/property/draftTeam> ?uri . }"
    results = DBpedia_query(query)
    print(results)

    print('used time ', datetime.now() - start_time)