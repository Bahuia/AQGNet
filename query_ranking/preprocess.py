# -*- coding: utf-8 -*-
# !/usr/bin/python

"""
# @Time    : 2020/5/1
# @Author  : Yongrui Chen
# @File    : preprocess.py
# @Software: PyCharm
"""

import os
import re
import sys
import json
import math
import pickle
import random
import argparse
from nltk.stem import WordNetLemmatizer

sys.path.append("..")
from utils.dictionary import init_vocab
from utils.utils import tokenize_by_uppercase, get_rels_from_query


p_where = re.compile(r'[{](.*?)[}]', re.S)

def parse_type(type):
    prefix = "<http://dbpedia.org/ontology/"
    type_name = type[len(prefix):].strip(">")
    return tokenize_by_uppercase(type_name)

def parse_relation(relation):
    prefix = "<http://dbpedia.org/"
    rel = relation[len(prefix):].strip(">").split("/")
    rel_type = rel[0]
    rel_name = rel[1]
    return [rel_type] + tokenize_by_uppercase(rel_name)

def parse_entity(entity):
    prefix = "<http://dbpedia.org/resource/"
    entity_name = entity[len(prefix):].strip(">")
    return list(entity_name)


def parse_query(query):
    where_clauses = re.findall(p_where, query)
    assert len(where_clauses) == 1
    triples = where_clauses[0].strip(" ").split(" . ")

    types = []
    relations = []
    for triple in triples:
        s, p, o = triple.split(" ")
        if p == "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>":
            types.append(parse_type(o))
        else:
            rel = parse_relation(p)
            relations.append(rel)
    return relations + types

def get_triples_from_query(query):
    where_clauses = re.findall(p_where, query)[0]
    where_clauses = where_clauses.strip(" ").strip(".").strip(" ")
    triples = [[y.strip(" ") for y in x.strip(" ").split(" ") if y != ""]
               for x in where_clauses.split(". ")]
    triples = [" ".join(["?x" if y[0] == "?" and y[1] == "x" else y for y in x]) for x in triples]
    return triples

def cmp_queries(query1, query2):
    select_clause1 = query1[:query1.find("{")].strip(" ")
    select_clause2 = query2[:query2.find("{")].strip(" ")
    if select_clause1 != select_clause2:
        return False

    triples1 = set(get_triples_from_query(query1))
    triples2 = set(get_triples_from_query(query2))
    insect = triples1 & triples2

    if len(insect) == len(triples1) and len(insect) == len(triples2):
        return True
    return False

def preprocess(datas, training=False):
    processed_datas = []
    for d in datas:
        select_clause = d["query"][:d["query"].find("{")].strip(" ")
        triples = get_triples_from_query(d["query"])
        d["query"] = select_clause + "{ " + " . ".join(triples) +  " }"

        if training:
            d["cand_queries"] = [x for x in d["cand_queries"] if not cmp_queries(x, d["query"])]

        d["processed_question"] = d["question_toks"]
        d["processed_gold_query"] = parse_query(d["query"])

        processed_cand_queries = []
        for query in d["cand_queries"]:
            processed_cand_queries.append(parse_query(query))
        d["processed_cand_queries"] = processed_cand_queries

        processed_datas.append(d)
    return processed_datas

def get_cand_pool(datas):
    cand_pool = []
    for d in datas:
        cand_pool.append(d["processed_gold_query"])
        for query in d["processed_cand_queries"]:
            cand_pool.append(query)
    return cand_pool

def load_data(dir):
    datas = []
    files = os.listdir(dir)
    for file in files:
        file = os.path.join(dir, file)
        if not os.path.isdir(file):
            data = json.load(open(file, "r"))
            datas.append(data)
        else:
            datas.extend(load_data(file))
    return datas

def mk_vocabs(processed_datas):
    word_vocab = init_vocab()

    for d in processed_datas:
        for w_tok in d["processed_question"]:
            word_vocab.add(w_tok)

        for rel in d["processed_gold_query"]:
            for w_tok in rel:
                word_vocab.add(w_tok)

        for query in d["processed_cand_queries"]:
            for rel in query:
                for w_tok in rel:
                    word_vocab.add(w_tok)
    return word_vocab

if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data_path', type=str, help='dataset', required=True)
    arg_parser.add_argument('--output', type=str, help='output data')
    args = arg_parser.parse_args()

    is_train = args.data_path.find("train") != -1

    datas = load_data(args.data_path)
    datas = sorted(datas, key=lambda x: x["id"])
    processed_datas = preprocess(datas, is_train)

    pickle.dump(processed_datas, open(args.output, "wb"))

    if is_train:
        vocab_dir = "../vocab/"
        if not os.path.exists(vocab_dir):
            os.makedirs(vocab_dir)

        word_vocab = mk_vocabs(processed_datas)
        print("Word vocabulary size: {} ".format(len(word_vocab)))
        pickle.dump(word_vocab, open(os.path.join(vocab_dir, "ranking_word_vocab.pkl"), "wb"))

        cand_pool = get_cand_pool(processed_datas)
        print("Candidate query pool size: {}".format(len(cand_pool)))
        pickle.dump(cand_pool, open(os.path.join(vocab_dir, "cand_pool.pkl"), "wb"))

