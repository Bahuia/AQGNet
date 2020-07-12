# -*- coding: utf-8 -*-
# !/usr/bin/python

"""
# @Time    : 2020/5/1
# @Author  : Yongrui Chen
# @File    : eval.py
# @Software: PyCharm
"""

import os
import re
import sys
import time
import json
import glob
import torch
import pickle
import argparse

sys.path.append("..")
from rules.grammar import AbstractQueryGraph
from utils.utils import generate_cand_queries


if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data_path', type=str, help='dataset', required=True)
    arg_parser.add_argument('--use_gold_structure', action='store_true', help='use small data', dest='gold')
    arg_parser.add_argument('--output', type=str, help='output data')
    arg_parser.add_argument('--kb_endpoint', type=str, required=True)
    args = arg_parser.parse_args()

    datas = pickle.load(open(args.data_path, "rb"))

    p_where = re.compile(r'[{](.*?)[}]', re.S)

    if not os.path.exists(args.output):
        os.makedirs(args.output)


    for i, d in enumerate(datas):
        if i % 100 == 0:
            out_dir = os.path.join(args.output, str(i) + "-" + str(i + 99))
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

        print("processing {} ...".format(d["id"]))


        # for training data
        if args.gold:
            where_clauses = re.findall(p_where, d["query"])[0]
            where_clauses = where_clauses.strip(" ").strip(".").strip(" ")
            triples = [[y.strip(" ") for y in x.strip(" ").split(" ")]
                       for x in where_clauses.split(". ")]
            gold_type = None
            for t in triples:
                if t[1] == "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>":
                    gold_type = t[-1].strip("<").strip(">")
            if gold_type:
                cand_types = set(d["cand_types"])
                cand_types.add(gold_type)
                d["cand_types"] = [x for x in list(cand_types)]

        aqgs = [[d["gold_aqg"], d]] if args.gold else d["pred_aqgs"]

        cands = []
        for aqg, data in aqgs:
            cands = generate_cand_queries(aqg, data, args.kb_endpoint)
            if cands == "TimeOut":
                cands = []
            if len(cands) != 0:
                break


        d.pop("gold_aqg")
        if "pred_aqgs" in d:
            d.pop("pred_aqgs")
        d["cand_queries"] = cands
        print(len(cands))

        out_path = os.path.join(out_dir, str(d["id"]) + ".json")
        json.dump(d, open(out_path, "w"), indent=4)

    print("\nFinish.\nResults save to \"{}\".\n".format(args.output))

