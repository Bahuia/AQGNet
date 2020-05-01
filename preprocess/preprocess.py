# -*- coding: utf-8 -*-
# !/usr/bin/python

"""
# @Time    : 2020/5/1
# @Author  : Yongrui Chen
# @File    : preprocess.py
# @Software: PyCharm
"""

import os
import sys
import json
import math
import pickle
import random
import argparse
from nltk.stem import WordNetLemmatizer

sys.path.append("..")
from utils.dictionary import init_vocab
from utils.utils import tokenize_by_uppercase
from rules.sparql import SPARQLParser
from pargs import pargs


wordnet_lemmatizer = WordNetLemmatizer()

stop_words = ["the", ""]


def preprocess(datas):

    processed_datas = []
    for d in datas:
        d["id"] = int(d["id"])
        question = d["question"].lower()
        if d["entity1_mention"] != "":
            question = question.replace(d["entity1_mention"].lower(), "<e>")
        if d["entity2_mention"] != "":
            question = question.replace(d["entity2_mention"].lower(), "<e>")

        question = question.replace("?", " ?")
        question = question.replace("'s", " 's")
        question = question.replace("'re", " 're")

        question_toks = question.split(" ")
        question_toks = [wordnet_lemmatizer.lemmatize(x) for x in question_toks if x not in stop_words]

        d["question_toks"] = question_toks

        parser = SPARQLParser()
        rule_labels, aqg = parser.parse(d["query"])

        d["rule_labels"] = rule_labels
        d["gold_aqg"] = aqg

        processed_datas.append(d)
    return processed_datas


def mk_vocabs(processed_datas):
    word_vocab = init_vocab()

    for d in processed_datas:
        for tok in d["question_toks"]:
            word_vocab.add(tok)
    return word_vocab


if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data_path', type=str, help='dataset', required=True)
    arg_parser.add_argument('--output', type=str, help='output data')
    args = arg_parser.parse_args()

    is_train = args.data_path.find("train") != -1

    datas = json.load(open(args.data_path, "r"))
    processed_datas = preprocess(datas)

    pickle.dump(processed_datas, open(args.output, "wb"))

    if is_train:
        vocab_dir = "../vocab/"
        if not os.path.exists(vocab_dir):
            os.makedirs(vocab_dir)

        word_vocab = mk_vocabs(processed_datas)
        print("Word vocabulary size: {} \n".format(len(word_vocab)))
        pickle.dump(word_vocab, open(os.path.join(vocab_dir, "word_vocab.pkl"), "wb"))

