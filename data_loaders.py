# -*- coding: utf-8 -*-
# !/usr/bin/python
import sys
import math
import torch
import copy
import json
import random
import pickle
import numpy as np

sys.path.append("..")
from utils.utils import pad, mk_graph_for_gnn
from utils.utils import pad, check_query_equal, check_in
from utils.utils import text_to_tensor_1d, text_to_tensor_2d, pad_tensor_1d, pad_tensor_2d
from rules.grammar import AbstractQueryGraph


class GenerationDataLoader:

    def __init__(self, args):
        self.args = args
        self.wo_vocab = pickle.load(open(self.args.wo_vocab, 'rb'))

        self.wo_pad = self.wo_vocab.lookup(self.wo_vocab.pad_token)

    def process_one_data(self, d):

        question_toks = d["question_toks"]
        q = text_to_tensor_1d(question_toks, self.wo_vocab)
        gold_objs = d["rule_labels"]

        gold_graphs = []
        g = AbstractQueryGraph()
        g.init_state()
        for i, obj in enumerate(gold_objs):
            vertices, v_labels, edges = g.get_state()
            gold_graphs.append(mk_graph_for_gnn(vertices, v_labels, edges))
            op = g.cur_operation
            g.update_state(op, obj)

        return q, gold_graphs, gold_objs, d


    def load_data(self, datas, bs, use_small=False, shuffle=True):

        if use_small:
            datas = datas[:10]

        if shuffle:
            random.shuffle(datas)

        bl_x = []
        batch_index = -1  # the index of sequence batches
        sample_index = 0  # sequence index within each batch

        for d in datas:
            if sample_index % bs == 0:
                sample_index = 0
                batch_index += 1
                bl_x.append([])
            x = self.process_one_data(d)
            bl_x[batch_index].append(x)
            sample_index += 1

        self.iters = []
        self.n_batch = len(bl_x)
        for x in bl_x:
            batch = self.fix_batch(x)
            self.iters.append(batch)

    def fix_batch(self, x):

        q, gold_graphs, gold_objs, data = zip(*x)
        q, q_lens = pad_tensor_1d(q, self.wo_pad)

        if self.args.cuda:
            q = q.to(self.args.gpu)
            gold_graphs = [[[y.to(self.args.gpu) for y in g] for g in s] for s in gold_graphs]

        return q, q_lens, gold_graphs, gold_objs, data

    def next_batch(self):
        for b in self.iters:
            yield b


class RankingDataLoader:

    def __init__(self, args):
        self.args = args
        self.wo_vocab = pickle.load(open(self.args.wo_vocab, 'rb'))

        self.wo_pad = self.wo_vocab.lookup(self.wo_vocab.pad_token)

        self.cand_pool = pickle.load(open(self.args.cand_pool, "rb"))

    def process_one_data(self, d, training=False):

        q = text_to_tensor_1d(d["processed_question"], self.wo_vocab)

        pos_r, pos_r_lens = text_to_tensor_2d(d["processed_gold_query"], self.wo_vocab)

        cand_queries = d["processed_cand_queries"]
        if training:
            for i in range(len(cand_queries), self.args.ns):
                tmp = random.randint(0, len(self.cand_pool) - 1)
                while (check_in(self.cand_pool[tmp], cand_queries)):
                    tmp = random.randint(0, len(self.cand_pool) - 1)
                cand_queries.append(self.cand_pool[tmp])
            cand_queries = random.sample(cand_queries, self.args.ns)

        neg_rs = []
        neg_rs_lens = []
        for r in cand_queries:
            neg_r, neg_r_len = text_to_tensor_2d(r, self.wo_vocab)
            neg_rs.append(neg_r)
            neg_rs_lens.append(neg_r_len)

        neg_rs, neg_rs_blens = pad_tensor_2d(neg_rs, self.wo_pad)
        neg_rs_lens = torch.cat(neg_rs_lens, 0)

        return q, pos_r, pos_r_lens, \
               neg_rs, neg_rs_lens, neg_rs_blens, d


    def load_data(self, datas, bs, use_small=False, shuffle=True, training=False):

        if use_small:
            datas = datas[:100]

        if shuffle:
            random.shuffle(datas)

        bl_x = []
        batch_index = -1  # the index of sequence batches
        sample_index = 0  # sequence index within each batch

        miss_qid_list = []

        for d in datas:
            if not training and len(d["processed_cand_queries"]) == 0:
                miss_qid_list.append(d["id"])
                continue
            if sample_index % bs == 0:
                sample_index = 0
                batch_index += 1
                bl_x.append([])
            x = self.process_one_data(d, training=training)
            bl_x[batch_index].append(x)
            sample_index += 1

        self.iters = []
        self.n_batch = len(bl_x)
        for x in bl_x:
            batch = self.fix_batch(x)
            self.iters.append(batch)

    def fix_batch(self, x):

        q, pos_r, pos_r_lens, \
        neg_rs, neg_rs_lens, neg_rs_blens, data = zip(*x)

        q, q_lens = pad_tensor_1d(q, self.wo_pad)

        pos_r, pos_r_blens = pad_tensor_2d(pos_r, self.wo_pad)
        pos_r_lens = torch.cat(pos_r_lens, 0)

        neg_rs, _ = pad_tensor_2d(neg_rs, self.wo_pad)
        neg_rs_lens = torch.cat(neg_rs_lens, 0)
        neg_rs_blens = torch.cat(neg_rs_blens, 0)

        if self.args.cuda:
            device = self.args.gpu
            return q.to(device), q_lens, \
                   pos_r.to(device), pos_r_lens, pos_r_blens, \
                   neg_rs.to(device), neg_rs_lens, neg_rs_blens, data
        else:
            return q, q_lens, \
                   pos_r, pos_r_lens, pos_r_blens, \
                   neg_rs, neg_rs_lens, neg_rs_blens, data

    def next_batch(self):
        for b in self.iters:
            yield b