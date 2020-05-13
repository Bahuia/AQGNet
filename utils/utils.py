# -*- coding: utf-8 -*-
# !/usr/bin/python

"""
# @Time    : 2020/5/1
# @Author  : Yongrui Chen
# @File    : utils.py
# @Software: PyCharm
"""

import sys
from builtins import range
import numpy as np
import torch
import torch.nn as nn
import copy
import json
from nltk.stem import WordNetLemmatizer
import signal, functools

REAL = np.float32
if sys.version_info[0] >= 3:
    unicode = str


def identity(x):
    return x

def pad(tensor, length, pad_idx):
    """
    :param tensor: Size(src_sent_len, ...)
    :param length: target_sent_length
    :param pad_idx: index of padding token
    :return: Size(target_sent_length, ...)
    """
    return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).fill_(pad_idx)])

def text_to_tensor_1d(ex, dict):
    """
    :param ex: text, 1-d list of tokens
    :return: Size(sent_len)
    """
    return torch.LongTensor(dict.convert_to_index(ex))

def text_to_tensor_2d(ex, dict):
    """
    :param ex: [text_1, text_2, ... text_n]
    :return: Size(n, max_sent_len)
    """
    return pad_tensor_1d([dict.convert_to_index(y) for y in ex], dict.lookup(dict.pad_token))

def pad_tensor_1d(l, pad_idx):
    """
    :param l: [Size(len1), Size(len2), ..., Size(lenn)]
    :return: Size(n, max_sent_len), Size(n)
    """
    lens = [len(x) for x in l]
    m = max(lens)
    return torch.stack([pad(torch.LongTensor(x), m, pad_idx) for x in l], 0), torch.LongTensor(lens)

def pad_tensor_2d(l, pad_idx):
    """
    :param l: [Size(n1, len1), Size(n2, len2) ...]
    :return: Size(n1 + n2 + .. nn, max(len1, len2... lenn))
    """
    lens = [x.size(0) for x in l]
    m = max([x.size(1) for x in l])
    data = [pad(x.transpose(0, 1), m, pad_idx).transpose(0, 1) for x in l]
    data = torch.cat(data, 0)
    return data, torch.LongTensor(lens)

def length_array_to_mask_tensor(length_array, cuda=False, value=None):
    max_len = max(length_array)
    batch_size = len(length_array)

    mask = np.ones((batch_size, max_len), dtype=np.uint8)
    for i, seq_len in enumerate(length_array):
        mask[i][:seq_len] = 0

    if value != None:
        for b_id in range(len(value)):
            for c_id, c in enumerate(value[b_id]):
                if value[b_id][c_id] == [3]:
                    mask[b_id][c_id] = 1

    mask = torch.ByteTensor(mask)
    return mask.cuda() if cuda else mask


def mk_graph_for_gnn(vertices, v_labels, edges):
    v_index = {v: i for i, v in enumerate(vertices)}

    v_tensor = torch.LongTensor([v_labels[x] for x in vertices])
    e_tensor = torch.LongTensor([x[-1] for x in edges])
    v_num = len(vertices)
    e_num = len(edges)
    adj_sz = v_num + 1 + e_num

    adj = torch.zeros(adj_sz, adj_sz)
    for i in range(adj_sz):
        adj[i, v_num] = 1
        adj[v_num, i] = 1
    for i in range(adj_sz):
        adj[i, i] = 1
    for i, e in enumerate(edges):
        a = v_index[e[0]]
        b = v_index[e[1]]
        c = i + v_num + 1
        adj[a, c] = 1
        adj[c, b] = 1
    return v_tensor, e_tensor, adj

def tokenize_by_uppercase(s):
    tokens = []
    last = 0
    for i, c in enumerate(s):
        if c.isupper():
            tokens.append(s[last: i])
            last = i
    tokens.append(s[last: len(s)])
    return tokens[1:]


class TimeoutError(Exception): pass


def timeout(seconds, error_message="Timeout Error: the cmd 30s have not finished."):
    def decorated(func):
        result = ""

        def _handle_timeout(signum, frame):
            global result
            result = "TimeOut"
            raise TimeoutError(error_message)

        def wrapper(*args, **kwargs):
            global result
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)

            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
                return result
            return result

        return functools.wraps(func)(wrapper)
    return decorated

def formalize_aqg(aqg, data):

    type_v = -1
    for v, label in aqg.v_labels.items():
        if label == 3:
            type_v = v
            break

    ent_v_num = 0
    for v, label in aqg.v_labels.items():
        if label == 2:
            ent_v_num += 1

    cand_types = ["<" + x + ">" for x in data["cand_types"]]

    if type_v == -1:
        if len(cand_types) > 0 and ent_v_num == 1:
            if len(aqg.edges) >= 4:
                v_add = len(aqg.vertices)
                aqg.add_vertex(v_add, 3)
                assert aqg.v_labels[1] == 1
                aqg.add_edge(1, v_add, 2)
                aqg.pred_obj_labels.extend([3, 1, 2])
            else:
                # pass
                v_add = len(aqg.vertices)
                aqg.add_vertex(v_add, 3)
                assert aqg.v_labels[0] == 0
                aqg.add_edge(0, v_add, 2)
                aqg.pred_obj_labels.extend([3, 0, 2])
    else:
        if len(cand_types) == 0:
            attached_u = -1
            for v1, v2, e_label in aqg.edges:
                if v1 == type_v:
                    attached_u = v2
                    break
            assert attached_u != -1

            aqg.remove_edge(attached_u, type_v)
            aqg.vertices.remove(type_v)
            aqg.v_labels.pop(type_v)

            pred_obj_labels = [x for x in aqg.pred_obj_labels]
            type_label_index = -1
            for i, obj in enumerate(pred_obj_labels):
                if i % 3 == 1 and obj == 3:
                    type_label_index = i
            aqg.pred_obj_labels = pred_obj_labels[:type_label_index] + pred_obj_labels[type_label_index + 3:]
    return aqg

def kb_constraint(aqg, data, kb_endpoint):

    cand_vertices = {2: []}

    if data["entity1_uri"] != "":
        cand_vertices[2].append("<" + data["entity1_uri"] + ">")
    if data["entity2_uri"] != "":
        cand_vertices[2].append("<" + data["entity2_uri"] + ">")

    cand_vertices[3] = ["<" + x + ">" for x in data["cand_types"]]

    grounding_res = aqg.grounding(cand_vertices, kb_endpoint)

    if len(grounding_res) == 0:
        # type
        type_v = -1
        for v, label in aqg.v_labels.items():
            if label == 3:
                type_v = v
                break

        if type_v != -1:
            attached_u = -1
            for v1, v2, e_label in aqg.edges:
                if v1 == type_v:
                    attached_u = v2
                    break
            assert attached_u != -1

            aqg.remove_edge(attached_u, type_v)
            aqg.vertices.remove(type_v)
            aqg.v_labels.pop(type_v)

            pred_obj_labels = [x for x in aqg.pred_obj_labels]
            type_label_index = -1
            for i, obj in enumerate(pred_obj_labels):
                if i % 3 == 1 and obj == 3:
                    type_label_index = i
            aqg.pred_obj_labels = pred_obj_labels[:type_label_index] + pred_obj_labels[type_label_index+3:]

    print(data["id"], len(grounding_res))
    # for x in grounding_res:
    #     print(x)
    return aqg

def generate_cand_queries(aqg, data, kb_endpoint):
    cand_vertices = {2: []}

    if data["entity1_uri"] != "":
        cand_vertices[2].append("<" + data["entity1_uri"] + ">")
    if data["entity2_uri"] != "":
        cand_vertices[2].append("<" + data["entity2_uri"] + ">")

    cand_vertices[3] = ["<" + x + ">" for x in data["cand_types"]]

    grounding_res = aqg.grounding(cand_vertices, kb_endpoint)
    return grounding_res

def check_relation(rel):
    if rel.find("http://dbpedia.org/property/") != -1 or \
        rel.find("http://dbpedia.org/ontology/") != -1:
        return True
    return False

def aeq(*args):
    base = args[0]
    for a in args[1:]:
        assert a == base, str(args)

def to_unicode(text, encoding='utf8', errors='strict'):
    """Convert a string (bytestring in `encoding` or unicode), to unicode.
    :param text:
    :param encoding:
    :param errors: errors can be 'strict', 'replace' or 'ignore' and defaults to 'strict'.
    """
    if isinstance(text, unicode):
        return text
    return unicode(text, encoding, errors=errors)


def load_glove_vocab(filename, vocab_size, binary=False, encoding='utf8', unicode_errors='ignore'):
    vocab = set()

    with open(filename, 'rb') as fin:
        # header = to_unicode(fin.readline(), encoding=encoding)
        # vocab_size, vector_size = map(int, header.split())  # throws for invalid file format

        if binary:
            for _ in range(vocab_size):
                # mixed text and binary: read text first, then binary
                word = []
                while True:
                    ch = fin.read(1)
                    if ch == b' ':
                        break
                    if ch != b'\n':  # ignore newlines in front of words (some binary files have)
                        word.append(ch)
                word = to_unicode(b''.join(word), encoding=encoding, errors=unicode_errors)
                vocab.add(word)
        else:
            for line_no, line in enumerate(fin):
                parts = to_unicode(line.rstrip(), encoding=encoding, errors=unicode_errors).split(" ")
                word = parts[0]
                vocab.add(word)
    return vocab


def load_word2vec_format(filename, word_idx, binary=False, normalize=False,
                         encoding='utf8', unicode_errors='ignore'):
    """
    load Word Embeddings
    If you trained the C model using non-utf8 encoding for words, specify that
    encoding in `encoding`.
    :param filename :
    :param word_idx :
    :param binary   : a boolean indicating whether the data is in binary word2vec format.
    :param normalize:
    :param encoding :
    :param unicode_errors: errors can be 'strict', 'replace' or 'ignore' and defaults to 'strict'.
    """
    vocab = set()

    with open(filename, 'rb') as fin:
        # header = to_unicode(fin.readline(), encoding=encoding)
        # vocab_size, vector_size = map(int, header.split())  # throws for invalid file format
        vocab_size = 400000
        vector_size = 300
        word_matrix = torch.zeros(len(word_idx), vector_size)

        def add_word(_word, _weights):
            if _word not in word_idx:
                return
            vocab.add(_word)
            word_matrix[word_idx[_word]] = _weights

        if binary:
            binary_len = np.dtype(np.float32).itemsize * vector_size
            for _ in range(vocab_size):
                # mixed text and binary: read text first, then binary
                word = []
                while True:
                    ch = fin.read(1)
                    if ch == b' ':
                        break
                    if ch != b'\n':  # ignore newlines in front of words (some binary files have)
                        word.append(ch)
                word = to_unicode(b''.join(word), encoding=encoding, errors=unicode_errors)
                weights = torch.from_numpy(np.fromstring(fin.read(binary_len), dtype=REAL))
                add_word(word, weights)
        else:
            for line_no, line in enumerate(fin):
                parts = to_unicode(line.rstrip(), encoding=encoding, errors=unicode_errors).split(" ")
                if len(parts) != vector_size + 1:
                    raise ValueError("invalid vector on line %s (is this really the text format?)" % line_no)
                word, weights = parts[0], list(map(float, parts[1:]))
                weights = torch.Tensor(weights)
                add_word(word, weights)
    if word_idx is not None:
        assert (len(word_idx), vector_size) == word_matrix.size()
    if normalize:
        # each row normalize to 1
        word_matrix = torch.renorm(word_matrix, 2, 0, 1)
    return word_matrix, vector_size, vocab