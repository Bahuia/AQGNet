# -*- coding: utf-8 -*-
# !/usr/bin/python
from six import iteritems
import codecs


def init_vocab(unk=None, pad=None):
    vocab = Dictionary()
    if unk is not None:
        vocab.add_unk_token(unk)
    else:
        vocab.add_unk_token()
    if pad is not None:
        vocab.add_pad_token(pad)
    else:
        vocab.add_pad_token()
    return vocab


class Dictionary(object):
    """
    a class for manage word dictionary by count/top setting.
    """
    def __init__(self, lower=True):
        # word -> index, index start from 0
        self.word2index = dict()
        # index -> word, index start from 0
        self.index2word = dict()
        # word -> count
        self.word_count = dict()
        self.lower = lower
        # special entries will not be pruned
        self.special = set()

    def __add__(self, d):
        """
        merge two dictionary
        """
        assert type(d) == Dictionary
        assert self.lower == d.lower

        word_set = set(self.word2index) | set(d.word_index)
        new_d = Dictionary(self.lower)
        new_d.special = self.special | d.special

        for w in word_set:
            new_d.add(w, count=self.lookup_count(w))
            new_d.add(w, count=d.lookup_count(w))
        return new_d

    def __getitem__(self, word):
        return self.word2index[word]

    def __contains__(self, word):
        return word in self.word2index

    def __len__(self):
        return len(self.word2index)

    def __iter__(self):
        for word in self.word2index:
            yield word

    def size(self):
        return len(self)

    def lookup(self, key, default=None):
        key = self.lower_(key)
        try:
            return self.word2index[key]
        except KeyError:
            return default

    def lower_(self, key):
        if isinstance(key, int):
            return key
        return key.lower() if self.lower else key

    def add(self, key, idx=None, count=1):
        """
        add word to dictionary
        :param idx: use 'idx' as its index if given.
        :param count: use 'count' as its count if given, default is 1.
        """
        key = self.lower_(key)
        if idx is not None:
            self.index2word[idx] = key
            self.word2index[key] = idx
        else:
            if key not in self.word2index:
                idx = len(self.word2index)
                self.index2word[idx] = key
                self.word2index[key] = idx
        if key not in self.word_count:
            self.word_count[key] = count
        else:
            self.word_count[key] += count

    def add_special(self, key, idx=None, count=1):
        self.add(key, idx, count)
        self.special.add(key)

    def add_specials(self, key, idxs):
        for key, idx in zip(key, idxs):
            self.add_special(key, idx=idx)

    def add_unk_token(self, unk_token='<unk>'):
        self.add_special(unk_token)
        self.unk_token = unk_token

    def add_pad_token(self, pad_token='<pad>'):
        self.add_special(pad_token)
        self.pad_token = pad_token

    def add_start_token(self, start_token='<s>'):
        self.add_special(start_token)
        self.start_token = start_token

    def add_end_token(self, end_token='</s>'):
        self.add_special(end_token)
        self.end_token = end_token

    def lookup_count(self, key):
        key = self.lower_(key)
        try:
            return self.word_count[key]
        except KeyError:
            return 0

    def sort(self, reverse=True):
        """
        sort dict by count
        :param reverse: default is True, high -> low
                                   False, low -> high
        """
        count_word = list()
        indexs = list()
        for w in self.word2index:
            if w in self.special:
                continue
            count_word.append((self.word_count[w], w))
            indexs.append(self.word2index[w])

        count_word.sort(reverse=reverse)
        indexs.sort(reverse=reverse)

        for index, (_, word) in zip(indexs, count_word):
            self.word2index[word] = index
            self.index2word[index] = word

    def clear_dictionary(self, keep_special=True):
        special_count_index = None
        if keep_special:
            special_count_index = [(word, self.word_count[word], self.word2index[word]) for word in self.special]
        else:
            self.special = set()
        self.word_count = dict()
        self.word2index = dict()
        self.index2word = dict()
        if special_count_index:
            for word, count, index in special_count_index:
                self.add_special(key=word, count=count, idx=index)

    def cut_by_top(self, top_k = 30000):
        """
        cut dictionary by top count
        """
        if len(self.word2index) <= top_k:
            print("Word number (%s) is samller Top k (%s)" % (len(self.word2index), top_k))
            return

        word_count = list()
        for word, count in iteritems(self.word_count):
            word_count.append((count, word))
        word_count.sort(reverse=True)

        self.clear_dictionary(keep_special=True)

        added_top_num = 0
        for count, word in word_count:
            if added_top_num >= top_k:
                break
            if word not in self.special:
                self.add(key=word, count=count)
                added_top_num += 1

        print("After cut, Dictionary Size is %d" % len(self))

    def write_to_file(self, filename):
        with codecs.open(filename, 'w', encoding='utf-8') as fout:
            for word, index in iteritems(self.word2index):
                write_str = "%s %s\n" % (word, index)
                fout.write(write_str.encode('utf8'))

    def convert_to_index(self, words, bos_word=None, eos_word=None):
        """
        convert 'word' to indices.
        :param unk_word: use 'unkword' if not found.
        :param bos_word: optionally insert 'bosword' at the beginning.
        :param eos_word: and 'eosword' at the end. 
        """
        vec = []
        if bos_word is not None:
            vec += [self.lookup(bos_word)]
        unk = self.lookup(self.unk_token)
        vec += [self.lookup(word, default=unk) for word in words]

        if eos_word is not None:
            vec += [self.lookup(eos_word)]
        return vec

    def convert_to_word(self, indexs):
        """
        convert 'indexs' to words.
        """
        return [self.index2word[index] for index in indexs]

    def contains(self, word):
        key = self.lower_(word)
        if key in self.word2index:
            return True
        else:
            return False
