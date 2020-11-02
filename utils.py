#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 4
nmt.py: NMT Model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Vera Lin <veralin@stanford.edu>
"""

import math
from typing import List
import sys
import copy


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')


def pad_sents(sents, pad_token):
    """ Pad list of sentences according to the longest sentence in the batch.
        The paddings should be at the end of each sentence.
    @param sents (list[list[str]]): list of sentences, where each sentence
                                    is represented as a list of words
    @param pad_token (str): padding token
    @returns sents_padded (list[list[str]]): list of sentences where sentences shorter
        than the max length sentence are padded out with the pad_token, such that
        each sentences in the batch now has equal length.
    """
    sents_padded = []

    ### YOUR CODE HERE (~6 Lines)
    max_sent = 0
    for sent in sents:
        if len(sent) > max_sent:
            max_sent = len(sent)  
    for sent in sents:
        new_sent = copy.deepcopy(sent)
        new_sent.extend( [pad_token] * (max_sent - len(sent)))
        sents_padded.append(new_sent)

    ### END YOUR CODE

    return sents_padded


def read_corpus(file_path, source):
    """ Read file, where each sentence is dilineated by a `\n`.
    @param file_path (str): path to file containing corpus
    @param source (str): "tgt" or "src" indicating whether text
        is of the source language or target language
    """
    data = []
    for line in open(file_path):
        sent = nltk.word_tokenize(line)
        # only append <s> and </s> to the target sentence
        if source == 'tgt':
            sent = ['<s>'] + sent + ['</s>']
        data.append(sent)

    return data


def batch_iter(data, batch_size, shuffle=False):
    """ Yield batches of source and target sentences reverse sorted by length (largest to smallest).
    @param data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
    @param batch_size (int): batch size
    @param shuffle (boolean): whether to randomly shuffle the dataset
    """
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_sents = [e[0] for e in examples]
        tgt_sents = [e[1] for e in examples]

        yield src_sents, tgt_sents

def check_pad_sentences():

    sentence_1 = "Hello Mr. Smith, how are you doing today? The weather is great, and Python is awesome."
    sentence_2 = "The sky is pinkish-blue. You shouldn't eat cardboard."
    sentence_3 = "Hello Mr. Smith, how are you doing today? The weather is great, and Python"

    sents = []
    pad_token = '|' 
    sents.append(word_tokenize(sentence_1))
    sents.append(word_tokenize(sentence_2))
    sents.append(word_tokenize(sentence_3))

    new_sents = pad_sents(sents, pad_token)
    print(type(new_sents)) 
    for sent in new_sents:
        print(sent)
        assert len(sent) >= 18    

    print("check_pad_sentences test passed!")


if __name__ == "__main__":
    args = sys.argv
    if len(args) != 2:
        raise Exception("You did not provide a valid keyword. Either provide 'part_c' or 'part_d', when executing this script")
    elif args[1] == "check_pad":
        check_pad_sentences()



