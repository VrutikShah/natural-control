"""This file contains code to read tokenized data from file,
truncate, pad and process it into batches ready for training"""

import random
import time
import re

import numpy as np
from preprocess import PAD_ID, UNK_ID, SOS_ID, Vocab, create_vocab_class, instruction_tokenizer

class Batch(object):
    """A class to hold the information needed for a training batch"""

    def __init__(self, context_ids, context_tokens, qn_ids, qn_mask, qn_tokens, ans_ids, ans_mask,
                 ans_tokens, batch_size):
        """
        Inputs:
          {context/qn}_ids: Numpy arrays.
            Shape (batch_size, {context_len/question_len}). Contains padding.
          {context/qn}_mask: Numpy arrays, same shape as _ids.
            Contains 1s where there is real data, 0s where there is padding.
          {context/qn/ans}_tokens: Lists length batch_size, containing lists (unpadded) of tokens (strings)
          ans_span: numpy array, shape (batch_size, 2)
        """
        self.context_ids = context_ids
        # self.context_mask = context_mask
        self.context_tokens = context_tokens
        # self.context_embeddings = context_embeddings

        self.qn_ids = qn_ids
        self.qn_mask = qn_mask
        self.qn_tokens = qn_tokens

        self.ans_ids = ans_ids
        self.ans_mask = ans_mask
        self.ans_tokens = ans_tokens

        self.batch_size = batch_size


def split_by_whitespace(sentence):
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(" ", space_separated_fragment))
    return [w for w in words if w]


def intstr_to_intlist(string):
    """Given a string e.g. '311 9 1334 635 6192 56 639', returns as a list of integers"""
    return [int(s) for s in string.split()]


def sentence_to_token_ids(sentence, word2id, is_instr=False):
    """Turns an already-tokenized sentence string into word indices
    e.g. "i do n't know" -> [9, 32, 16, 96]
    Note any token that isn't in the word2id mapping gets mapped to the id for UNK
    """
    if is_instr:
        tokens = instruction_tokenizer(sentence)  # list of strings
    else:
        tokens = split_by_whitespace(sentence)

    # if simply split in tokens.
    ids = [word2id.get(w, UNK_ID) for w in tokens]
    ''' for debugging
    if UNK_ID in ids:
        print(tokens[ids.index(UNK_ID)], " ".join(tokens))
    '''
    return tokens, ids

def padded(token_batch, batch_pad=0):
    """
    Inputs:
      token_batch: List (length batch size) of lists of ints.
      batch_pad: Int. Length to pad to. If 0, pad to maximum length sequence in token_batch.
    Returns:
      List (length batch_size) of padded of lists of ints.
        All are same length - batch_pad if batch_pad!=0, otherwise the maximum length in token_batch
    """
    maxlen = max(map(lambda x: len(x), token_batch)) if batch_pad == 0 else batch_pad
    return map(lambda token_list: token_list + [PAD_ID] * (maxlen - len(token_list)), token_batch)

def reorganize(context_line, ans_line):
    start = ans_line.strip().split()[0]
    context_trip_list = context_line.strip().split(';')
    trips_contain_start = []
    trips_not_contain_start = []

    for trip_str in context_trip_list:
        if start in trip_str:
            trips_contain_start.append(trip_str)
        else:
            trips_not_contain_start.append(trip_str)
    if trips_not_contain_start[0][0] != ' ':
        trips_not_contain_start[0] = ' ' + trips_not_contain_start[0]
    organized_context_line = ";".join(trips_contain_start + trips_not_contain_start).strip() + '\n'

    #assert len(organized_context_line) == len(context_line), "len {} {}{} len {}".\
    #      format(len(context_line), context_line, organized_context_line, len(organized_context_line))
    return organized_context_line


def refill_batches(batches, word2id, context2id, ans2id, context_file, qn_file, ans_file, batch_size, context_len,
                   question_len, ans_len, discard_long, shuffle=True, output_goal=False):
    """
    Adds more batches into the "batches" list.
    Inputs:
      batches: list to add batches to
      word2id: dictionary mapping word (string) to word id (int)
      context_file, qn_file, ans_file: paths to {train/dev}.{context/question/answer} data files
      batch_size: int. how big to make the batches
      context_len, question_len: max length of context and question respectively
      discard_long: If True, discard any examples that are longer than context_len or question_len.
        If False, truncate those exmaples instead.
    """
    print("Refilling batches...")
    tic = time.time()
    examples = []  # list of (qn_ids, context_ids, ans_span, ans_tokens) triples
    context_line, qn_line, ans_line = context_file.readline(), qn_file.readline(), ans_file.readline()  # read the next line from each
    # print(context_line,qn_line,ans_line)
    while context_line and qn_line and ans_line:  # while you haven't reached the end

        # Reorganize the map to make the nodes containing the start point comes at the front.
        context_line = reorganize(context_line, ans_line)
        # Convert tokens to word ids
        context_tokens, context_ids = sentence_to_token_ids(context_line, context2id)
        qn_tokens, qn_ids = sentence_to_token_ids(qn_line, word2id, is_instr=True)

        ans_tokens, ans_ids = sentence_to_token_ids(ans_line, ans2id)

        ############# reorganize ans tokens into [start] + [action list] (+ [end]) #####################
        if output_goal:
            ans_tokens = [ans_tokens[0]] + ans_tokens[1::2] + [ans_tokens[-1]]
            ans_ids = [ans_ids[0]] + ans_ids[1::2] + [ans_ids[-1]]
        else:
            ans_tokens = [ans_tokens[0]] + ans_tokens[1::2]
            ans_ids = [ans_ids[0]] + ans_ids[1::2]
        ##############################################################################################s
        

        # read the next line from each file
        context_line, qn_line, ans_line = context_file.readline(), qn_file.readline(), ans_file.readline()

        # discard or truncate too-long questions
        if len(qn_ids) > question_len:
            if discard_long:
                continue
            else:  # truncate
                qn_ids = qn_ids[:question_len]

        # discard or truncate too-long contexts
        if len(context_ids) > context_len:
            if discard_long:
                continue
            else:  # truncate
                context_ids = context_ids[:context_len]

        # discard or truncate too-long answer
        if len(ans_ids) > ans_len:
            if discard_long:
                continue
            else:  # truncate
                ans_ids = ans_ids[:ans_len]

        # add to examples
        examples.append((context_ids, context_tokens, qn_ids, qn_tokens, ans_ids, ans_tokens))

        # stop refilling if you have 160 batches
        if len(examples) == batch_size * 160:
            break

    # Once you've either got 160 batches or you've reached end of file:

    # Sort by context length for speed
    # Note: if you sort by context length, then you'll have batches which contain the same context many times
    # (because each context appears several times, with different questions)
    # shuffle==False means to not change the sequence of the input data, thus no sorting.
    if shuffle:
        examples = sorted(examples, key=lambda e: len(e[0]))

    # Make into batches and append to the list batches
    for batch_start in range(0, len(examples), batch_size):
        # Note: each of these is a list length batch_size of lists of ints (except on last iter when it might be less than batch_size)
        context_ids_batch, context_tokens_batch, qn_ids_batch, qn_tokens_batch, ans_span_batch, ans_tokens_batch = zip(*examples[batch_start:batch_start + batch_size])

        batches.append(
            (context_ids_batch, context_tokens_batch, qn_ids_batch, qn_tokens_batch, ans_span_batch, ans_tokens_batch))
    if shuffle:
        # shuffle the batches
        random.shuffle(batches)

    toc = time.time()
    print("Refilling batches took %.2f seconds" % (toc - tic))
    return


def get_batch_generator(word2id, context2id, ans2id, context_path, qn_path, ans_path, batch_size, graph_vocab_class,
                        context_len, question_len, answer_len, discard_long, shuffle=True, output_goal=False):
    
    context_file, qn_file, ans_file = open(context_path, encoding="utf-8"), open(qn_path, encoding="utf-8"), open(ans_path, encoding="utf-8")
    batches = []


    while True:
        if len(batches) == 0:  # add more batches
            refill_batches(batches, word2id, context2id, ans2id, context_file, qn_file, ans_file, batch_size,
                           context_len, question_len, answer_len, discard_long, shuffle=shuffle, output_goal=output_goal)
        if len(batches) == 0:
            break

        # Get next batch. These are all lists length batch_size
        (context_ids, context_tokens, qn_ids, qn_tokens, ans_ids, ans_tokens) = batches.pop(0)

        # Pad context_ids and qn_ids
        qn_ids = padded(qn_ids, question_len)  # pad questions to length question_len
        context_ids = padded(context_ids, context_len)  # pad contexts to length context_len
        ans_ids = padded(ans_ids, answer_len) # pad ans to maximum length

        # Make qn_ids into a np array and create qn_mask
        qn_ids = np.array(list(qn_ids))  # shape (batch_size, question_len)
        qn_mask = (qn_ids != PAD_ID).astype(np.int32)  # shape (batch_size, question_len)

        # Make context_ids into a np array and create context_mask
        context_ids = np.array(list(context_ids))  # shape (batch_size, context_len)
        # context_mask = (context_ids != PAD_ID).astype(np.int32)  # shape (batch_size, context_len)

        # Make ans_ids into a np array and create ans_mask
        ans_ids = np.array(list(ans_ids))
        ans_mask = (ans_ids != PAD_ID).astype(np.int32)
        # print(list(ans_ids), list(context_ids), list(qn_ids))
        # interpret graph as triplets and append the first token
        # if not show_start_tokens:
        # context_embeddings, context_mask = compute_graph_embedding(context_tokens, graph_vocab_class, context_mask.shape[1])
        # else:
        #     context_embeddings, context_mask = compute_graph_embedding(context_tokens, graph_vocab_class, context_mask.shape[1],
        #                                                     np.array([ans_token[0] for ans_token in ans_tokens]))
        
        # Make into a Batch object
        batch = Batch(context_ids, context_tokens, qn_ids, qn_mask, qn_tokens, ans_ids, ans_mask, ans_tokens, batch_size)
        # print(len(batch))
        yield batch

    return