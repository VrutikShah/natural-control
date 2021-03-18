import os
import json
import sys
import logging
# import torch
# from model import QAModel
# from train import train
import pickle

from preprocess import get_glove, create_vocabulary, create_vocab_class
logging.basicConfig(level=logging.INFO)

NUM_EPOCH = 100000
LEARNING_RATE =0.001
BATCH_SIZE = 128
HIDDEN_SIZE = 128
CONTEXT_LEN = 300
QUESTION_LEN = 150
ANSWER_LEN = 50
EMBEDDING_SIZE = 100
LOAD_PREV = False

# SAMPLING CONSTANTS

with open("./data/emb_matrix.pkl", "rb") as f:
    emb_matrix = pickle.load(f)

with open("./data/word2id.pkl", "rb") as f:
    word2id=  pickle.load(f)

with open("./data/id2word.pkl", "rb") as f:
    id2word = pickle.load(f)



context_vocab_path = "./data/vocab200.context"
train_context_path = "./data/train.graph"
context_vocab, rev_context_vocab = create_vocabulary(context_vocab_path,train_context_path,200)


# qa_model = QAModel(id2word, word2id, emb_matrix, context_vocab, rev_context_vocab, context_vocab, HIDDEN_SIZE, EMBEDDING_SIZE, NO_CLASS)
# file_handler = logging.FileHandler(os.path.join("./model", "log.txt"))
# logging.getLogger().addHandler(file_handler)

# if LOAD_PREV:
#     qa_model.load_state_dict(torch.load("./data"))

train(qa_model, NUM_EPOCH, "./data/train.graph", "./data/train.instructions", "./data/train.answer", BATCH_SIZE)
# TODO 
# qa_model.train(train_context_path, train_qn_path, train_ans_path, dev_qn_path, dev_context_path, dev_ans_path)