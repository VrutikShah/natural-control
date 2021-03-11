import os
import json
import sys
import logging

from preprocess import get_glove, create_vocabulary, create_vocab_class
logging.basicConfig(level=logging.INFO)


DATA_DIR = "./data"
model_DIR = "./models"

NUM_EPOCH = 100000
LEARNING_RATE =0.001
BATCH_SIZE = 128
HIDDEN_SIZE = 128
CONTEXT_LEN = 300
QUESTION_LEN = 150
ANSWER_LEN = 50
EMBEDDING_SIZE = 100

# SAMPLING CONSTANTS