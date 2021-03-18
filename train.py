# For Training Steps

import time
import torch
import torch.nn as nn

from data_batcher import get_batch_generator

loss_func = nn.CrossEntropyLoss()

def loss(loss_func,logits,target):
    return loss_func(logits,target)



def train_iteration(qamodel,batch):
    context_ids = batch.context_ids
    qn_ids = batch.qn_ids_batch
    ans_ids = batch.ans_ids
    _,_ = qamodel(ans_ids,context_ids)
    return _


def train(qamodel,num_epochs, context_path, qn_path, ans_path, batch_size):
    epoch = 0
    # initialise optimiser
    while epoch<num_epochs:
        epoch+=1
        epoch_start_time = time.time()
        num_iters = 0
        for batch in get_batch_generator(qamodel.word2id, qamodel.context2id, qamodel.ans2id, context_path,
                                            qn_path, ans_path, batch_size, qamodel.graph_vocab_class,
                                            context_len=300, question_len=150,
                                            answer_len=50, discard_long=False):
            _,_  = train_iteration(qamodel,batch)
            num_iters += 1
            # loss backward
            # optimiser step
            # if num_iters%print_every:

            # add line to print at print_every

        epoch_end_time = time.time()

            

