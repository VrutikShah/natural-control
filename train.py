# For Training Steps

import time
import torch
import torch.nn as nn
import torch.optim as optim

from data_batcher import get_batch_generator

loss_func = nn.CrossEntropyLoss()

def loss(loss_func,logits,target):
    return loss_func(logits,target)



def train_iteration(qamodel,batch,criterion,encoder_optimizer,decoder_optimizer):
    context_ids = batch.context_ids
    qn_ids = batch.qn_ids
    ans_ids = batch.ans_ids
    qn_mask = batch.qn_mask
    batch_size = batch.batch_size

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    decoder_outputs= qamodel(qn_ids,context_ids,ans_ids,qn_mask)
    loss = 0    #loss per batch
    for idx,dec_out in enumerate(decoder_outputs):
        loss += criterion(dec_out,ans_ids[idx])
    
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item()/batch_size


def train(qamodel,num_epochs, context_path, qn_path, ans_path, batch_size):
    epoch = 0
    criterion = nn.CrossEntropyLoss()
    encoder_optimizer = optim.Adam(qamodel.encoder.parameters(),lr = 0.001)
    decoder_optimizer = optim.Adam(qamodel.decoder.parameters(),lr = 0.001)
    # initialise optimiser
    while epoch<num_epochs:
        epoch+=1
        epoch_loss = 0
        epoch_start_time = time.time()
        num_iters = 0
        for batch in get_batch_generator(qamodel.word2id, qamodel.context2id, qamodel.ans2id, context_path,
                                            qn_path, ans_path, batch_size, qamodel.graph_vocab_class,
                                            context_len=300, question_len=150,
                                            answer_len=50, discard_long=False):
            batch_loss  = train_iteration(qamodel,batch,criterion,encoder_optimizer,decoder_optimizer)
            num_iters += 1
            epoch_loss += batch_loss
        
            # loss backward
            # optimiser step
            # if num_iters%print_every:

            # add line to print at print_every
        epoch_end_time = time.time()
        time_of_epoch = epoch_end_time - epoch_start_time
        print(time_of_epoch)

            

