import time
import torch
import torch.nn as nn
import torch.optim as optim

from data_batcher import get_batch_generator
from tqdm import tqdm
from test import compute_all_metrics
import logging

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging.basicConfig(filename="loss.logs", filemode="a", level=logging.INFO)

def val_iteration(qamodel, batch, criterion):
    context_ids = batch.context_ids
    qn_ids = batch.qn_ids
    ans_ids = batch.ans_ids
    qn_mask = batch.qn_mask
    batch_size = batch.batch_size

    context_ids = torch.from_numpy(context_ids).long().to(device)
    decoder_outputs= qamodel(qn_ids,context_ids,ans_ids,qn_mask, 1)
    loss = 0    #loss per batch
    # print(len(decoder_outputs))
    # preds = []
    ans_ids = torch.tensor(ans_ids).transpose(0,1).to(device)
    pred = ''
    gt = ''
    # print(ans_ids)
    # print(qamodel.id2ans)
    for idx,dec_out in enumerate(decoder_outputs):
        val,id_ = torch.topk(dec_out,1)
        gt_id_ = ans_ids[idx].item()

        token = qamodel.id2ans[id_.item()]
        gt_token = qamodel.id2ans[gt_id_]
        gt+= " " + gt_token
        pred+=" " + token    
        # print(dec_out.shape)
        # dec_out = [bsz,output_vocab_size]
        # print(ans_ids[idx],ans_ids[idx].shape)
        loss += criterion(dec_out,ans_ids[idx])
    logging.info(pred)
    logging.info(gt)
    logging.info('\n')
    # preds.append(pred)
    # compute_all_metrics(preds, )
    
    return pred,gt, loss.item()/batch_size




def train_iteration(qamodel,batch,criterion,encoder_optimizer,decoder_optimizer):
    context_ids = batch.context_ids
    qn_ids = batch.qn_ids
    ans_ids = batch.ans_ids
    qn_mask = batch.qn_mask
    batch_size = batch.batch_size

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

#     qn_ids = torch.from_numpy(qn_ids).long().to(device)
    context_ids = torch.from_numpy(context_ids).long().to(device)
#     ans_ids = torch.from_numpy(ans_ids).long().to(device)
#     qn_mask = torch.from_numpy(qn_mask).long().to(device)
    
#     print(f"qn_ids - {qn_ids.is_cuda}, context_ids - {context_ids.is_cuda}, ans_ids - {ans_ids.is_cuda}, qn_mask - {qn_mask.is_cuda}")
    decoder_outputs= qamodel(qn_ids,context_ids,ans_ids,qn_mask, batch_size)
    # print(decoder_outputs)
    loss = 0    #loss per batch
    # print(len(decoder_outputs))
    ans_ids = torch.tensor(ans_ids).transpose(0,1).to(device)
    ans_ids = torch.tensor(ans_ids).transpose(0,1).to(device)
    pred = ''
    gt = ''
    for idx,dec_out in enumerate(decoder_outputs):
        val,id_ = torch.topk(dec_out[0],1)
        gt_id_ = ans_ids[0, idx].item()

        token = qamodel.id2ans[id_.item()]
        gt_token = qamodel.id2ans[gt_id_]
        gt+= " " + gt_token
        pred+=" " + token
        # print(dec_out.shape)
        # dec_out = [bsz,output_vocab_size]
        # print(ans_ids[idx],ans_ids[idx].shape)
        # print("LOL")
        # loss += criterion(dec_out,ans_ids[idx])
        # if idx > 1:
        #     break

    logging.info(f"Pred - {pred}")
    logging.info(f"GT = {gt}")
    # loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return 0 #loss.item()/batch_size


def train(qamodel,num_epochs, context_path, qn_path, ans_path, batch_size, learning_rate):
    epoch = 0
    num_epochs=1
    criterion = nn.CrossEntropyLoss()
    encoder_optimizer = optim.Adam(qamodel.encoder.parameters(),lr = learning_rate)
    decoder_optimizer = optim.Adam(qamodel.decoder.parameters(),lr = learning_rate)
    # initialise optimiser

    # 

    while epoch<num_epochs:
        train_batches = get_batch_generator(qamodel.word2id, qamodel.context2id, qamodel.ans2id, context_path,
                                                qn_path, ans_path, batch_size, qamodel.graph_vocab_class,
                                                context_len=300, question_len=150,
                                                answer_len=50, discard_long=False)
        val_batches = get_batch_generator(qamodel.word2id, qamodel.context2id, qamodel.ans2id, "./data/dev.graph",
                                            "./data/dev.instruction", "./data/dev.answer",
                                            batch_size=1, graph_vocab_class=qamodel.graph_vocab_class,
                                            context_len=300, question_len=150,
                                            answer_len=50, discard_long=False)
        epoch+=1
        epoch_loss = 0
        epoch_start_time = time.time()
        num_iters = 0
        qamodel.train()
        for batch in (train_batches):
            
            if batch.qn_ids.shape[0]!=batch.batch_size:
                continue
            batch_loss  = train_iteration(qamodel,batch,criterion,encoder_optimizer,decoder_optimizer)
            num_iters += 1
            epoch_loss += batch_loss
            if num_iters%50==0:
                print(f'End of {num_iters} batches with loss = {batch_loss}')
            # except RuntimeError as e:
            #     print(e)
            #     continue
            # loss backward
            # optimiser step
            # if num_iters%print_every:

            # add line to print at print_every
        print('End of epoch',epoch,' | Loss = ',epoch_loss)
        epoch_end_time = time.time()
        time_of_epoch = epoch_end_time - epoch_start_time

        ckp = {
            'epoch': epoch,
            'state_dict': qamodel.state_dict(),
            'encoder_optimizer': encoder_optimizer.state_dict(),
            'decoder_optimizer': decoder_optimizer.state_dict(),
            'loss': epoch_loss,
        }

        filename = f"models/f_{epoch}.pt"
        torch.save(ckp, filename)

        # Validation
        idx = 0
        with torch.no_grad():    
            qamodel.eval()
            for batch in (val_batches):
                idx +=1
                
                if batch.qn_ids.shape[0]!=batch.batch_size:
                    continue
                if idx > 1:
                    break
                pred,gt,loss = val_iteration(qamodel,batch,criterion)
                f1, em, ed, res = compute_all_metrics(pred,gt)
                logging.info(loss)
                logging.info(f1)
                logging.info(ed)
                



            
# def validate():
#     for batch in tqdm(get_batch_generator(qamodel.word2id, qamodel.context2id, qamodel.ans2id, context_path,
#                                             qn_path, ans_path, batch_size, qamodel.graph_vocab_class,
#                                             context_len=300, question_len=150,
#                                             answer_len=50, discard_long=False),total=127):
#         compute_all_metrics(preds, true)
