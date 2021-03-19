# the Model
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from preprocess import PAD_ID, SOS_ID, create_vocab_class


class Encoder(nn.Module):
    def __init__(self,hidden_size,embedding_size,keep_prob):
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.gru = nn.GRU(embedding_size,hidden_size,dropout=1-keep_prob,bidirectional=True)
    
    def forward(self, embedded_inputs):
        output,hidden = self.gru(embedded_inputs)
        
        return output
        # output = [seq_len,batch_size,hidden_size*2]
        # hidden = [2,batch_size,hidden_size]
        
        # can add Dropout layer

# class Decoder(nn.Module):
#     def __init__(self, batch_size, hidden_size, tgt_vocab_size, max_decoder_length, embeddings, 
#                 keep_prob, sampling_prob, schedule_embed=False, pred_method='greedy'):
#         self.hidden_size = hidden_size
#         self.projection_layer = nn.Linear(hidden_size,tgt_vocab_size)
#         self.gru = nn.GRU(hidden_size,hidden_size)
#         self.batch_size = batch_size
#         self.embeddings = embeddings
#         self.start_id = SOS_ID
#         self.end_id = PAD_ID
#         self.tgt_vocab_size = tgt_vocab_size
#         self.max_decoder_length = max_decoder_length
#         self.keep_prob = keep_prob
#         self.schedule_embed = schedule_embed
#         self.pred_method = pred_method
#         self.beam_width = 9
#         self.sampling_prob = sampling_prob

#     def forward(self, blended_reps_final, encoder_hidden, decoder_emb_inputs, ans_masks, ans_ids, context_masks):
#         start_ids = ans_ids[:,0]
#         train_output = blended_reps_final
#         context_lengths = torch.Tensor([context_masks.size(1)]*self.batch_size)
#         decoder_lengths = torch.Tensor([context_masks.size(1)]*self.batch_size)

#         # traininghelper vali lines

#         pred_start_ids = ans_ids[:,0]

#         # pred_helper vali lines


class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, hidden, encoder_outputs, src_len=None):
        '''
        :param hidden: 
            previous hidden state of the decoder, in shape (layers*directions,B,H)
        :param encoder_outputs:
            encoder outputs from Encoder, in shape (T,B,H)
        :param src_len:
            used for masking. NoneType or tensor in shape (B) indicating sequence length
        :return
            attention energies in shape (B,T)
        '''
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)
        H = hidden.repeat(max_len,1,1).transpose(0,1)
        encoder_outputs = encoder_outputs.transpose(0,1) # [B*T*H]
        attn_energies = self.score(H,encoder_outputs) # compute attention score
        
        if src_len is not None:
            mask = []
            for b in range(src_len.size(0)):
                mask.append([0] * src_len[b].item() + [1] * (encoder_outputs.size(1) - src_len[b].item()))
            mask = cuda_(torch.ByteTensor(mask).unsqueeze(1)) # [B,1,T]
            attn_energies = attn_energies.masked_fill(mask, -1e18)
        
        return F.softmax(attn_energies).unsqueeze(1) # normalize with softmax

    def score(self, hidden, encoder_outputs):
        energy = F.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2))) # [B*T*2H]->[B*T*H]
        energy = energy.transpose(2,1) # [B*H*T]
        v = self.v.repeat(encoder_outputs.data.shape[0],1).unsqueeze(1) #[B*1*H]
        energy = torch.bmm(v,energy) # [B*1*T]
        return energy.squeeze(1) #[B*T]

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, embed_size, output_size, n_layers=1, dropout_p=0.1):
        super(DecoderRNN, self).__init__()
        # Define parameters
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        # Define layers
        self.embedding = nn.Embedding(output_size, embed_size)
        self.dropout = nn.Dropout(dropout_p)
        self.attn = Attn('concat', hidden_size)
        self.gru = nn.GRU(hidden_size + embed_size, hidden_size, n_layers, dropout=dropout_p)
        #self.attn_combine = nn.Linear(hidden_size + embed_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, word_input, last_hidden, encoder_outputs):
        '''
        :param word_input: === decoder_input
            word input for current time step, in shape (B)
        :param last_hidden:=== decoder_hidden
            last hidden stat of the decoder, in shape (layers*direction*B*H)
        :param encoder_outputs:
            encoder outputs in shape (T*B*H)
        :return
            decoder output
        Note: we run this one step at a time i.e. you should use a outer loop 
            to process the whole sequence
            '''
        # Get the embedding of the current input word (last output word)
        word_embedded = self.embedding(word_input).view(1, word_input.size(0), -1) # (1,B,V)
        word_embedded = self.dropout(word_embedded)
        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attn(last_hidden[-1], encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,V)
        context = context.transpose(0, 1)  # (1,B,V)
        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat((word_embedded, context), 2)
        #rnn_input = self.attn_combine(rnn_input) # use it in case your size of rnn_input is different
        output, hidden = self.gru(rnn_input, last_hidden)
        output = output.squeeze(0)  # (1,B,V)->(B,V)
        # context = context.squeeze(0)
        # update: "context" input before final layer can be problematic.
        # output = F.log_softmax(self.out(torch.cat((output, context), 1)))
        output = F.log_softmax(self.out(output))
        # Return final output, hidden state
        return output, hidden



class BasicAttn(nn.Module):
    def __init__(self, keep_prob, key_vec_size, value_vec_size):
        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size

    def forward(self, values, values_mask, keys):
        attn_logits_mask = torch.unsqueeze(values_mask, 1) # -> (batch_size, 1, num_values)
        
        w = torch.zeros(self.key_vec_size, self.value_vec_size)
        w = nn.init.xavier_normal_(w)

        values_t = torch.transpose(values, 1, 2) # -> (batch_size, value_vec_size, num_values)
        def fn(a, x):
            return torch.matmul(x, W).numpy()

        list_ = [fn(8, keys[i, :,:]) for i in range(keys.shape[0])]
        part_logits = torch.Tensor(list_) # (batch_size, num_keys, value_vec)

        attn_logits = torch.bmm(part_logits, values_t) # -> (batch_size, num_keys, num_values)
        _, attn_dist = masked_softmax(attn_logits, attn_logits_mask, dim)

        output = torch.matmul(attn_dist, values)

        return attn_dist, output


class QAModel(nn.Module):
    def __init__(self, id2word, word2id, emb_matrix, ans2id, id2ans, context2id,hidden_size, embedding_size, tgt_vocab_size):
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.tgt_vocab_size = tgt_vocab_size
        self.id2word = id2word
        self.word2id = word2id
        self.ans_vocab_size = len(ans2id)
        self.ans2id = ans2id
        self.id2ans = id2ans
        self.emb_matrix = emb_matrix
        self.context2id = context2id
        self.embedding = nn.Embedding(len(context2id),embedding_size)
        self.linear21 = nn.Linear(2*hidden_size,hidden_size)
        self.linear41 = nn.Linear(4*hidden_size,hidden_size)
        self.graph_vocab_class = create_vocab_class(context2id)
        self.context_dimension_compressed = len(self.graph_vocab_class.all_tokens) + len(self.graph_vocab_class.nodes)

        self.encoder = Encoder(self.hidden_size,self.embedding_size, self.keep_prob)
        self.decoder = DecoderRNN(hidden_size,embedding_size,tgt_vocab_size)

    def forward(self,qn_ids,context_ids,ans_ids,qn_mask):
        self.context_embs = self.embedding(context_ids)
        context_hiddens = self.encoder(self.context_embs)  # (batch_size, context_len, hidden_size*2)

        self.qn_embs = self.get_embeddings(self.ans2id,self.emb_matrix,qn_ids,self.embedding_size)
        question_hiddens = self.encoder(self.qn_embs)  # (batch_size, question_len, hidden_size*2)
        question_last_hidden = question_hiddens[:, -1, :]
        question_last_hidden = self.linear21(question_last_hidden)

        attn_layer = BasicAttn(self.keep_prob, self.hidden_size * 2, self.hidden_size * 2)
        _, attn_output = attn_layer(question_hiddens, qn_mask, context_hiddens)
        # Concat attn_output to context_hiddens to get blended_reps
        blended_reps = torch.cat(context_hiddens, attn_output, axis=2)  # (batch_size, context_len, hidden_size*4)
        blended_reps_final = self.linear41(blended_reps)
        self.dec_hidden = question_last_hidden
        # Idhar for loop lagaane ka hai
        decoder_outputs = []
        for idx in range(len(ans_ids)):
            self.dec_output,self.dec_hidden = self.decoder(self.ans_ids[idx],self.dec_hidden,blended_reps_final)
            decoder_outputs.append(self.dec_output)
            # topk
            # loss add
        return self.decoder_outputs #, loss
        
        # ----------------------------------- #
        
    def get_embeddings(self,token2id,embed_matrix,input_ids,embed_size):
        array = np.zeros(len(input_ids),embed_size)
        for idx,token_id in enumerate(input_ids):
            # either token or token ID, check that
            array[idx,:] = embed_matrix[token_id]
        vector = torch.from_numpy(array)
        return vector



def masked_softmax(logits, masks, dim):
    inf_mask = (1 - masks.type(torch.FloatTensor)) * (-1e30)
    masked_logits = torch.add(logits, inf_mask)
    softmax_out = nn.Softmax(masked_logits, dim)
    return masked_logits, softmax_out
