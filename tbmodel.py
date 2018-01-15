import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.sparse as sparse
import numpy as np
import utils as utils

class hw(nn.Module):
    """Highway layers

    args: 
        size: input and output dimension
        dropout_ratio: dropout ratio
    """
   
    def __init__(self, size, num_layers = 1, dropout_ratio = 0.5):
        super(hw, self).__init__()
        self.size = size
        self.num_layers = num_layers
        self.trans = nn.ModuleList()
        self.gate = nn.ModuleList()
        self.dropout = nn.Dropout(p=dropout_ratio)

        for i in range(num_layers):
            tmptrans = nn.Linear(size, size)
            tmpgate = nn.Linear(size, size)
            self.trans.append(tmptrans)
            self.gate.append(tmpgate)

    def rand_init(self):
        """
        random initialization
        """
        for i in range(self.num_layers):
            utils.init_linear(self.trans[i])
            utils.init_linear(self.gate[i])

    def forward(self, x):
        """
        update statics for f1 score

        args: 
            x (ins_num, hidden_dim): input tensor
        return:
            output tensor (ins_num, hidden_dim)
        """
        
        
        g = nn.functional.sigmoid(self.gate[0](x))
        h = nn.functional.relu(self.trans[0](x))
        x = g * h + (1 - g) * x

        for i in range(1, self.num_layers):
            x = self.dropout(x)
            g = nn.functional.sigmoid(self.gate[i](x))
            h = nn.functional.relu(self.trans[i](x))
            x = g * h + (1 - g) * x

        return x

class CRF_L(nn.Module):
    """Conditional Random Field (CRF) layer. This version is used in Ma et al. 2016, has more parameters than CRF_S
 
    args: 
        hidden_dim : input dim size 
        tagset_size: target_set_size 
        if_biase: whether allow bias in linear trans    
    """
    

    def __init__(self, hidden_dim, tagset_size, if_bias=True):
        super(CRF_L, self).__init__()
        self.tagset_size = tagset_size
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size * self.tagset_size, bias=if_bias)

    def rand_init(self):
        """random initialization
        """
        utils.init_linear(self.hidden2tag)

    def forward(self, feats):
        """
        args: 
            feats (batch_size, seq_len, hidden_dim) : input score from previous layers
        return:
            output from crf layer (batch_size, seq_len, tag_size, tag_size)
        """
        return self.hidden2tag(feats).view(-1, self.tagset_size, self.tagset_size)

class CRF_S(nn.Module):
    """Conditional Random Field (CRF) layer. This version is used in Lample et al. 2016, has less parameters than CRF_L.

    args: 
        hidden_dim: input dim size
        tagset_size: target_set_size
        if_biase: whether allow bias in linear trans
 
    """
    
    def __init__(self, hidden_dim, tagset_size, if_bias=True):
        super(CRF_S, self).__init__()
        self.tagset_size = tagset_size
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size, bias=if_bias)
        self.transitions = nn.Parameter(torch.Tensor(self.tagset_size, self.tagset_size))

    def rand_init(self):
        """random initialization
        """
        utils.init_linear(self.hidden2tag)
        self.transitions.data.zero_()

    def forward(self, feats):
        """
        args: 
            feats (batch_size, seq_len, hidden_dim) : input score from previous layers
        return:
            output from crf layer ( (batch_size * seq_len), tag_size, tag_size)
        """
        
        scores = self.hidden2tag(feats).view(-1, self.tagset_size, 1)
        ins_num = scores.size(0)
        crf_scores = scores.expand(ins_num, self.tagset_size, self.tagset_size) + self.transitions.view(1, self.tagset_size, self.tagset_size).expand(ins_num, self.tagset_size, self.tagset_size)

        return crf_scores
        
class CRFRepack:
    """Packer for word level model
    
    args:
        tagset_size: target_set_size
        if_cuda: whether use GPU
    """

    def __init__(self, tagset_size, if_cuda):
        
        self.tagset_size = tagset_size
        self.if_cuda = if_cuda

    def repack_vb(self, feature, target, mask):
        """packer for viterbi loss

        args: 
            feature (Seq_len, Batch_size): input feature
            target (Seq_len, Batch_size): output target
            mask (Seq_len, Batch_size): padding mask
        return:
            feature (Seq_len, Batch_size), target (Seq_len, Batch_size), mask (Seq_len, Batch_size)
        """
        
        if self.if_cuda:
            fea_v = autograd.Variable(feature.transpose(0, 1)).cuda()
            tg_v = autograd.Variable(target.transpose(0, 1)).unsqueeze(2).cuda()
            mask_v = autograd.Variable(mask.transpose(0, 1)).cuda()
        else:
            fea_v = autograd.Variable(feature.transpose(0, 1))
            tg_v = autograd.Variable(target.transpose(0, 1)).contiguous().unsqueeze(2)
            mask_v = autograd.Variable(mask.transpose(0, 1)).contiguous()
        return fea_v, tg_v, mask_v

    def repack_gd(self, feature, target, current):
        """packer for greedy loss

        args: 
            feature (Seq_len, Batch_size): input feature
            target (Seq_len, Batch_size): output target
            current (Seq_len, Batch_size): current state
        return:
            feature (Seq_len, Batch_size), target (Seq_len * Batch_size), current (Seq_len * Batch_size, 1, 1)
        """
        if self.if_cuda:
            fea_v = autograd.Variable(feature.transpose(0, 1)).cuda()
            ts_v = autograd.Variable(target.transpose(0, 1)).cuda().view(-1)
            cs_v = autograd.Variable(current.transpose(0, 1)).cuda().view(-1, 1, 1)
        else:
            fea_v = autograd.Variable(feature.transpose(0, 1))
            ts_v = autograd.Variable(target.transpose(0, 1)).contiguous().view(-1)
            cs_v = autograd.Variable(current.transpose(0, 1)).contiguous().view(-1, 1, 1)
        return fea_v, ts_v, cs_v

    def convert_for_eval(self, target):
        """convert target to original decoding

        args: 
            target: input labels used in training
        return:
            output labels used in test
        """
        return target % self.tagset_size

class CRFRepack_WC:
    """Packer for model with char-level and word-level

    args:
        tagset_size: target_set_size
        if_cuda: whether use GPU
        
    """

    def __init__(self, tagset_size, if_cuda):
        
        self.tagset_size = tagset_size
        self.if_cuda = if_cuda

    def repack_vb(self, f_f, f_p, b_f, b_p, w_f, target, mask, len_b):
        """packer for viterbi loss

        args: 
            f_f (Char_Seq_len, Batch_size) : forward_char input feature 
            f_p (Word_Seq_len, Batch_size) : forward_char input position
            b_f (Char_Seq_len, Batch_size) : backward_char input feature
            b_p (Word_Seq_len, Batch_size) : backward_char input position
            w_f (Word_Seq_len, Batch_size) : input word feature
            target (Seq_len, Batch_size) : output target
            mask (Word_Seq_len, Batch_size) : padding mask
            len_b (Batch_size, 2) : length of instances in one batch
        return:
            f_f (Char_Reduced_Seq_len, Batch_size), f_p (Word_Reduced_Seq_len, Batch_size), b_f (Char_Reduced_Seq_len, Batch_size), b_p (Word_Reduced_Seq_len, Batch_size), w_f (size Word_Seq_Len, Batch_size), target (Reduced_Seq_len, Batch_size), mask  (Word_Reduced_Seq_len, Batch_size)

        """
        mlen, _ = len_b.max(0)
        mlen = mlen.squeeze()
        ocl = b_f.size(1)
        if self.if_cuda:
            f_f = autograd.Variable(f_f[:, 0:mlen[0]].transpose(0, 1)).cuda()
            f_p = autograd.Variable(f_p[:, 0:mlen[1]].transpose(0, 1)).cuda()
            b_f = autograd.Variable(b_f[:, -mlen[0]:].transpose(0, 1)).cuda()
            b_p = autograd.Variable((b_p[:, 0:mlen[1]] - ocl + mlen[0]).transpose(0, 1)).cuda()
            w_f = autograd.Variable(w_f[:, 0:mlen[1]].transpose(0, 1)).cuda()
            tg_v = autograd.Variable(target[:, 0:mlen[1]].transpose(0, 1)).unsqueeze(2).cuda()
            mask_v = autograd.Variable(mask[:, 0:mlen[1]].transpose(0, 1)).cuda()
        else:
            f_f = autograd.Variable(f_f[:, 0:mlen[0]].transpose(0, 1))
            f_p = autograd.Variable(f_p[:, 0:mlen[1]].transpose(0, 1))
            b_f = autograd.Variable(b_f[:, -mlen[0]:].transpose(0, 1))
            b_p = autograd.Variable((b_p[:, 0:mlen[1]] - ocl + mlen[0]).transpose(0, 1))
            w_f = autograd.Variable(w_f[:, 0:mlen[1]].transpose(0, 1))
            tg_v = autograd.Variable(target[:, 0:mlen[1]].transpose(0, 1)).unsqueeze(2)
            mask_v = autograd.Variable(mask[:, 0:mlen[1]].transpose(0, 1)).contiguous()
        return f_f, f_p, b_f, b_p, w_f, tg_v, mask_v

    def convert_for_eval(self, target):
        """convert for eval

        args: 
            target: input labels used in training
        return:
            output labels used in test
        """
        return target % self.tagset_size

class CRFLoss_gd(nn.Module):
    """loss for greedy decode loss, i.e., although its for CRF Layer, we calculate the loss as 

    .. math::
        \sum_{j=1}^n \log (p(\hat{y}_{j+1}|z_{j+1}, \hat{y}_{j}))

    instead of 
    
    .. math::
        \sum_{j=1}^n \log (\phi(\hat{y}_{j-1}, \hat{y}_j, \mathbf{z}_j)) - \log (\sum_{\mathbf{y}' \in \mathbf{Y}(\mathbf{Z})} \prod_{j=1}^n \phi(y'_{j-1}, y'_j, \mathbf{z}_j) )

    args:
        tagset_size: target_set_size
        start_tag: ind for <start>
        end_tag: ind for <pad>
        average_batch: whether average the loss among batch
    
    """

    def __init__(self, tagset_size, start_tag, end_tag, average_batch=True):
        super(CRFLoss_gd, self).__init__()
        self.tagset_size = tagset_size
        self.average_batch = average_batch
        self.crit = nn.CrossEntropyLoss(size_average=self.average_batch)

    def forward(self, scores, target, current):
        """
        args: 
            scores (Word_Seq_len, Batch_size, target_size_from, target_size_to): crf scores
            target (Word_Seq_len, Batch_size): golden list
            current (Word_Seq_len, Batch_size): current state
        return:
            crf greedy loss
        """
        ins_num = current.size(0)
        current = current.expand(ins_num, 1, self.tagset_size)
        scores = scores.view(ins_num, self.tagset_size, self.tagset_size)
        current_score = torch.gather(scores, 1, current).squeeze()
        return self.crit(current_score, target)

class CRFLoss_vb(nn.Module):
    """loss for viterbi decode

    .. math::
        \sum_{j=1}^n \log (\phi(\hat{y}_{j-1}, \hat{y}_j, \mathbf{z}_j)) - \log (\sum_{\mathbf{y}' \in \mathbf{Y}(\mathbf{Z})} \prod_{j=1}^n \phi(y'_{j-1}, y'_j, \mathbf{z}_j) )

    args:
        tagset_size: target_set_size
        start_tag: ind for <start>
        end_tag: ind for <pad>
        average_batch: whether average the loss among batch
        
    """

    def __init__(self, tagset_size, start_tag, end_tag, average_batch=True):
        super(CRFLoss_vb, self).__init__()
        self.tagset_size = tagset_size
        self.start_tag = start_tag
        self.end_tag = end_tag
        self.average_batch = average_batch

    def forward(self, scores, target, mask):
        """
        args:
            scores (seq_len, bat_size, target_size_from, target_size_to) : crf scores
            target (seq_len, bat_size, 1) : golden state
            mask (size seq_len, bat_size) : mask for padding
        return:
            loss
        """

        # calculate batch size and seq len
        seq_len = scores.size(0)
        bat_size = scores.size(1)

        # calculate sentence score
        tg_energy = torch.gather(scores.view(seq_len, bat_size, -1), 2, target).view(seq_len, bat_size)  # seq_len * bat_size
        tg_energy = tg_energy.masked_select(mask).sum()

        # calculate forward partition score

        # build iter
        seq_iter = enumerate(scores)
        # the first score should start with <start>
        _, inivalues = seq_iter.__next__()  # bat_size * from_target_size * to_target_size
        # only need start from start_tag
        partition = inivalues[:, self.start_tag, :].clone()  # bat_size * to_target_size
        # iter over last scores
        for idx, cur_values in seq_iter:
            # previous to_target is current from_target
            # partition: previous results log(exp(from_target)), #(batch_size * from_target)
            # cur_values: bat_size * from_target * to_target
            cur_values = cur_values + partition.contiguous().view(bat_size, self.tagset_size, 1).expand(bat_size, self.tagset_size, self.tagset_size)
            cur_partition = utils.log_sum_exp(cur_values, self.tagset_size)
            # (bat_size * from_target * to_target) -> (bat_size * to_target)
            # partition = utils.switch(partition, cur_partition, mask[idx].view(bat_size, 1).expand(bat_size, self.tagset_size)).view(bat_size, -1)
            mask_idx = mask[idx, :].view(bat_size, 1).expand(bat_size, self.tagset_size)
            partition.masked_scatter_(mask_idx, cur_partition.masked_select(mask_idx))  #0 for partition, 1 for cur_partition
            
        #only need end at end_tag
        partition = partition[:, self.end_tag].sum()
        # average = mask.sum()

        # average_batch
        if self.average_batch:
            loss = (partition - tg_energy) / bat_size
        else:
            loss = (partition - tg_energy)

        return loss

class CRFDecode_vb():
    """Batch-mode viterbi decode

    args:
        tagset_size: target_set_size
        start_tag: ind for <start>
        end_tag: ind for <pad>
        average_batch: whether average the loss among batch
        
    """

    def __init__(self, tagset_size, start_tag, end_tag, average_batch=True):
        self.tagset_size = tagset_size
        self.start_tag = start_tag
        self.end_tag = end_tag
        self.average_batch = average_batch

    def decode(self, scores, mask):
        """Find the optimal path with viterbe decode

        args:
            scores (size seq_len, bat_size, target_size_from, target_size_to) : crf scores 
            mask (seq_len, bat_size) : mask for padding
        return:
            decoded sequence (size seq_len, bat_size)
        """
        # calculate batch size and seq len

        seq_len = scores.size(0)
        bat_size = scores.size(1)

        mask = 1 - mask
        decode_idx = torch.LongTensor(seq_len-1, bat_size)

        # calculate forward score and checkpoint

        # build iter
        seq_iter = enumerate(scores)
        # the first score should start with <start>
        _, inivalues = seq_iter.__next__()  # bat_size * from_target_size * to_target_size
        # only need start from start_tag
        forscores = inivalues[:, self.start_tag, :]  # bat_size * to_target_size
        back_points = list()
        # iter over last scores
        for idx, cur_values in seq_iter:
            # previous to_target is current from_target
            # partition: previous results log(exp(from_target)), #(batch_size * from_target)
            # cur_values: bat_size * from_target * to_target
            cur_values = cur_values + forscores.contiguous().view(bat_size, self.tagset_size, 1).expand(bat_size, self.tagset_size, self.tagset_size)

            forscores, cur_bp = torch.max(cur_values, 1)
            cur_bp.masked_fill_(mask[idx].view(bat_size, 1).expand(bat_size, self.tagset_size), self.end_tag)
            back_points.append(cur_bp)

        pointer = back_points[-1][:, self.end_tag]
        decode_idx[-1] = pointer
        for idx in range(len(back_points)-2, -1, -1):
            pointer = torch.gather(back_points[idx], 1, pointer.contiguous().view(bat_size, 1))
            decode_idx[idx] = pointer
        return decode_idx


class LM_LSTM_CRF(nn.Module):
    """LM_LSTM_CRF model

    args: 
        tagset_size: size of label set
        char_size: size of char dictionary
        char_dim: size of char embedding
        char_hidden_dim: size of char-level lstm hidden dim
        char_rnn_layers: number of char-level lstm layers
        embedding_dim: size of word embedding
        word_hidden_dim: size of word-level blstm hidden dim
        word_rnn_layers: number of word-level lstm layers
        vocab_size: size of word dictionary
        dropout_ratio: dropout ratio
        large_CRF: use CRF_L or not, refer model.crf.CRF_L and model.crf.CRF_S for more details
        if_highway: use highway layers or not
        in_doc_words: number of words that occurred in the corpus (used for language model prediction)
        highway_layers: number of highway layers
    """
    
    def __init__(self, tagset_size, char_size, char_dim, char_hidden_dim, char_rnn_layers, embedding_dim, word_hidden_dim, word_rnn_layers, vocab_size, dropout_ratio, large_CRF=True, if_highway = False, in_doc_words = 2, highway_layers = 1):

        super(LM_LSTM_CRF, self).__init__()
        #self.stack_lstm = nn.LSTM(embedding_dim + char_hidden_dim * 2, word_hidden_dim // 2, num_layers=word_rnn_layers, bidirectional=True, dropout=dropout_ratio)
        #self.buffer_lstm = nn.LSTM(embedding_dim + char_hidden_dim * 2, word_hidden_dim // 2, num_layers=word_rnn_layers, bidirectional=True, dropout=dropout_ratio)
        #self.output_lstm = nn.LSTM(embedding_dim + char_hidden_dim * 2, word_hidden_dim // 2, num_layers=word_rnn_layers, bidirectional=True, dropout=dropout_ratio)
        #self.action_lstm = nn.LSTM(embedding_dim + char_hidden_dim * 2, word_hidden_dim // 2, num_layers=word_rnn_layers, bidirectional=True, dropout=dropout_ratio)

        self.char_dim = char_dim
        self.char_hidden_dim = char_hidden_dim
        self.char_size = char_size
        self.word_dim = embedding_dim
        self.word_hidden_dim = word_hidden_dim
        self.word_size = vocab_size
        self.if_highway = if_highway

        self.char_embeds = nn.Embedding(char_size, char_dim)
        self.forw_char_lstm = nn.LSTM(char_dim, char_hidden_dim, num_layers=char_rnn_layers, bidirectional=False, dropout=dropout_ratio)
        self.back_char_lstm = nn.LSTM(char_dim, char_hidden_dim, num_layers=char_rnn_layers, bidirectional=False, dropout=dropout_ratio)
        self.char_rnn_layers = char_rnn_layers

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)

        self.word_lstm = nn.LSTM(embedding_dim + char_hidden_dim * 2, word_hidden_dim // 2, num_layers=word_rnn_layers, bidirectional=True, dropout=dropout_ratio)

        self.word_rnn_layers = word_rnn_layers

        self.dropout = nn.Dropout(p=dropout_ratio)

        self.tagset_size = tagset_size
        if large_CRF:
            self.crf = CRF_L(word_hidden_dim, tagset_size)
        else:
            self.crf = CRF_S(word_hidden_dim, tagset_size)

        if if_highway:
            self.forw2char = hw(char_hidden_dim, num_layers=highway_layers, dropout_ratio=dropout_ratio)
            self.back2char = hw(char_hidden_dim, num_layers=highway_layers, dropout_ratio=dropout_ratio)
            self.forw2word = hw(char_hidden_dim, num_layers=highway_layers, dropout_ratio=dropout_ratio)
            self.back2word = hw(char_hidden_dim, num_layers=highway_layers, dropout_ratio=dropout_ratio)
            self.fb2char = hw(2 * char_hidden_dim, num_layers=highway_layers, dropout_ratio=dropout_ratio)

        self.char_pre_train_out = nn.Linear(char_hidden_dim, char_size)
        self.word_pre_train_out = nn.Linear(char_hidden_dim, in_doc_words)

        self.batch_size = 1
        self.word_seq_length = 1

    def set_batch_size(self, bsize):
        """
        set batch size
        """
        self.batch_size = bsize

    def set_batch_seq_size(self, sentence):
        """
        set batch size and sequence length
        """
        tmp = sentence.size()
        self.word_seq_length = tmp[0]
        self.batch_size = tmp[1]

    def rand_init_embedding(self):
        """
        random initialize char-level embedding
        """
        utils.init_embedding(self.char_embeds.weight)

    def load_pretrained_word_embedding(self, pre_word_embeddings):
        """
        load pre-trained word embedding

        args:
            pre_word_embeddings (self.word_size, self.word_dim) : pre-trained embedding
        """
        assert (pre_word_embeddings.size()[1] == self.word_dim)
        self.word_embeds.weight = nn.Parameter(pre_word_embeddings)

    def rand_init(self, init_char_embedding=True, init_word_embedding=False):
        """
        random initialization

        args:
            init_char_embedding: random initialize char embedding or not
            init_word_embedding: random initialize word embedding or not
        """
        
        if init_char_embedding:
            utils.init_embedding(self.char_embeds.weight)
        if init_word_embedding:
            utils.init_embedding(self.word_embeds.weight)
        if self.if_highway:
            self.forw2char.rand_init()
            self.back2char.rand_init()
            self.forw2word.rand_init()
            self.back2word.rand_init()
            self.fb2char.rand_init()
        utils.init_lstm(self.forw_char_lstm)
        utils.init_lstm(self.back_char_lstm)
        utils.init_lstm(self.word_lstm)
        utils.init_linear(self.char_pre_train_out)
        utils.init_linear(self.word_pre_train_out)
        self.crf.rand_init()

    def word_pre_train_forward(self, sentence, position, hidden=None):
        """
        output of forward language model

        args:
            sentence (char_seq_len, batch_size): char-level representation of sentence
            position (word_seq_len, batch_size): position of blank space in char-level representation of sentence
            hidden: initial hidden state

        return:
            language model output (word_seq_len, in_doc_word), hidden
        """
        
        embeds = self.char_embeds(sentence)
        d_embeds = self.dropout(embeds)
        lstm_out, hidden = self.forw_char_lstm(d_embeds)

        tmpsize = position.size()
        position = position.unsqueeze(2).expand(tmpsize[0], tmpsize[1], self.char_hidden_dim)
        select_lstm_out = torch.gather(lstm_out, 0, position)
        d_lstm_out = self.dropout(select_lstm_out).view(-1, self.char_hidden_dim)

        if self.if_highway:
            char_out = self.forw2word(d_lstm_out)
            d_char_out = self.dropout(char_out)
        else:
            d_char_out = d_lstm_out

        pre_score = self.word_pre_train_out(d_char_out)
        return pre_score, hidden

    def word_pre_train_backward(self, sentence, position, hidden=None):
        """
        output of backward language model

        args:
            sentence (char_seq_len, batch_size): char-level representation of sentence (inverse order)
            position (word_seq_len, batch_size): position of blank space in inversed char-level representation of sentence
            hidden: initial hidden state

        return:
            language model output (word_seq_len, in_doc_word), hidden
        """
        embeds = self.char_embeds(sentence)
        d_embeds = self.dropout(embeds)
        lstm_out, hidden = self.back_char_lstm(d_embeds)
        
        tmpsize = position.size()
        position = position.unsqueeze(2).expand(tmpsize[0], tmpsize[1], self.char_hidden_dim)
        select_lstm_out = torch.gather(lstm_out, 0, position)
        d_lstm_out = self.dropout(select_lstm_out).view(-1, self.char_hidden_dim)

        if self.if_highway:
            char_out = self.back2word(d_lstm_out)
            d_char_out = self.dropout(char_out)
        else:
            d_char_out = d_lstm_out

        pre_score = self.word_pre_train_out(d_char_out)
        return pre_score, hidden

    def forward(self, forw_sentence, forw_position, back_sentence, back_position, word_seq, hidden=None):
        '''
        args:
            forw_sentence (char_seq_len, batch_size) : char-level representation of sentence
            forw_position (word_seq_len, batch_size) : position of blank space in char-level representation of sentence
            back_sentence (char_seq_len, batch_size) : char-level representation of sentence (inverse order)
            back_position (word_seq_len, batch_size) : position of blank space in inversed char-level representation of sentence
            word_seq (word_seq_len, batch_size) : word-level representation of sentence
            hidden: initial hidden state

        return:
            crf output (word_seq_len, batch_size, tag_size, tag_size), hidden
        '''

        self.set_batch_seq_size(forw_position)

        #embedding layer
        forw_emb = self.char_embeds(forw_sentence)
        back_emb = self.char_embeds(back_sentence)

        #dropout
        d_f_emb = self.dropout(forw_emb)
        d_b_emb = self.dropout(back_emb)

        #forward the whole sequence
        forw_lstm_out, _ = self.forw_char_lstm(d_f_emb)#seq_len_char * batch * char_hidden_dim

        back_lstm_out, _ = self.back_char_lstm(d_b_emb)#seq_len_char * batch * char_hidden_dim

        #select predict point
        forw_position = forw_position.unsqueeze(2).expand(self.word_seq_length, self.batch_size, self.char_hidden_dim)
        select_forw_lstm_out = torch.gather(forw_lstm_out, 0, forw_position)

        back_position = back_position.unsqueeze(2).expand(self.word_seq_length, self.batch_size, self.char_hidden_dim)
        select_back_lstm_out = torch.gather(back_lstm_out, 0, back_position)

        fb_lstm_out = self.dropout(torch.cat((select_forw_lstm_out, select_back_lstm_out), dim=2))
        if self.if_highway:
            char_out = self.fb2char(fb_lstm_out)
            d_char_out = self.dropout(char_out)
        else:
            d_char_out = fb_lstm_out

        #word
        word_emb = self.word_embeds(word_seq)
        d_word_emb = self.dropout(word_emb)

        #combine
        word_input = torch.cat((d_word_emb, d_char_out), dim = 2)

        #word level lstm
        lstm_out, _ = self.word_lstm(word_input)
        d_lstm_out = self.dropout(lstm_out)

        #convert to crf
        crf_out = self.crf(d_lstm_out)
        crf_out = crf_out.view(self.word_seq_length, self.batch_size, self.tagset_size, self.tagset_size)
        
        return crf_out