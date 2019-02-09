import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class AttentionalBiLSTM(nn.Module):
    def __init__(self, information_center):
        self.information_center = information_center
        self.model_type = information_center.model_type
        self.num_usr = information_center.num_usr
        self.num_prd = information_center.num_prd
        self.emb_size = information_center.emb_size
        self.each_state = int(self.information_center.state_size/2)
        super(AttentionalBiLSTM, self).__init__()
        if self.model_type == 'encoder_cust':
            print(" Out-Of-Memory")
            raise NotImplementedError
        else:
            self.rnn = nn.LSTM(
                            input_size=self.emb_size, 
                            hidden_size=self.each_state, 
                            num_layers=1, 
                            bias=True, 
                            batch_first=True,
                            bidirectional=True)

        if self.model_type == 'attention_cust':
            # ADDITIONAL INFORMATION
            self.att_usr = nn.Embedding(self.num_usr, self.each_state*2)
            self.att_prd = nn.Embedding(self.num_prd, self.each_state*2)
            torch.nn.init.uniform_(self.att_usr.weight, -0.01, 0.01)
            torch.nn.init.uniform_(self.att_prd.weight, -0.01, 0.01)
        else:
            self.z = nn.Linear(self.each_state*2, 1, bias=False)
            torch.nn.init.xavier_uniform_(self.z.weight)

    def forward(self, x_batch, x_lens_batch, usr_batch, prd_batch):
        batch_size = x_batch.size(0)
        if self.model_type == 'encoder_cust':
            print(" Out-Of-Memory")
            raise NotImplementedError
        else:
            x_pack = nn.utils.rnn.pack_padded_sequence(x_batch, x_lens_batch, batch_first=True)
            (h0, c0) = self.information_center.to_var(torch.zeros((2, 2, batch_size, self.each_state)))
            hiddens, _ = self.rnn(x_pack, (h0, c0))
            hiddens, _ = nn.utils.rnn.pad_packed_sequence(hiddens, batch_first=True)

        if self.model_type == 'attention_cust':
            # ADDITIONAL INFORMATION
            usr_batch = self.att_usr(usr_batch)
            prd_batch = self.att_prd(prd_batch)
            meta_batch = torch.cat([usr_batch, prd_batch], dim=1)
        else:
            meta_batch = None
        
        hidden_reduced = self.reduce_attention(hiddens, x_lens_batch, meta_batch)
        return hidden_reduced

    def reduce_attention(self, hiddens_batch, lens_batch, meta_batch):
        batch_size = hiddens_batch.size(0)
        x = None
        for i in range(batch_size):
            lens = lens_batch[i] # length
            hidden = hiddens_batch[i][:lens] # length, state_size(=each_state*2)
            if self.model_type == 'attention_cust':
                energy = torch.mm(hidden.repeat(1, 2), meta_batch[i].unsqueeze(dim=1)).t()
            else:
                energy = self.z(hidden).t()
            attention = F.softmax(energy, dim=1)
            hidden_reduced = torch.mm(attention, hidden)
            x = hidden_reduced if (x is None) else torch.cat([x, hidden_reduced], dim=0)
        return x

class  cust(nn.Module):
    def __init__(self, information_center):
        self.information_center = information_center
        self.model_type = self.information_center.model_type
        self.num_usr = self.information_center.num_usr
        self.num_prd = self.information_center.num_prd
        self.num_label = self.information_center.num_label
        self.state_size = self.information_center.state_size
        self.emb_size = self.information_center.emb_size
        super( cust, self).__init__()

        self.embed = nn.Embedding(information_center.vocab_size, information_center.emb_size, padding_idx=0)
        if self.model_type == 'word_cust':
            # word embedding transformation parameters
            self.word_usr = nn.Embedding(self.num_usr, self.emb_size*self.emb_size)
            self.word_prd = nn.Embedding(self.num_prd, self.emb_size*self.emb_size)
            torch.nn.init.xavier_uniform_(self.word_usr.weight)
            torch.nn.init.xavier_uniform_(self.word_prd.weight)

        self.attentional_rnn = AttentionalBiLSTM(information_center)

        if self.model_type == 'linear_cust':
            self.linear_usr = nn.Embedding(self.num_usr, self.state_size*self.num_label)
            self.linear_prd = nn.Embedding(self.num_prd, self.state_size*self.num_label)
            torch.nn.init.uniform_(self.linear_usr.weight, -0.01, 0.01)
            torch.nn.init.uniform_(self.linear_prd.weight, -0.01, 0.01)
        else:
            self.W = nn.Linear(self.state_size, self.num_label, bias=False)
            torch.nn.init.xavier_uniform_(self.W.weight)

        if self.model_type == 'bias_cust':
            self.b_usr = nn.Embedding(self.num_usr, self.state_size)
            self.b_prd = nn.Embedding(self.num_prd, self.state_size)
            torch.nn.init.uniform_(self.b_usr.weight, -0.01, 0.01)
            torch.nn.init.uniform_(self.b_prd.weight, -0.01, 0.01)
            self.Y = nn.Linear(self.state_size*2, self.num_label, bias=False)
            torch.nn.init.xavier_uniform_(self.Y.weight)
        else:
            self.b = nn.Parameter(torch.zeros((1, self.num_label)))

    def forward(self, x_batch, x_lens_batch, usr_batch, prd_batch):
        batch_size = len(x_batch)
        # DATA PREPARATION : PADDING
        x_batch = self.information_center.add_pad(x_batch, x_lens_batch)

        # DATA PREPARATION : TO LIST
        x_lens_batch = [int(w) for w in x_lens_batch]

        # DATA PREPARATION : VARAIABLE WRAPPING
        x_batch = self.information_center.to_var(torch.LongTensor(x_batch))
        usr_batch = self.information_center.to_var(torch.from_numpy(usr_batch.astype(np.int64)))
        prd_batch = self.information_center.to_var(torch.from_numpy(prd_batch.astype(np.int64)))

        # 1. WORD EMBEDDING
        x_batch = self.embed(x_batch) 
        if self.model_type == 'word_cust':
            # 1-1) Word Embedding Transformation [300*300]
            word_usr = self.word_usr(usr_batch).view(batch_size, self.emb_size, self.emb_size) # batch_size, emb_size, emb_size
            word_prd = self.word_prd(prd_batch).view(batch_size, self.emb_size, self.emb_size) # batch_size, emb_size, emb_size
            ru = torch.bmm(x_batch, word_usr) # batch_size, maxlen, emb_size
            rp = torch.bmm(x_batch, word_prd) # batch_size, maxlen, emb_size
            r = ru + rp
            x_batch = x_batch + torch.tanh(r) # residual addition

        # 2. LSTM with Attention
        x_batch = self.attentional_rnn(x_batch, x_lens_batch, usr_batch, prd_batch)
        
        # 3. FC Weight Matrix
        if self.model_type == 'linear_cust':
            linear_usr = self.linear_usr(usr_batch).view(batch_size, self.state_size, self.num_label)
            linear_prd = self.linear_prd(prd_batch).view(batch_size, self.state_size, self.num_label)
            W = torch.cat([linear_usr, linear_prd], dim=1)
            x_batch = x_batch.unsqueeze(dim=1).repeat(1,1,2)
            # [batch_size, 1, state_size*4] * [batch_size, state_size*4, num_labels]
            prediction = torch.bmm(x_batch, W).squeeze(dim=1)
        else:
            prediction = self.W(x_batch)

        # 4. FC bias
        if self.model_type == 'bias_cust':
            b_usr = self.b_usr(usr_batch)
            b_prd = self.b_prd(prd_batch)
            b_latent = torch.cat([b_usr, b_prd], dim=1)
            b = self.Y(b_latent)
            prediction = prediction + b
        else:
            prediction = prediction + self.b
        
        return prediction
