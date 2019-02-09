import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class Coefficient(nn.Module):
    def __init__(self, information_center):
        super(Coefficient, self).__init__()
        self.key = nn.Linear(information_center.key_size, information_center.num_bases, bias=False)
        torch.nn.init.xavier_uniform_(self.key.weight)

        self.query_usr = nn.Embedding(information_center.num_usr, information_center.query_size)
        self.query_prd = nn.Embedding(information_center.num_prd, information_center.query_size)
        torch.nn.init.uniform_(self.query_usr.weight, -0.01, 0.01)
        torch.nn.init.uniform_(self.query_prd.weight, -0.01, 0.01)
        
    def forward(self, usr_batch, prd_batch):
        query = torch.cat([self.query_usr(usr_batch), self.query_prd(prd_batch)], dim=1)
        z = self.key(query)
        p = F.softmax(z, dim=1)
        return p

class AttentionalBiLSTM(nn.Module):
    def __init__(self, information_center):
        self.information_center = information_center
        super(AttentionalBiLSTM, self).__init__()
        self.model_type = information_center.model_type
        self.each_state = int(information_center.state_size/2)
        self.num_bases = information_center.num_bases
        self.emb_size = information_center.emb_size
        self.state_size = information_center.state_size
        
        if self.model_type=='attention_basis_cust':
            # ADDITIONAL INFORMATION
            self.att_coefficient = Coefficient(information_center)
            self.att_bases = nn.Embedding(self.num_bases, self.each_state*2)
            torch.nn.init.xavier_uniform_(self.att_bases.weight)
        else:
            self.z = nn.Linear(self.each_state*2, 1, bias=False)
            torch.nn.init.xavier_uniform_(self.z.weight)

        if self.model_type == 'encoder_basis_cust':
            self.weight_ih_l0 = nn.Parameter(torch.zeros(self.num_bases, self.each_state*4, self.emb_size))
            self.weight_hh_l0 = nn.Parameter(torch.zeros(self.num_bases, self.each_state*4, self.each_state))
            self.bias_l0 = nn.Parameter(torch.zeros(self.num_bases, self.each_state*4))
            self.weight_ih_l0_reverse = nn.Parameter(torch.zeros(self.num_bases, self.each_state*4, self.emb_size))
            self.weight_hh_l0_reverse = nn.Parameter(torch.zeros(self.num_bases, self.each_state*4, self.each_state))
            self.bias_l0_reverse = nn.Parameter(torch.zeros(self.num_bases, self.each_state*4))
            nn.init.xavier_uniform_(self.weight_ih_l0)
            nn.init.xavier_uniform_(self.weight_hh_l0)
            nn.init.constant_(self.bias_l0, 0)
            nn.init.xavier_uniform_(self.weight_ih_l0_reverse)
            nn.init.xavier_uniform_(self.weight_hh_l0_reverse)
            nn.init.constant_(self.bias_l0_reverse, 0)
            self.encoder_coefficient = Coefficient(information_center)
        else:
            self.rnn = nn.LSTM(
                        input_size=information_center.emb_size, 
                        hidden_size=self.each_state, 
                        num_layers=1, 
                        bias=True, 
                        batch_first=True,
                        bidirectional=True)
        
    def forward(self, x_batch, x_lens_batch, usr_batch, prd_batch):
        batch_size = x_batch.size(0)

        if self.model_type == 'encoder_basis_cust':
            x_lens_batch = np.array(x_lens_batch, dtype=np.int64)
            # low-rank factorization
            c_batch = self.encoder_coefficient(usr_batch, prd_batch) # batch_size, num_bases
            num_bases = self.num_bases
            cell_size = self.each_state
            input_size = self.emb_size
            
            batch_size = x_batch.size(0)
            maxlength = int(np.max(x_lens_batch))
            
            # make variable for backward path
            reverse_idx = np.arange(maxlength-1, -1, -1)
            reverse_idx = self.information_center.to_var(torch.from_numpy(reverse_idx.astype(np.int64)))
            x_batch_reverse = x_batch[:, reverse_idx, :]

            weight_ih_l0 = torch.mm(c_batch , self.weight_ih_l0.view(num_bases, -1)).view(batch_size, cell_size*4, input_size) # batch_size, cell_size*4, input_size
            weight_hh_l0 = torch.mm(c_batch , self.weight_hh_l0.view(num_bases, -1)).view(batch_size, cell_size*4, cell_size) # batch_size, cell_size*4, cell_size
            bias_l0 = torch.mm(c_batch, self.bias_l0) # batch_size, cell_size*4
            weight_ih_l0_reverse = torch.mm(c_batch , self.weight_ih_l0_reverse.view(num_bases, -1)).view(batch_size, cell_size*4, input_size) # batch_size, cell_size*4, input_size
            weight_hh_l0_reverse = torch.mm(c_batch , self.weight_hh_l0_reverse.view(num_bases, -1)).view(batch_size, cell_size*4, cell_size) # batch_size, cell_size*4, cell_size
            bias_l0_reverse = torch.mm(c_batch, self.bias_l0_reverse) # batch_size, cell_size*4
            
            (h0, c0) = self.information_center.to_var(torch.zeros((2, batch_size, cell_size, 1))) # only for forward path
            (h0_reverse, c0_reverse) = self.information_center.to_var(torch.zeros((2, batch_size, cell_size, 1))) # only for forward path
            hidden = (h0, c0)
            hidden_reverse = (h0_reverse, c0_reverse)
            htops = None
            htops_reverse = None
            for i in range(maxlength):
                hx, cx = hidden  # batch_size, cell_size, 1
                ix = x_batch[:, i, :] # batch_size, input_size
                ix = ix.unsqueeze(dim=2) # batch_size, input_size, 1

                i2h = torch.bmm(weight_ih_l0, ix)
                i2h = i2h.squeeze(dim=2) # batch_size, cell_size*4
                h2h = torch.bmm(weight_hh_l0, hx)
                h2h = h2h.squeeze(dim=2) # batch_size, cell_size*4
                
                gates = i2h + h2h + bias_l0 # batch_size, cell_size*4
                ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
                
                ingate = torch.sigmoid(ingate)
                forgetgate = torch.sigmoid(forgetgate)
                cellgate = torch.tanh(cellgate)  # o_t
                outgate = torch.sigmoid(outgate)
                
                cx = cx.squeeze(dim=2) # batch_size, cell_size
                cy = (forgetgate * cx) + (ingate * cellgate)
                hy = outgate * torch.tanh(cy) # batch_size, cell_size

                mask = (x_lens_batch-1) < i
                mask = np.nonzero(mask)[0]
                mask = self.information_center.to_var_fixed(torch.from_numpy(mask))
                if mask.size(0)>0:
                    cy[mask] = self.information_center.to_var_fixed(torch.zeros(mask.size(0), cell_size))
                    hy[mask] = self.information_center.to_var_fixed(torch.zeros(mask.size(0), cell_size))
                
                if (htops is None): htops = hy.unsqueeze(dim=1)
                else: htops = torch.cat((htops, hy.unsqueeze(dim=1)), dim=1)

                cx = cy.unsqueeze(dim=2)
                hx = hy.unsqueeze(dim=2)
                hidden = (hx, cx)

                ###############################################################################
                
                # reverse
                hx_reverse, cx_reverse = hidden_reverse  # batch_size, cell_size, 1
                ix_reverse = x_batch_reverse[:, i, :] # batch_size, input_size
                ix_reverse = ix_reverse.unsqueeze(dim=2) # batch_size, input_size, 1

                i2h = torch.bmm(weight_ih_l0_reverse, ix_reverse)
                i2h = i2h.squeeze(dim=2) # batch_size, cell_size*4
                h2h = torch.bmm(weight_hh_l0_reverse, hx_reverse)
                h2h = h2h.squeeze(dim=2) # batch_size, cell_size*4
                
                gates = i2h + h2h + bias_l0_reverse # batch_size, cell_size*4
                ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
                
                ingate = torch.sigmoid(ingate)
                forgetgate = torch.sigmoid(forgetgate)
                cellgate = torch.tanh(cellgate)  # o_t
                outgate = torch.sigmoid(outgate)
                
                cx_reverse = cx_reverse.squeeze(dim=2) # batch_size, cell_size
                cy_reverse = (forgetgate * cx_reverse) + (ingate * cellgate)
                hy_reverse = outgate * torch.tanh(cy_reverse) # batch_size, cell_size
                
                # mask
                mask_reverse = (maxlength-i) > x_lens_batch
                mask_reverse = np.nonzero(mask_reverse)[0]
                mask_reverse = self.information_center.to_var_fixed(torch.from_numpy(mask_reverse))
                if mask_reverse.size(0) > 0:
                    cy_reverse[mask_reverse] = self.information_center.to_var_fixed(torch.zeros(mask_reverse.size(0), cell_size))
                    hy_reverse[mask_reverse] = self.information_center.to_var_fixed(torch.zeros(mask_reverse.size(0), cell_size))

                if (htops_reverse is None): htops_reverse = hy_reverse.unsqueeze(dim=1)
                else: htops_reverse = torch.cat((htops_reverse, hy_reverse.unsqueeze(dim=1)), dim=1)

                cx_reverse = cy_reverse.unsqueeze(dim=2)
                hx_reverse = hy_reverse.unsqueeze(dim=2)
                hidden_reverse = (hx_reverse, cx_reverse)
            
            # reverse order of backward batch
            reverse_idx = np.arange(maxlength-1, -1, -1)
            reverse_idx = self.information_center.to_var(torch.from_numpy(reverse_idx.astype(np.int64)))
            htops_reverse = htops_reverse[:, reverse_idx, :]

            # concatenate forward and backward path
            hiddens = torch.cat((htops, htops_reverse), dim=2)
        else:
            x_pack = nn.utils.rnn.pack_padded_sequence(x_batch, x_lens_batch, batch_first=True)
            (h0, c0) = self.information_center.to_var(torch.zeros((2, 2, batch_size, self.each_state)))
            hiddens, _ = self.rnn(x_pack, (h0, c0))
            hiddens, _ = nn.utils.rnn.pad_packed_sequence(hiddens, batch_first=True)

        if self.information_center.model_type == 'attention_basis_cust':
            # LOW RANK FACTORIZATION: ATTENTION
            c = self.att_coefficient(usr_batch, prd_batch)
            b = self.att_bases.weight
            meta_batch = torch.mm(c, b)
        else:
            meta_batch = None

        hidden_reduced = self.reduce_attention(hiddens, x_lens_batch, meta_batch)
        return hidden_reduced

    def reduce_attention(self, hiddens_batch, lens_batch, meta_batch=None):
        batch_size = hiddens_batch.size(0)
        x = None
        for i in range(batch_size):
            lens = lens_batch[i]
            hidden = hiddens_batch[i][:lens]
            if self.information_center.model_type == 'attention_basis_cust':
                energy = torch.mm(meta_batch[i].unsqueeze(dim=0), hidden.t())
            else:
                energy = self.z(hidden).t()
            attention = F.softmax(energy, dim=1)
            hidden_reduced = torch.mm(attention, hidden)
            x = hidden_reduced if (x is None) else torch.cat([x, hidden_reduced], dim=0)
        return x

class basis_cust(nn.Module):

    def __init__(self, information_center):
        self.information_center = information_center
        self.model_type = self.information_center.model_type
        self.num_bases = self.information_center.num_bases
        self.num_usr = self.information_center.num_usr
        self.num_prd = self.information_center.num_prd
        self.num_label = self.information_center.num_label
        self.state_size = self.information_center.state_size
        self.emb_size = self.information_center.emb_size
        super(basis_cust, self).__init__()

        self.embed = nn.Embedding(information_center.vocab_size, information_center.emb_size, padding_idx=0)
        if self.model_type=='word_basis_cust':
            self.word_coefficient = Coefficient(information_center)
            self.word_bases = nn.Parameter(torch.zeros(self.num_bases, self.emb_size*self.emb_size))
            torch.nn.init.xavier_uniform_(self.word_bases)    

        self.attentional_rnn = AttentionalBiLSTM(information_center)
        
        if self.model_type=='linear_basis_cust':
            # ADDITIONAL INFORMATION : FC Weight Matrix
            self.W_coefficient = Coefficient(information_center)
            self.W_bases = nn.Parameter(torch.zeros((self.num_bases, self.state_size*self.num_label)))
            torch.nn.init.xavier_uniform_(self.W_bases)
        else:
            self.W = nn.Linear(self.state_size, self.num_label, bias=False)
            torch.nn.init.xavier_uniform_(self.W.weight)

        if self.model_type=='bias_basis_cust':
            # ADDITIONAL INFORMATION : FC bias
            self.b_coefficient = Coefficient(information_center)
            self.b_bases = nn.Embedding(self.num_bases, self.state_size)
            self.Y = nn.Linear(self.state_size, self.num_label, bias=False)
            torch.nn.init.xavier_uniform_(self.b_bases.weight)
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
        if self.model_type == 'word_basis_cust':
            # 1-1) Word Embedding Transformation [300*300]
            c = self.word_coefficient(usr_batch, prd_batch)
            b = self.word_bases
            W = torch.mm(c, b).view(batch_size, self.emb_size, self.emb_size) # batch_size, emb_size, emb_size
            r = torch.bmm(x_batch, W) # batch_size, maxlen, emb_size
            x_batch = x_batch + torch.tanh(r) # residual addition
        
        # 2. LSTM with Attention
        x_batch = self.attentional_rnn(x_batch, x_lens_batch, usr_batch, prd_batch)

        # 3. FC Weight Matrix
        if self.model_type == 'linear_basis_cust':
            C = self.W_coefficient(usr_batch, prd_batch)
            W = torch.mm(C, self.W_bases).view(batch_size, self.state_size, self.num_label)
            prediction = torch.bmm(x_batch.unsqueeze(dim=1), W).squeeze(dim=1)
        else:
            prediction = self.W(x_batch)
        
        # 4. FC bias
        if self.model_type == 'bias_basis_cust':
            C = self.b_coefficient(usr_batch, prd_batch)
            b_latent = torch.mm(C, self.b_bases.weight)
            b = self.Y(b_latent)
            prediction = prediction + b # + bias
        else:
            prediction = prediction + self.b

        return prediction