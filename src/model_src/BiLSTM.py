import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class AttentionalBiLSTM(nn.Module):
    def __init__(self, information_center):
        self.information_center = information_center
        super(AttentionalBiLSTM, self).__init__()
        self.rnn = nn.LSTM(
                        input_size=self.information_center.emb_size, 
                        hidden_size=int(self.information_center.state_size/2), # bidirectional 
                        num_layers=1, 
                        bias=True, 
                        batch_first=True,
                        bidirectional=True)
        self.each_state = int(self.information_center.state_size/2)
        self.z = nn.Linear(self.each_state*2, 1, bias=False)
        torch.nn.init.xavier_uniform_(self.z.weight)

    def forward(self, x_batch, x_lens_batch):
        batch_size = x_batch.size(0)
        x_pack = nn.utils.rnn.pack_padded_sequence(x_batch, x_lens_batch, batch_first=True)
        (h0, c0) = self.information_center.to_var(torch.zeros((2, 2, batch_size, self.each_state)))
        x_batch, _ = self.rnn(x_pack, (h0, c0))
        x_batch, _ = nn.utils.rnn.pad_packed_sequence(x_batch, batch_first=True)
        x_batch = self.reduce_attention(x_batch, x_lens_batch)
        return x_batch

    def reduce_attention(self, hiddens_batch, lens_batch):
        batch_size = hiddens_batch.size(0)
        x = None
        for i in range(batch_size):
            lens = lens_batch[i]
            hidden = hiddens_batch[i][:lens] # length, state_size
            energy = self.z(hidden).t()
            attention = F.softmax(energy, dim=1)
            hidden_reduced = torch.mm(attention, hidden)
            x = hidden_reduced if (x is None) else torch.cat([x, hidden_reduced], dim=0)
        return x

class  BiLSTM(nn.Module):

    def __init__(self, information_center):
        self.information_center = information_center
        super( BiLSTM, self).__init__()

        self.embed = nn.Embedding(information_center.vocab_size, information_center.emb_size, padding_idx=0)
        self.attentional_rnn = AttentionalBiLSTM(information_center)

        self.FC = nn.Linear(information_center.state_size, information_center.num_label)
        torch.nn.init.xavier_uniform_(self.FC.weight)
        torch.nn.init.constant_(self.FC.bias, 0)

    def forward(self, x_batch, x_lens_batch):
        # DATA PREPARATION : PADDING
        x_batch = self.information_center.add_pad(x_batch, x_lens_batch)
        
        # DATA PREPARATION : TO LIST
        x_lens_batch = [int(w) for w in x_lens_batch]

        # DATA PREPARATION : VARAIABLE WRAPPING
        x_batch = self.information_center.to_var(torch.LongTensor(x_batch))
        
        # FORWARD PROPAGATION
        x_batch = self.embed(x_batch) # WORD EMBEDDING
        x_batch = self.attentional_rnn(x_batch, x_lens_batch) # LSTM - Attention

        # FC
        x_batch = self.FC(x_batch)
        return x_batch
