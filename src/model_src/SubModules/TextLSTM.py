import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import itertools # for MultiDocumentTextLSTM
class TextLSTM(nn.Module):
    def __init__(self, 
        input_size, hidden_size, 
        num_layers=1, bidirectional=False, dropout=0, bias=True, 
        batch_first=True,
        device='cpu'):
        super(TextLSTM, self).__init__()
        self.batch_first=batch_first
        self.bidirectional=bidirectional
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device
        if self.num_layers==1: dropout=0
        self.rnn = nn.LSTM(
                        input_size=input_size, 
                        hidden_size=hidden_size, 
                        num_layers=num_layers, 
                        bias=bias, 
                        batch_first=batch_first,
                        bidirectional=bidirectional,
                        dropout=dropout)
    def zero_init(self, batch_size):
        nd = 1 if not self.bidirectional else 2
        h0 = Variable(torch.zeros((self.num_layers*nd, batch_size, self.hidden_size))).to(self.device)
        c0 = Variable(torch.zeros((self.num_layers*nd, batch_size, self.hidden_size))).to(self.device)
        return (h0, c0)
    def forward(self, inputs, length, rnn_init=None, is_sorted=False):
        if rnn_init is None:
            rnn_init = self.zero_init(inputs.size(0))
        if not is_sorted:
            sort_idx = torch.sort(-length)[1]
            inputs = inputs[sort_idx]
            length = length[sort_idx]
            # h0: size=(num_layers*bidriectional, batch_size, hidden_dim)
            # c0: size=(num_layers*bidriectional, batch_size, hidden_dim)
            h0, c0 = rnn_init
            rnn_init = (h0[:, sort_idx, :], c0[:, sort_idx, :])
            unsort_idx = torch.sort(sort_idx)[1]
        x_pack = nn.utils.rnn.pack_padded_sequence(inputs, length, batch_first=self.batch_first)
        output, (hn, cn) = self.rnn(x_pack, rnn_init)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=self.batch_first)
        if not is_sorted:
            output = output[unsort_idx]
            hn = hn[:, unsort_idx, :]
            cn = cn[:, unsort_idx, :]
        # batch_size, length, hidden_size
        # batch_size, num_layers*bidirectional, hidden_size
        return output, (hn,cn)

class MultiDocumentTextLSTM(TextLSTM):
    def __init__(self, *args, **kwargs):
        super(MultiDocumentTextLSTM, self).__init__(*args, **kwargs)
    def txt_pad(self, text, length, padidx=0):
        """ 
        text: list of token indices
        length: list of length of tokens (same size with text)
        return: padded text tokens, ndarray, np.int64
        """
        maxlen = max(length)
        padded_sentences = []
        for l, x in zip(length, text):
            if l<maxlen: padded_sentences.append(x+[padidx]*(maxlen-l))
            else: padded_sentences.append(x)
        return np.array(padded_sentences, dtype=np.int64)

    def forward(self, inputs, num_docs, length, rnn_init=None, is_sorted=True):
        """
        inputs: 4d tensor (already padded)
        num_docs: number of documents for each instance in batch
        length: 2d list (length for each document) 
        description)
            > pack documents from (N, D, L, word_dim) to (N*D, L, word_dim)
            > pass LSTM 
            > unpack documents from (N*D, L, hidden_dim) to (N,D,L,hidden_dim)
        """
        batch_size = inputs.size(0)
        max_num_docs = inputs.size(1)
        max_length = inputs.size(2)
        word_dim = inputs.size(3)
        # 1. Get batch index for unpacking
        # batch_index = itertools.chain(*[[i]*d for i, d in enumerate(num_docs)])

        # 2. Pack documents
        length_padded = self.txt_pad(length, num_docs)
        length_merged = itertools.chain(*length_padded) # N*D
        inputs_merged = itertools.chain(*inputs) # N*D, L, word_dim

        # 3. Pass LSTM
        # output: N*D, L, hidden_dim
        output, (hn,cn) = super(MultiDocumentTextLSTM, self)(self, inputs_merged, length_merged, rnn_init, is_sorted)

        # 4. Unpack documents
        output = output.view(batch_size, max_num_docs, max_length, -1)
        hn = hn.view(hn.size(0), batch_size, max_num_docs, hn.size(2), -1)
        cn = cn.view(cn.size(0), batch_size, max_num_docs, cn.size(2), -1)
        return output, (hn,cn)