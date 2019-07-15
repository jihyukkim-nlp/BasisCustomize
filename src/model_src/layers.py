import torch, torch.nn as nn, torch.nn.functional as F
from .SubModules.Attention import LinearAttentionWithQuery, LinearAttentionWithoutQuery
from .SubModules.TextLSTM import TextLSTM

class BasicWordEmb(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.word_em = nn.Embedding(args.vocab_size, args.word_dim, padding_idx=args._ipad)
    def forward(self, review, **kwargs):
        return self.word_em(review)
class CustWordEmb(nn.Module):
    def __init__(self, args, meta_param_manager):
        super().__init__()
        self.word_dim = args.word_dim
        self.word_em = nn.Embedding(args.vocab_size, args.word_dim, padding_idx=args._ipad)
        for name, num_meta in args.meta_units:
            setattr(self, "num_"+name, num_meta)
            # word embedding transformation parameters
            setattr(self, name, nn.Embedding(num_meta, args.word_dim*args.word_dim))
            meta_param_manager.register("CustWordEmb."+name, getattr(self, name).weight)
    def forward(self, review, **kwargs):
        x = self.word_em(review)
        r = None
        for name, idx in kwargs.items():
            v=getattr(self, name)(idx).view(x.shape[0], self.word_dim, self.word_dim)
            rv = torch.bmm(x, v)
            if (r is not None): r += rv
            else: r = rv
        x = x + torch.tanh(r) # residual addition
        return x
class BasisCustWordEmb(nn.Module):
    def __init__(self, args, meta_param_manager):
        super().__init__()
        self.word_dim = args.word_dim
        self.word_em = nn.Embedding(args.vocab_size, args.word_dim, padding_idx=args._ipad)
        for name, num_meta in args.meta_units:
            setattr(self, "num_"+name, num_meta)
            # word embedding transformation parameters
            setattr(self, name, nn.Embedding(num_meta, args.meta_dim))
            meta_param_manager.register("BasisCustWordEmb."+name, getattr(self, name).weight)
        self.P = nn.Sequential(
            nn.Linear(args.meta_dim*len(args.meta_units), args.key_query_size), # From MetaData to Query 
            nn.Tanh(),
            nn.Linear(args.key_query_size, args.num_bases, bias=False), # Calculate Weights of each Basis: Key & Query Inner-product
            nn.Softmax(dim=1),
            nn.Linear(args.num_bases, args.word_dim*args.word_dim), # Weighted Sum of Bases
            )
    def forward(self, review, **kwargs):
        x = self.word_em(review)
        query = torch.cat(
            [getattr(self, name)(idx)
            for name, idx in kwargs.items()], dim=1)
        t = self.P(query).view(x.shape[0], self.word_dim, self.word_dim)
        r = torch.bmm(x, t)
        return x+torch.tanh(r)


class BasicBiLSTM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.LSTM = TextLSTM(
            input_size=args.word_dim,
            hidden_size=args.state_size//2, # //2 for bidirectional
            bidirectional=True,
            device=args.device
            )
    def forward(self, x, length, **kwargs):
        return self.LSTM(inputs=x, length=length)[0]
class BasisCustBiLSTM(nn.Module):
    def __init__(self, args, meta_param_manager):
        super().__init__()
        self.device = args.device
        self.num_bases = args.num_bases
        self.each_state = args.state_size//2
        self.word_dim = args.word_dim
        for name, num_meta in args.meta_units:
            setattr(self, "num_"+name, num_meta)
            setattr(self, name, nn.Embedding(num_meta, args.meta_dim))
            meta_param_manager.register("BasisCustBiLSTM."+name, getattr(self, name).weight)
        self.weight_ih_l0 = nn.Parameter(torch.zeros(args.num_bases, args.state_size*2, args.word_dim))
        self.weight_hh_l0 = nn.Parameter(torch.zeros(args.num_bases, args.state_size*2, args.state_size//2))
        self.bias_l0 = nn.Parameter(torch.zeros(args.num_bases, args.state_size*2))
        self.weight_ih_l0_reverse = nn.Parameter(torch.zeros(args.num_bases, args.state_size*2, args.word_dim))
        self.weight_hh_l0_reverse = nn.Parameter(torch.zeros(args.num_bases, args.state_size*2, args.state_size//2))
        self.bias_l0_reverse = nn.Parameter(torch.zeros(args.num_bases, args.state_size*2))
        self.P = nn.Sequential(
            nn.Linear(args.meta_dim*len(args.meta_units), args.key_query_size), # From MetaData to Query 
            nn.Tanh(),
            nn.Linear(args.key_query_size, args.num_bases, bias=False), # Calculate Weights of each Basis: Key & Query Inner-product
            nn.Softmax(dim=1),
            )
    def forward(self, x, length, **kwargs):
        # low-rank factorization
        # c_batch = self.encoder_coefficient(usr_batch, prd_batch) # batch_size, num_bases
        query = torch.cat([getattr(self, name)(idx)
            for name, idx in kwargs.items()], dim=1)
        c_batch = self.P(query)
        num_bases = self.num_bases
        cell_size = self.each_state
        input_size = self.word_dim
        
        batch_size = x.size(0)
        maxlength = torch.max(length).item()
        
        # make variable for backward path
        reverse_idx = torch.arange(maxlength-1, -1, -1).to(self.device)
        # reverse_idx = torch.from_numpy(reverse_idx)
        x_reverse = x[:, reverse_idx, :]

        weight_ih_l0 = torch.mm(c_batch , self.weight_ih_l0.view(num_bases, -1)).view(batch_size, cell_size*4, input_size) # batch_size, cell_size*4, input_size
        weight_hh_l0 = torch.mm(c_batch , self.weight_hh_l0.view(num_bases, -1)).view(batch_size, cell_size*4, cell_size) # batch_size, cell_size*4, cell_size
        bias_l0 = torch.mm(c_batch, self.bias_l0) # batch_size, cell_size*4
        weight_ih_l0_reverse = torch.mm(c_batch , self.weight_ih_l0_reverse.view(num_bases, -1)).view(batch_size, cell_size*4, input_size) # batch_size, cell_size*4, input_size
        weight_hh_l0_reverse = torch.mm(c_batch , self.weight_hh_l0_reverse.view(num_bases, -1)).view(batch_size, cell_size*4, cell_size) # batch_size, cell_size*4, cell_size
        bias_l0_reverse = torch.mm(c_batch, self.bias_l0_reverse) # batch_size, cell_size*4
        
        (h0, c0) = torch.zeros((2, batch_size, cell_size, 1)).to(self.device) # only for forward path
        (h0_reverse, c0_reverse) = torch.zeros((2, batch_size, cell_size, 1)).to(self.device) # only for forward path
        hidden = (h0, c0)
        hidden_reverse = (h0_reverse, c0_reverse)
        htops = None
        htops_reverse = None
        for i in range(maxlength):
            hx, cx = hidden  # batch_size, cell_size, 1
            ix = x[:, i, :] # batch_size, input_size
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

            mask = (length-1) < i
            if mask.sum()>0:
                cy[mask] = torch.zeros(mask.sum(), cell_size).to(self.device)
                hy[mask] = torch.zeros(mask.sum(), cell_size).to(self.device)
            
            if (htops is None): htops = hy.unsqueeze(dim=1)
            else: htops = torch.cat((htops, hy.unsqueeze(dim=1)), dim=1)

            cx = cy.unsqueeze(dim=2)
            hx = hy.unsqueeze(dim=2)
            hidden = (hx, cx)

            ###############################################################################
            
            # reverse
            hx_reverse, cx_reverse = hidden_reverse  # batch_size, cell_size, 1
            ix_reverse = x_reverse[:, i, :] # batch_size, input_size
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
            mask_reverse = (maxlength-i) > length
            # mask_reverse = np.nonzero(mask_reverse)[0]
            # mask_reverse = torch.from_numpy(mask_reverse).to(self.device)
            if mask_reverse.sum() > 0:
                cy_reverse[mask_reverse] = torch.zeros(mask_reverse.sum(), cell_size).to(self.device)
                hy_reverse[mask_reverse] = torch.zeros(mask_reverse.sum(), cell_size).to(self.device)

            if (htops_reverse is None): htops_reverse = hy_reverse.unsqueeze(dim=1)
            else: htops_reverse = torch.cat((htops_reverse, hy_reverse.unsqueeze(dim=1)), dim=1)

            cx_reverse = cy_reverse.unsqueeze(dim=2)
            hx_reverse = hy_reverse.unsqueeze(dim=2)
            hidden_reverse = (hx_reverse, cx_reverse)
        
        # reverse order of backward batch
        reverse_idx = torch.arange(maxlength-1, -1, -1).to(self.device)
        # reverse_idx = torch.from_numpy(reverse_idx).to(self.device)
        htops_reverse = htops_reverse[:, reverse_idx, :]

        # concatenate forward and backward path
        hiddens = torch.cat((htops, htops_reverse), dim=2)
        return hiddens

class BasicAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.attention = LinearAttentionWithoutQuery(
            encoder_dim=args.state_size,
            device=args.device,
            )
    def forward(self, x, mask, **kwargs):
        return self.attention(x, mask=mask)[0]
class CustAttention(nn.Module):
    def __init__(self, args, meta_param_manager):
        super().__init__()
        for name, num_meta in args.meta_units:
            setattr(self, "num_"+name, num_meta)
            setattr(self, name, nn.Embedding(num_meta, args.meta_dim))
            meta_param_manager.register("CustAttention."+name, getattr(self, name).weight)
        self.attention = LinearAttentionWithQuery(encoder_dim=args.state_size, query_dim=args.meta_dim*len(args.meta_units))
    def forward(self, x, mask, **kwargs):
        return self.attention(
            x, 
            query=torch.cat([
                getattr(self, name)(idx)
                for name, idx in kwargs.items()], dim=1
                ).unsqueeze(dim=1).repeat(1, x.shape[1], 1),
            mask=mask)[0]
class BasisCustAttention(nn.Module):
    def __init__(self, args, meta_param_manager):
        super().__init__()
        for name, num_meta in args.meta_units:
            setattr(self, "num_"+name, num_meta)
            setattr(self, name, nn.Embedding(num_meta, args.meta_dim))
            meta_param_manager.register("BasisCustAttention."+name, getattr(self, name).weight)
        self.P = nn.Sequential(
            nn.Linear(args.meta_dim*len(args.meta_units), args.key_query_size), # From MetaData to Query 
            nn.Tanh(),
            nn.Linear(args.key_query_size, args.num_bases, bias=False), # Calculate Weights of each Basis: Key & Query Inner-product
            nn.Softmax(dim=1),
            nn.Linear(args.num_bases, args.state_size), # Weighted Sum of Bases
            )
        self.attention = LinearAttentionWithQuery(encoder_dim=args.state_size, query_dim=args.state_size)
    def forward(self, x, mask, **kwargs):
        return self.attention(
            x,
            query=self.P(torch.cat([
                getattr(self, name)(idx)
                for name, idx in kwargs.items()], dim=1
            ).unsqueeze(dim=1).repeat(1, x.shape[1], 1)),
            mask=mask)[0]

class BasicLinear(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.W = nn.Linear(args.state_size, args.num_label, bias=False)
    def forward(self, x, **kwargs):
        return self.W(x)
class CustLinear(nn.Module):
    def __init__(self, args, meta_param_manager):
        super().__init__()
        self.state_size = args.state_size
        self.num_label= args.num_label
        for name, num_meta in args.meta_units:
            setattr(self, "num_"+name, num_meta)
            setattr(self, name, nn.Embedding(num_meta, args.state_size*args.num_label))
            meta_param_manager.register("CustLinear."+name, getattr(self, name).weight)
    def forward(self, x, **kwargs):
        W = torch.cat([
            getattr(self, name)(idx).view(x.shape[0], self.state_size, self.num_label)
            for name, idx in kwargs.items()], dim=1)
        x = x.unsqueeze(dim=1).repeat(1,1,len(kwargs))
        return torch.bmm(x, W).squeeze(dim=1)
class BasisCustLinear(nn.Module):
    def __init__(self, args, meta_param_manager):
        super().__init__()
        self.state_size = args.state_size
        self.num_label = args.num_label
        for name, num_meta in args.meta_units:
            setattr(self, "num_"+name, num_meta)
            setattr(self, name, nn.Embedding(num_meta, args.meta_dim))
            meta_param_manager.register("BasisCustLinear."+name, getattr(self, name).weight)
        self.P = nn.Sequential(
            nn.Linear(args.meta_dim*len(args.meta_units), args.key_query_size), # From MetaData to Query 
            nn.Tanh(),
            nn.Linear(args.key_query_size, args.num_bases, bias=False), # Calculate Weights of each Basis: Key & Query Inner-product
            nn.Softmax(dim=1),
            nn.Linear(args.num_bases, args.state_size*args.num_label), # Weighted Sum of Bases
            )
    def forward(self, x, **kwargs):
        W = self.P(
            torch.cat([getattr(self, name)(idx) for name, idx in kwargs.items()], dim=1)
            ).view(x.shape[0], self.state_size, self.num_label)
        return torch.bmm(x.unsqueeze(dim=1), W).squeeze(dim=1)

class BasicBias(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.b = nn.Parameter(torch.zeros((1, args.num_label)))
    def forward(self, **kwargs):
        return self.b
class CustBias(nn.Module):
    def __init__(self, args, meta_param_manager):
        super().__init__()
        for name, num_meta in args.meta_units:
            setattr(self, "num_"+name, num_meta)
            setattr(self, name, nn.Embedding(num_meta, args.state_size))
            meta_param_manager.register("CustBias."+name, getattr(self, name).weight)
        self.Y = nn.Linear(args.state_size*len(args.meta_units), args.num_label, bias=False)
    def forward(self, **kwargs):
        return self.Y(torch.cat([
            getattr(self, name)(idx)
            for name, idx in kwargs.items()], dim=1))
class BasisCustBias(nn.Module):
    def __init__(self, args, meta_param_manager):
        super().__init__()
        for name, num_meta in args.meta_units:
            setattr(self, "num_"+name, num_meta)
            setattr(self, name, nn.Embedding(num_meta, args.meta_dim))
            meta_param_manager.register("BasisCustBias."+name, getattr(self, name).weight)
        self.P = nn.Sequential(
            nn.Linear(args.meta_dim*len(args.meta_units), args.key_query_size), # From MetaData to Query 
            nn.Tanh(),
            nn.Linear(args.key_query_size, args.num_bases, bias=False), # Calculate Weights of each Basis: Key & Query Inner-product
            nn.Softmax(dim=1),
            nn.Linear(args.num_bases, args.state_size), # Weighted Sum of Bases
            )
        self.Y = nn.Linear(args.state_size, args.num_label, bias=False)
    def forward(self, **kwargs):
        return self.Y(
            self.P(torch.cat([getattr(self, name)(idx) for name, idx in kwargs.items()], dim=1))
            )

class MetaParamManager:
    def __init__(self):
        self.meta_em = {}
    def state_dict(self):
        return self.meta_em
    def register(self, name, param):
        self.meta_em[name]=param
