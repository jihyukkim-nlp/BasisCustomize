import torch, torch.nn as nn, torch.nn.functional as F
from .layers import *

class Classifier(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.meta_param_manager = MetaParamManager()
        if args.model_type == 'word_cust':
            self.word_em = CustWordEmb(args, self.meta_param_manager)
        elif args.model_type == 'word_basis_cust':
            self.word_em = BasisCustWordEmb(args, self.meta_param_manager)
        else:
            self.word_em = BasicWordEmb(args)
        
        if args.model_type == 'encoder_cust':
            raise Exception("Out-Of-Memory occurs ... ")
        elif args.model_type == 'encoder_basis_cust':
            self.encoder = BasisCustBiLSTM(args, self.meta_param_manager)
        else:
            self.encoder = BasicBiLSTM(args)

        if args.model_type == 'attention_cust':
            self.attention = CustAttention(args, self.meta_param_manager)
        elif args.model_type == 'attention_basis_cust':
            self.attention = BasisCustAttention(args, self.meta_param_manager)
        else:
            self.attention = BasicAttention(args)

        if args.model_type == 'linear_cust':
            self.W = CustLinear(args, self.meta_param_manager)
        elif args.model_type == 'linear_basis_cust':
            self.W = BasisCustLinear(args, self.meta_param_manager)
        else:
            self.W = BasicLinear(args)

        if args.model_type == 'bias_cust':
            self.b = CustBias(args, self.meta_param_manager)
        elif args.model_type == 'bias_basis_cust':
            self.b = BasisCustBias(args, self.meta_param_manager)
        else:
            self.b = BasicBias(args)
        self.word_em_weight = self.word_em.word_em.weight # for pretrained word em vector loading
    def forward(self, review, length, mask, **kwargs):
        x = self.word_em(review, **kwargs)
        # 2. BiLSTM 
        x = self.encoder(x, length, **kwargs)
        # 3. Attention
        x = self.attention(x, mask, **kwargs)
        # 4. FC Weight Matrix
        x = self.W(x, **kwargs)
        # 5. FC bias
        x += self.b(**kwargs)
        return x