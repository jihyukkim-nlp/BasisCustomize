import torch, torch.nn as nn, torch.nn.functional as F
from torch.autograd import Variable
def masked_softmax(logits, mask, dim=1, epsilon=1e-5):
	""" logits, mask has same size """
	masked_logits = logits.masked_fill(mask == 0, -1e9)
	max_logits = torch.max(masked_logits, dim=dim, keepdim=True)[0]
	exps = torch.exp(masked_logits-max_logits)
	masked_exps = exps * mask.float()
	masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
	return masked_exps/masked_sums

class AttentionWithoutQuery(nn.Module):
	def __init__(self, encoder_dim, device=torch.device('cpu')):
		super(AttentionWithoutQuery, self).__init__()
		self.encoder_dim=encoder_dim
		self.device=device
	def forward(self, encoder_dim, length=None):
		"""
		encoded_vecs: batch_size, max_length, encoder_hidden_dim
		(optional) length: list of lengths of encoded_vecs
			> if length is given then perform masked_softmax
			> None indicate fixed number of length (all same length in batch)
		"""
		pass
class LinearAttentionWithoutQuery(AttentionWithoutQuery):
	def __init__(self, encoder_dim, device=torch.device('cpu')):
		super().__init__(encoder_dim, device)
		self.z = nn.Linear(self.encoder_dim, 1, bias=False)
	def forward(self, encoded_vecs, mask=None):
		logits = self.z(encoded_vecs).squeeze(dim=2)
		if (mask is not None):
			# batch_size, max_length
			attention = masked_softmax(logits=logits, mask=mask, dim=1)
		else:
			# batch_size, max_length
			attention = F.softmax(logits, dim=1)
		return (
			torch.bmm(attention.unsqueeze(dim=1), encoded_vecs).squeeze(dim=1),
			attention
			)
class MLPAttentionWithoutQuery(AttentionWithoutQuery):
	def __init__(self, encoder_dim, device=torch.device('cpu')):
		"""
		ev_t: encoded_vecs
		u_t = tanh(W*(ev_t)+b)
		a_t = softmax(v^T u_t)
		"""
		super(MLPAttentionWithoutQuery, self).__init__(encoder_dim, device)
		self.W = nn.Sequential(
			nn.Linear(
				self.encoder_dim,
				self.encoder_dim
				),
			nn.Tanh(),
			nn.Linear(self.encoder_dim, 1, bias=False)
			)
	def forward(self, encoded_vecs, length=None):
		
		# batch_size, max_length
		logits = self.W(encoded_vecs).squeeze(dim=2)
		if (length is not None):
			N, L = logits.size()
			mask = [[1]*l + [0]*(L-l) for l in length]
			mask = torch.LongTensor(mask).to(self.device)
			# batch_size, max_length
			attention = masked_softmax(logits=logits, mask=mask, dim=1)
		else:
			# batch_size, max_length
			attention = F.softmax(logits, dim=1)

		# batch_size, encoder_dim
		return (
			torch.bmm(attention.unsqueeze(dim=1), encoded_vecs).squeeze(dim=1),
			attention
			)

class AttentionWithQuery(nn.Module):
	""" AttentionWithQuery
	e.g. Language Translation, SA with meta information
	"""
	def __init__(self, encoder_dim, query_dim, device=torch.device('cpu')):
		super(AttentionWithQuery, self).__init__()
		self.encoder_dim=encoder_dim
		self.query_dim=query_dim
		self.device=device
	def forward(self, encoded_sequence, length=None): pass
class LinearAttentionWithQuery(AttentionWithQuery):
	def __init__(self, encoder_dim, query_dim, device=torch.device('cpu')):
		super().__init__(encoder_dim, query_dim, device)
	def forward(self, encoded_vecs, query, mask=None):
		logits = (encoded_vecs*query).sum(dim=2)
		if (mask is not None):
			# batch_size, max_length
			attention = masked_softmax(logits=logits, mask=mask, dim=1)
		else:
			# batch_size, max_length
			attention = F.softmax(logits, dim=1)
		return (
			torch.bmm(attention.unsqueeze(dim=1), encoded_vecs).squeeze(dim=1),
			attention
			)
class MLPAttentionWithQuery(AttentionWithQuery):
	def __init__(self, encoder_dim, query_dim, device=torch.device('cpu')):
		""" ev_t: encoded_vecs, q_t: query
		u_t = tanh(W*(ev_t, q_t)+b)
		a_t = softmax(v^T u_t)
		"""
		super(MLPAttentionWithQuery, self).__init__(encoder_dim, query_dim, device)
		self.W = nn.Sequential(
			nn.Linear(
				self.encoder_dim+self.query_dim,
				self.encoder_dim
				),
			nn.Tanh(),
			nn.Linear(self.encoder_dim, 1, bias=False)
			)
		for p in self.parameters():
			if p.dim()>1:
				nn.init.xavier_normal_(p)
	def forward(self, encoded_vecs, query, length=None):
		"""
		encoded_vecs: batch_size, max_length, encoder_hidden_dim
		query: batch_size, max_length, query_dim
		(optional) length: list of lengths of encoded_vecs
			> if length is given then perform masked_softmax
			> None indicate fixed number of length (all same length in batch)
		"""
		# in case, query.size()=(batch_size, query_dim)
		# indicates query is length independent, e.g. attention for classification
		if query.dim()==2: 
			query = query.unsqueeze(dim=1).repeat(1, encoded_vecs.size(1), 1)
		# batch_size, max_length
		logits = self.W(torch.cat([encoded_vecs, query], dim=2)).squeeze(dim=2)
		if (length is None):
			# batch_size, max_length
			attention = F.softmax(logits, dim=1)
		else:
			N, L = logits.size()
			mask = [[1]*l + [0]*(L-l) for l in length]
			mask = Variable(torch.LongTensor(mask)).to(self.device)
			# batch_size, max_length
			attention = masked_softmax(logits=logits, mask=mask, dim=1)

		# batch_size, encoder_dim
		return (
			torch.bmm(attention.unsqueeze(dim=1), encoded_vecs).squeeze(dim=1),
			attention
			)

