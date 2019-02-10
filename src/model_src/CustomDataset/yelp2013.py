import torch
from torch.utils.data import Dataset, DataLoader
import pickle, numpy as np, os
from model_src.SubModules.TextUtils import text_padding

class CustomDataset(Dataset):
	def __init__(self, data):
		self.data = data
	def __getitem__(self, index): return self.data[index]
	def __len__(self): return len(self.data)
class yelp2013():
	def __init__(self, args):
		self.device = args.device

		# load vocabulary
		with open(args.vocab_dir, "r") as f:
			vocab = f.read().split('\n')
		self.word2idx = {w:i for i, w in enumerate(vocab)}
		self.idx2word = {i:w for i, w in enumerate(vocab)}
		self._ipad = self.word2idx['<PAD>']
		args._ipad = self._ipad
		args.vocab_size = len(self.word2idx)

		self.train_dataloader = DataLoader(
			dataset=CustomDataset(
				data=self.cust_read_data(args.train_datadir) if 'cust' in args.model_type \
				else self.basic_read_data(args.train_datadir)),
			batch_size=args.batch_size,
			collate_fn=self.cust_collate_fn if 'cust' in args.model_type else self.basic_collate_fn,
			shuffle=True,
			)
		self.dev_dataloader = DataLoader(
			dataset=CustomDataset(
				data=self.cust_read_data(args.dev_datadir) if 'cust' in args.model_type \
				else self.basic_read_data(args.train_datadir)),
			batch_size=args.batch_size,
			collate_fn=self.cust_collate_fn if 'cust' in args.model_type else self.basic_collate_fn,
			shuffle=False,
			)
		self.test_dataloader = DataLoader(
			dataset=CustomDataset(
				data=self.cust_read_data(args.test_datadir) if 'cust' in args.model_type \
				else self.basic_read_data(args.train_datadir)),
			batch_size=args.batch_size,
			collate_fn=self.cust_collate_fn if 'cust' in args.model_type else self.basic_collate_fn,
			shuffle=False,
			)

		args.num_label = 5 # rating from 0 to 4
		# (name of meta unit, number of meta unit)
		args.meta_units = [('user', 1631), ('product', 1633)] 
	def cust_collate_fn(self, sample_batch):
		user, product, rating, length, review = list(zip(*sample_batch))
		review = text_padding(text=review, length=length, padding_idx=self._ipad)
		mask = [[1]*l + [0]*(review.shape[1]-l) for l in length]
		user = torch.tensor(user).to(self.device)
		product = torch.tensor(product).to(self.device)
		rating = torch.tensor(rating).to(self.device)
		length = torch.tensor(length).to(self.device)
		review = torch.tensor(review).to(self.device)
		mask = torch.tensor(mask).to(self.device)
		return rating, {"review":review, "mask":mask, "length":length, "user":user, "product":product}
	def basic_collate_fn(self, sample_batch):
		rating, length, review = list(zip(*sample_batch))
		review = text_padding(text=review, length=length, padding_idx=self._ipad)
		mask = [[1]*l + [0]*(review.shape[1]-l) for l in length]
		rating = torch.tensor(rating).to(self.device)
		length = torch.tensor(length).to(self.device)
		review = torch.tensor(review).to(self.device)
		mask = torch.tensor(mask).to(self.device)
		return rating, {"review":review, "mask":mask, "length":length}

	def cust_read_data(self, path):
		with open(path, 'r') as f:
			data = f.read().split("\n")
		user, product, rating, length, review = list(zip(*[x.split(",") for x in data]))
		user = [int(x) for x in user]
		product = [int(x) for x in product]
		rating = [int(x) for x in rating]
		length = [int(x) for x in length]
		review = [[int(x) for x in xs.split("_")] for xs in review]
		return list(zip(user, product, rating, length, review))
	def basic_read_data(self, path):
		with open(path, 'r') as f:
			data = f.read().split("\n")
		_, _, rating, length, review = list(zip(*[x.split(",") for x in data]))
		rating = [int(x) for x in rating]
		length = [int(x) for x in length]
		review = [[int(x) for x in xs.split("_")] for xs in review]
		return list(zip(rating, length, review))