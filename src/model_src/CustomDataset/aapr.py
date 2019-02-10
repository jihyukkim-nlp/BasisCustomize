from torch.utils.data import Dataset, DataLoader

class aapr(Dataset):
	def __init__(self, path):
		self.data = self.read_data(path)
	def __getitem__(self, index): return self.data[index]
	def __len__(self): return len(self.data)
	def custom_collate_fn(self, sample_batch):
		pass
	def read_data(self, path):
		pass