import os, pickle, numpy as np, time
from tqdm import tqdm

if not os.path.exists("./processed_data"):
	os.mkdir("./processed_data")
with open('../../predefined_vocab/yelp2013/42939.vocab', "r") as f:
	vocab = f.read().split("\n")
word2idx = {w:i for i,w in enumerate(vocab)}

with open('./yelp-2013-seg-20-20.train.ss') as f:
	data = f.read().split("\n")[:-1]
user, product, rating, review = list(zip(*[x.split("\t\t") for x in data]))

users = set(user); user2idx = {u:i for i,u in enumerate(users)}
products = set(product); product2idx = {p:i for i,p in enumerate(products)}

userid = [user2idx[x] for x in user] # convert str into integer index
productid = [product2idx[x] for x in product] # convert str into integer index
rating = [int(x)-1 for x in rating] # make rating start from 0 to 4
i_unk = word2idx['<UNK>']
review = [[word2idx.get(x, i_unk)for x in xs.split()]for xs in review]
length = [len(x) for x in review]
review = ["_".join([str(x) for x in xs]) for xs in review]

with open('./processed_data/train.txt', 'w') as f:
	f.write("\n".join(["{},{},{},{},{}".format(u,p,r,l,x) for u,p,r,l,x in zip(userid, productid, rating, length, review)]))

def process_data(in_path, out_path):
	with open(in_path, 'r') as f:
		data = f.read().split("\n")[:-1]
	user, product, rating, review = list(zip(*[x.split("\t\t") for x in data]))
	userid = [user2idx[x] for x in user]
	productid = [product2idx[x] for x in product]
	rating = [int(x)-1 for x in rating]
	i_unk = word2idx['<UNK>']
	review = [[word2idx.get(x, i_unk)for x in xs.split()]for xs in review]
	length = [len(x) for x in review]
	review = ["_".join([str(x) for x in xs]) for xs in review]
	with open(out_path, 'w') as f:
		f.write("\n".join(["{},{},{},{},{}".format(u,p,r,l,x) for u,p,r,l,x in zip(userid, productid, rating, length, review)]))

process_data(in_path='./yelp-2013-seg-20-20.dev.ss', out_path="./processed_data/dev.txt")
process_data(in_path='./yelp-2013-seg-20-20.test.ss', out_path="./processed_data/test.txt")
