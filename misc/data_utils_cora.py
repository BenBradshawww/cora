import torch
import numpy as np
from collections import defaultdict

def get_y_splits(y:torch.FloatTensor, idx_train, idx_val, idx_test):

	y_train = torch.zeros(y.shape)
	y_val = torch.zeros(y.shape)
	y_test = torch.zeros(y.shape)

	y_train[idx_train] = y[idx_train]
	y_val[idx_val] = y[idx_val]
	y_test[idx_test] = y[idx_test]

	return y_train, y_val, y_test

def get_splits(y:torch.FloatTensor, pct_train:int=10):

	index_storage = defaultdict(list)
	for index, y_val in enumerate(y.numpy()):
		y_index = np.where(y_val == 1)[0][0]
		index_storage[y_index].append(index)
	
	n_classes = y.shape[1]
	n_samples = y.shape[0]
	pct_train = 100//pct_train
	samples_per_class = n_samples//(pct_train*n_classes)

	idx_train = []
	for i in range(n_classes):
		idx_train.extend(index_storage[i][:samples_per_class])
		
	
	idx_train_set = set(idx_train)
	train_size = len(idx_train)
	
	all_idx = set(range(n_samples))
	remaining_idx = all_idx - idx_train_set
	remaining_idx = list(remaining_idx)
	idx_val = remaining_idx[train_size:train_size*2]
	idx_test = remaining_idx[train_size*2:]

	return idx_train, idx_val, idx_test
