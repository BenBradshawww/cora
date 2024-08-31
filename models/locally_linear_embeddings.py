import numpy as np
import networkx as nx
import heapq
import torch

def k_nearest_neighbours(G, node, k):
	distances = []

	for other_node in G.nodes():
		if node != other_node:
			try:
				dist = nx.shortest_path_length(G, node, other_node)
				distances.append((dist, other_node))
			except:
				continue

	k_nearest = heapq.nsmallest(k, distances)

	return [node for dist, node in k_nearest]


def locally_linear_embedding(A, n_neighbours, d):
	n_samples = A.shape[0]

	G = nx.from_numpy_array(A.detach().numpy())
	checkpoints = set([(i*n_samples)//10 for i in range(1, 11)])

	W = np.zeros((n_samples, n_samples))
	for i in range(n_samples):
		neighbours = k_nearest_neighbours(G, i, n_neighbours)
		found_n_neighbours = len(neighbours)
		Z = A[neighbours] - A[i] 
		C = np.dot(Z, Z.T)  
		C = C + np.identity(found_n_neighbours) * 1e-3  
		w = np.linalg.solve(C, np.ones(found_n_neighbours))  
		W[i, neighbours] = w / np.sum(w) 
		if i+1 in checkpoints:
			print(f'Found {i+1}/{n_samples} Neighbours')

	M = np.eye(n_samples) - W
	M = np.dot(M.T, M)

	eigenvalues, eigenvectors = np.linalg.eigh(M)

	i = 0
	while np.abs(eigenvalues[i]) < 1e-9:
		i+=1

	Y = eigenvectors[:, i:d+i] 
	
	return torch.FloatTensor(Y)