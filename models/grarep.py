import torch

def grarep_train(A, K:int=4, emb_size_per_K:int=32, l:int=1, verbose:int=1):

	print('Starting training...')

	scaled_A = A / A.sum(axis=1)
	A_k = torch.eye(n=scaled_A.shape[0])

	l = torch.tensor(l)
	W = torch.zeros((scaled_A.shape[0], emb_size_per_K*K))
	
	for i in range(K):
		if verbose: print('Training step %d' % (i+1))
		
		A_k = torch.mm(A_k, scaled_A)
		prob_trans = torch.log(A_k / torch.sum(A_k, dim=0).repeat(scaled_A.shape[0], 1)) - torch.log(l / scaled_A.shape[0])
		prob_trans[prob_trans < 0] = 0
		prob_trans[prob_trans == torch.nan] = 0

		U, S, VT = torch.svd(prob_trans)

		j = 0
		while torch.abs(S[j]) < 1e-9:
			j+=1

		U_d = U[:, j:emb_size_per_K+j]
		S_d = S[j:emb_size_per_K+j]

		W_d = U_d * torch.pow(input=S_d, exponent=(1/2))

		W[:, emb_size_per_K * i: emb_size_per_K * (i + 1)] = W_d[:, :]
		
	print('Training done')
	return W