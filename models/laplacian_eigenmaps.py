import torch
import numpy as np

def laplacian(A, verbose=0):
    W = A
    node_degrees = [int(x.item()) for x in A.sum(axis=1)]

    # Create diagonal degree matrix
    D = torch.diag(torch.tensor(node_degrees))

    if verbose:
        print('Adjacency matrix: \n', A)
        print('Degree matrix: \n', D)
        print('Laplacian matrix: \n', L)

    L = D - W

    return L

def laplacian_eigemaps(A, d, verbose=0):
    L = laplacian(A, verbose=verbose)
    eigenvalues, eigenvectors = torch.linalg.eigh(L)
    
    i = 0
    while np.abs(eigenvalues[i]) < 1e-9:
        i+=1

    return eigenvectors[:, i:d+i]
