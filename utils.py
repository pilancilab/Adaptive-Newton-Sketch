import numpy as np
import scipy.sparse

def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

def get_SJLT_matrix(m, n, s): 
    # This function returns SJLT sketching matrix in the form of a sparse matrix.
    # m: sketch size, n: number of data samples, s: sparsity
    nonzeros = 2*np.random.binomial(1, 0.5, size=(s*n)) - 1 # Rademacher random variables

    K = int(np.ceil(s*n / m)) # number of repetitions
    shuffled_row_indices = np.zeros((K*m), dtype=np.int32)
    all_row_indices = np.linspace(0, m-1, m, dtype=np.int32)
    
    for k in range(K):
        shuffled_row_indices[k*m:(k+1)*m] = np.random.permutation(all_row_indices)  

    I = shuffled_row_indices[0:s*n]
    J = np.repeat(np.linspace(0, n-1, n, dtype=np.int32), s)
    V = nonzeros

    S = scipy.sparse.coo_matrix((V,(I,J)), shape=(m,n), dtype=np.int8)
    S = S.tocsr()
    
    return S