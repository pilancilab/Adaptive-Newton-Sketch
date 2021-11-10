import numpy as np
import scipy.sparse

def gen_sketch_mat(m, n, method, sparsity=1):
    """Generate a sketch matrix.

    A sketch matrix $S\in\mathbb{R}^{m\times n}$ has the property that
    $\mathbb{E} S^T S = \mathbb{I}$.  

    Args:
        m (int): number of rows of the sketch matrix (desired rank)
        n (int): number of columns of the sketch matrix (size of matrix
            to be sketched)
        method (str): method for generating the sketch matrix.  Currently,
            only random normal sketch matrices are supported.
    Returns:
        np.ndarray: a sketch matrix
    """
    if method == 'Gaussian':
        S = np.random.randn(m, n) / np.sqrt(m)
    elif method == 'Rademacher':
        # Produces r.v. in {0, 1} with equal probability
        S = ((np.random.randn(m, n) > 0) * 2 - 1) /  np.sqrt(m)
    elif method == 'SJLT':
        S = get_SJLT_matrix(m,n,sparsity)/np.sqrt(sparsity)
    elif method == 'RRS':
        S = get_RRS_matrix(m,n)*np.sqrt(n/m)
    else:
        raise ValueError('Unrecognized sketch type: ' + method)
    return S

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

def get_RRS_matrix(m,n): 
    # This function returns random row sampling sketching matrix in the form of a sparse matrix.
    # n: number of data samples m: sketch size
    
    nonzeros = np.ones([m])
    
    all_row_indices = np.linspace(0, n-1, n, dtype=np.int32)
    
    shuffled_row_indices = np.random.permutation(all_row_indices)  
    
    I = np.linspace(0, m-1, m, dtype=np.int32)
    J = shuffled_row_indices[0:m]
    V = nonzeros

    S = scipy.sparse.coo_matrix((V,(I,J)), shape=(m,n), dtype=np.int8)
    S = S.tocsr()
    
    return S