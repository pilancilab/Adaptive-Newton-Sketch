import numpy as np
import scipy.sparse
from loss_funcs import safe_sigmoid

def logis_laplacian_weights(A, x):
    e = safe_sigmoid(A@x)
    return e*(1-e)

def logis_hessian_sqrt(A, x, y):
    d = (logis_laplacian_weights(A, x,) ** 0.5).reshape((-1, 1))
    if scipy.sparse.issparse(A):
        d = d.reshape([-1])
        d = scipy.sparse.diags([d],[0])
        B = d*A
    else:  
        B = A*d
    return B

def logis_sketched_hessian_sqrt(A, x, y, S):
    """Returns the square root of the sketched Hessian for the negative
    log-likelihood.
    
    The sketched Hessian takes the form
    $$
        A^T S^T W S A
    $$
    and because of its diagonal form, it may be represented as:
    $$
        B^T B
    $$
    where
    $$
        B = SAW^{\frac{1}{2}}
    $$

    Args:
        A (np.ndarray): Feature matrix
        x (np.ndarray): Coefficients
        y (np.ndarray): Targets
        S (np.ndarray): Sketch matrix
        
    Returns:
        np.ndarray: square root of Hessian, as described above as $B$.
    """
    # Reshape into column vector to broadcast across rows
    d = (logis_laplacian_weights(A, x,) ** 0.5).reshape((-1, 1))
    if scipy.sparse.issparse(A):
        d = d.reshape([-1])
        d = scipy.sparse.diags([d],[0])
        B = d*A
    else:  
        B = A*d
    return S@B


