import numpy as np
import scipy.sparse

def safe_sigmoid(x):
    return np.exp(np.fmin(x, 0)) / (1 + np.exp(-np.abs(x)))

def logis_loss(A, x, y):
    """Return logistic loss (negative log-likelihood)

    Args:
        A (np.ndarray): Feature matrix
        x (np.ndarray): Coefficients
        y (np.ndarray): Targets

    Returns:
        np.ndarray: negative log-likelihood
    """
    Ax = A@x
    return np.sum(np.fmax(Ax,0)+np.log(1 + np.exp(-np.abs(Ax))) - y * Ax)

def logis_loss_grad(A, x, y):
    """Calculates the gradient of the logistic loss.
    
    Adhering to the convention that responses are labeled {0, 1}, the
    log-likelihood is given by:
    $$
        l(x) = - \sum b_i\log p(a_i; x) + (1 - b_i)\log(1 - p(a_i; x))
    $$
    and so its gradient is given by:
    $$
        A^T \left( y - p \right)
    $$
    where each entry of $p$ is given by:
    $$
        p_i = \frac{\exp(a_i^T x)}{1 + \exp(a_i^T x)}
    $$

    Args:
        A (np.ndarray): Feature matrix
        x (np.ndarray): Coefficients

    Returns:
        np.ndarray: the negative gradient of the log-likelihood
            function.
    """
    # print(e.shape)
    return - A.T@(y - safe_sigmoid(A@x))

def newton_loss(dx, grad, B):
    """Calculates the gradient of the loss function for the approximate Newton
    step.

    The approximate Newton step is given by the minimizer of:
    $$
        g^T x + 1/2 ||Bx||_2^2
    $$
    where $B$ is the square root of the sketched Hessian.

    Args:
        dx (np.ndarray): vector of steps in each direction.
        grad (np.ndarray): the current gradient of the negative
            log-likeihood
        SAd (np.ndarray): square root of the sketched Hessian

    Returns:
        np.ndarray: gradient of the objective function outlined above
    """
    return grad.T@dx + 0.5 * np.linalg.norm(B@dx) ** 2
    # return grad.T.dot(dx) + 0.5 * np.linalg.multi_dot([dx.T, B.T, B, dx])

def newton_loss_grad(dx, grad, B):
    """Calculates the gradient of the loss function for the approximate Newton
    step.

    The approximate Newton step is given by the minimizer of:
    $$
        g^T x + 1/2 ||Bx||_2^2
    $$
    where $B$ is the square root of the sketched Hessian.

    It follows that its gradient is given by:
    $$
        g + B^T B x
    $$

    We use this gradient to perform gradient descent.

    Args:
        dx (np.ndarray): vector of steps in each direction.
        grad (np.ndarray): the current gradient of the negative
            log-likeihood
        SAd (np.ndarray): square root of the sketched Hessian

    Returns:
        np.ndarray: gradient of the objective function outlined above
    """
    # return grad + np.linalg.multi_dot([B.T, B, dx])
    p = B.T@(B@dx)
    # print(p.shape)
    # print(grad.shape)
    # print(B.shape)
    return grad+B.T@(B@dx)
