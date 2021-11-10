from functools import partial
from time import time

from loss_funcs import *
from second_order import *
from gen_sketch import *
from predict import *

import numpy as np

import scipy.sparse
import scipy.sparse.linalg

def line_search(x, dx, g, dg, a, b, max_iter=50):
    """Perform backtracking line search.

    Backtracking line search begins with an initial step-size dx and backtracks
    until the adjusted linear estimate overestimates the loss function $g$.
    For more information refer to pgs. 464-466 of Convex Optimization by Boyd.

    Args:
        x (np.ndarray): Coefficients
        dx (np.ndarray): Step direction
        g (function): Loss function
        dg (function): Loss function gradient
        a (numeric): scaling factor
        b (numeric): reduction factor

    Returns:
        float
    """
    tau = 1
    dgdx = dg(x).T.dot(dx)
    iter_num = 0
    while (g(x) + tau * a * dgdx < g(x + tau * dx)) and iter_num<max_iter:
        tau = tau * b
        iter_num += 1
    return tau

def line_search_step(A, x, dx, y, mu):
    a, b = 0.1, 0.5
    g = lambda x: logis_loss(A, x, y)+ 0.5*mu*np.linalg.norm(x)**2
    dg = lambda x: logis_loss_grad(A, x, y)+mu*x
    tau = line_search(x, dx, g, dg, a, b)
    return x + tau * dx, tau

def run_sketched_newton(
        A, y, Atest, ytest,
        sketch_type, m,
        nmax_iter, mu,
        opt, verbose, track_progress,
        f_ref, tolerance,
        ada_m, lbd_tol, lbd_tol2,
        sparsity, lbd_power):
    n = A.shape[0]
    d = A.shape[1]

    y = y.reshape([-1,1])
    ytest = ytest.reshape([-1,1])

    # Initialize weights vector
    x = np.zeros(d).reshape(-1, 1)

    nprog = np.zeros([nmax_iter, 9])
    sketch_flag = True

    if verbose:
        print(' t       m          loss        relerr       grad       tlbd       tau     train_acc test_acc  ntime  gtime')
    t = 0
    while t<nmax_iter:
        # Track total time per iteration (without gradient step)
        ntime = 0
        nstart = time()
        dx = np.zeros(d).reshape(-1, 1)
        # Create sketched Hessian square root
        if sketch_type == False:
            B = logis_hessian_sqrt(A, x, y)
            m = n
        else:
            if sketch_flag:
                S = gen_sketch_mat(m, n, sketch_type, sparsity=sparsity)
            B = logis_sketched_hessian_sqrt(A, x, y, S)
            # sketch_flag = False

        # print(B.shape)
        # Get gradient of logistic loss
        grad = logis_loss_grad(A, x, y)+mu*x
        # Add time elapsed
        ntime += time() - nstart

        # if m>d:
        #     opt = 'direct'
        # else:
        #     opt = 'smw'
        # print(opt)
        dx, gtime = solve_inner(B, grad, mu, opt)

        # print(dx.shape)

        # Time this block too
        nstart = time()
        x_past = x
        x, tau = line_search_step(A, x, dx, y, mu)
        # print('finish')
        loss = logis_loss(A, x, y) + 0.5*mu*np.linalg.norm(x)**2
        grad_nrm = np.linalg.norm(grad)
        tlbd = -grad.T@dx

        relerr = -1
        if f_ref is not None:
            relerr = np.abs(loss-f_ref)/(1+abs(f_ref))
        # Time this block too
        ntime += time() - nstart

        train_acc = evaluate_acc(A, x, y)
        test_acc = evaluate_acc(Atest, x, ytest)
        if verbose:
            print('{:2d}     {:.1e}     {:.2e}    {:.2e}    {:.2e}   {:.2e}   {:.2e}   {:.2f}    {:.2f}    {:.3f}   {:.3f}'.format(
                t, m, loss, relerr, grad_nrm, tlbd.item(), tau, train_acc*100, test_acc*100, ntime, gtime))
        if track_progress:
            nprog[t, :] = loss, grad_nrm, tlbd.item(), train_acc, test_acc, ntime, gtime, m, tau

        if np.abs(tlbd)<tolerance:
            nprog = nprog[:(t+1),:]
            break

        if ada_m and (sketch_type is not False) and t>0:
            # print('{:.2e} {:.2e}'.format(tlbd.item(), tlbd_past.item()))
            if tlbd-lbd_tol*tlbd_past*min(1,lbd_tol2*tlbd_past**(lbd_power-1))>0:
                if m<=0.25*n:
                    m = 2*m
                else:
                    sketch_type=False
                # x = x_past
                sketch_flag = True
            
        tlbd_past = tlbd

        t = t+1

    result = {}
    if track_progress:
        iter_num = t
        time_n = np.sum(nprog[:,5])
        time_g = np.sum(nprog[:,6])
        time_all = time_n+time_g
        result = {'iter_num':iter_num, 'sketch_dim':m, 'time_all': time_all,'time_n': time_n,'time_g':time_g, 'loss': loss, 'train_acc': train_acc, 'test_acc': test_acc}
        if verbose:
            print('Summary. iter_num: {:2d} sketch_dim: {:.1e} time_all: {:.2f} time_g: {:.2f} loss: {:.2e} train_acc: {:.2f} test_acc: {:2f}'.format(
                t, m, time_all, time_g, loss, train_acc, test_acc))

    return x, nprog, result

def solve_inner(B, grad, mu, opt):
    m, d = B.shape
    # print(m)
    if m>d:
        gtime = 0
        gstart = time()
        if opt=='native':
            if scipy.sparse.issparse(B):
                dx = -scipy.sparse.linalg.spsolve(B.T@B+mu*scipy.sparse.eye(d), grad)
            else:
                dx = -np.linalg.solve(B.T@B+mu*np.eye(d), grad)
        elif opt=='cg':
            dx, _ = scipy.sparse.linalg.cg(B.T@B+mu*scipy.sparse.eye(d), grad)
            dx = -dx
        # dx = -grad
        dx = dx.reshape([-1,1])

        gtime += time() - gstart
    else:
        gtime = 0
        gstart = time()
        if opt=='native':
            if scipy.sparse.issparse(B):
                BBmuI = B@(B.T)+mu*scipy.sparse.eye(m)
                BBinvBgrad = scipy.sparse.linalg.spsolve(BBmuI, B@grad)
            else:
                BBmuI = B@(B.T)+mu*np.eye(m)
                BBinvBgrad = np.linalg.solve(BBmuI, B@grad)
        elif opt=='cg':
            BBmuI = B@(B.T)+mu*scipy.sparse.eye(m)
            BBinvBgrad, _ = scipy.sparse.linalg.cg(BBmuI, B@grad)
        BBinvBgrad = BBinvBgrad.reshape([-1,1])
        dx = -(grad-B.T@BBinvBgrad)/mu
        gtime += time() - gstart
    return dx, gtime

def logis_gradient_descent(
        A, y, Atest, ytest,
        nmax_iter, mu, lr, tol,
        verbose, track_progress,
        interval, f_ref, 
        use_line_search, cstop):
    n = A.shape[0]
    d = A.shape[1]

    x = np.zeros(d).reshape(-1, 1)

    nprog = np.zeros([nmax_iter, 9])

    if verbose:
        print(' t       m          loss        relerr       grad       tlbd       tau     train_acc test_acc  ntime  gtime')
    t = 0
    while t<nmax_iter:
        t0 = time()
        grad = logis_loss_grad(A, x, y)+mu*x
        grad = grad.reshape([-1,1])

        if use_line_search:
            x, tau = line_search_step(A, x, -lr*grad, y, mu)
        else:
            x = x-lr*grad
            tau = 1
        loss = logis_loss(A, x, y) + 0.5*mu*np.linalg.norm(x)**2
        grad_nrm = np.linalg.norm(grad) 

        relerr = -1
        abserr = -1
        if f_ref is not None:
            relerr = np.abs(loss-f_ref)/(1+abs(f_ref))
            abserr = np.abs(loss-f_ref)
        
        ntime = time() - t0

        if t%interval==0:
            train_acc = evaluate_acc(A, x, y)
            test_acc = evaluate_acc(Atest, x, ytest)

        if verbose and t%interval==0:
            print('{:2d}     {:.1e}     {:.2e}    {:.2e}    {:.2e}   {:.2e}   {:.2e}   {:.2f}    {:.2f}    {:.3f}   {:.3f}'.format(
                t, 0, loss, relerr, grad_nrm, 0, lr*tau, train_acc*100, test_acc*100, ntime, 0))
        if track_progress:
            nprog[t, :] = loss, grad_nrm, 0, train_acc, test_acc, ntime, 0, 0, lr*tau

        if cstop ==1:
            if grad_nrm<tol:
                nprog = nprog[:(t+1),:]
                break
        if cstop ==2:
            if abserr<tol:
                nprog = nprog[:(t+1),:]
                break
        t += 1
    result = {}
    if track_progress:
        iter_num = t
        time_n = np.sum(nprog[:,5])
        time_g = np.sum(nprog[:,6])
        time_all = time_n+time_g
        result = {'iter_num':iter_num, 'time_all': time_all,'time_n': time_n,'time_g':time_g, 'loss': loss, 'train_acc': train_acc, 'test_acc': test_acc}
        if verbose:
            print('Summary. iter_num: {:2d} time_all: {:.2f} time_g: {:.2f} loss: {:.2e} train_acc: {:.2f} test_acc: {:2f}'.format(
                t, time_all, time_g, loss, train_acc, test_acc))

    return x, nprog, result

def logis_accelerated_gradient_descent(
        A, y, Atest, ytest,
        nmax_iter, mu, lr, tol,
        verbose, track_progress,
        interval, f_ref, 
        use_line_search, cstop):
    n = A.shape[0]
    d = A.shape[1]

    x = np.zeros(d).reshape(-1, 1)
    x_acc = x

    nprog = np.zeros([nmax_iter, 9])

    if verbose:
        print(' t       m          loss        relerr       grad       tlbd       tau     train_acc test_acc  ntime  gtime')
    t = 0
    while t<nmax_iter:
        t0 = time()

        grad = logis_loss_grad(A, x_acc, y)+mu*x_acc
        grad = grad.reshape([-1,1])

        x_old = x
        if use_line_search:
            x, tau = line_search_step(A, x_acc, -lr*grad, y, mu)
        else:
            x = x_acc-lr*grad
            tau = 1

        x_acc = x+t/(t+3)*(x-x_old)

        loss = logis_loss(A, x, y) + 0.5*mu*np.linalg.norm(x)**2
        grad_nrm = np.linalg.norm(grad) 

        relerr = -1
        abserr = -1
        if f_ref is not None:
            relerr = np.abs(loss-f_ref)/(1+abs(f_ref))
            abserr = np.abs(loss-f_ref)
        
        ntime = time() - t0

        if t%interval==0:
            train_acc = evaluate_acc(A, x, y)
            test_acc = evaluate_acc(Atest, x, ytest)

        if verbose and t%interval==0:
            print('{:2d}     {:.1e}     {:.2e}    {:.2e}    {:.2e}   {:.2e}   {:.2e}   {:.2f}    {:.2f}    {:.3f}   {:.3f}'.format(
                t, 0, loss, relerr, grad_nrm, 0, lr*tau, train_acc*100, test_acc*100, ntime, 0))
        if track_progress:
            nprog[t, :] = loss, grad_nrm, 0, train_acc, test_acc, ntime, 0, 0, lr*tau

        if cstop ==1:
            if grad_nrm<tol:
                nprog = nprog[:(t+1),:]
                break
        if cstop ==2:
            if abserr<tol:
                nprog = nprog[:(t+1),:]
                break
        t += 1
    result = {}
    if track_progress:
        iter_num = t
        time_n = np.sum(nprog[:,5])
        time_g = np.sum(nprog[:,6])
        time_all = time_n+time_g
        result = {'iter_num':iter_num, 'time_all': time_all,'time_n': time_n,'time_g':time_g, 'loss': loss, 'train_acc': train_acc, 'test_acc': test_acc}
        if verbose:
            print('Summary. iter_num: {:2d} time_all: {:.2f} time_g: {:.2f} loss: {:.2e} train_acc: {:.2f} test_acc: {:2f}'.format(
                t, time_all, time_g, loss, train_acc, test_acc))

    return x, nprog, result

