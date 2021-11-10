# main.py

import os
import numpy as np
import argparse
import pickle
import sklearn.datasets
from newton_sketch import *

def get_parser():
    parser = argparse.ArgumentParser(description='logistic regression')
    parser.add_argument("--data_name", type=str, default="random",
                        help="data name", choices=[ "rcv1", "gisette", "MNIST", "realsim",
                        "a8a-kernel", "epsilon", "phishing-kernel", "w7a-kernel"])
    parser.add_argument("--n", type=float, default=1e3, help="number of sample")
    parser.add_argument("--optim", type=str, default="Newton-Sketch",
                        choices=["Newton-Sketch", "Newton-Sketch-ada", "GradientDescent",
                        "Newton","NAG"])
    parser.add_argument("--d", type=float, default=1e3, help="number of dimension")
    parser.add_argument("--m", type=float, default=1e3, help="sketch dimension")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--data_dir", type=str, default="../datasets")
    parser.add_argument("--shuffle", action='store_true', help="whether to shuffle the index")
    parser.add_argument("--id", type=str, default="")
    parser.add_argument("--lbdtol", type=float, default=1)
    parser.add_argument("--lbdtol2", type=float, default=1)
    parser.add_argument("--lbdpow", type=float, default=1)
    parser.add_argument("--mu", type=float, default=1e-2)
    parser.add_argument("--max_iter", type=int, default=100)
    parser.add_argument("--sketch_type", type=str, default="SJLT",choices=["SJLT", "Gaussian",
                        "Rademacher", "RRS"])
    parser.add_argument("--sparsity", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1)
    parser.add_argument("--use_line_search",  action='store_true')
    parser.add_argument("--opt",  type=str, default='cg')

    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    print(args)
    data_name = args.data_name
    optim = args.optim
    n = int(args.n)
    d = int(args.d)
    m = int(args.m)
    mu = args.mu

    seed = args.seed
    np.random.seed(seed)

    print('load dataset')
    if data_name == 'rcv1': #20242 47236
        test_name = './results/rcv1-n{}-d{}-mu{:.1e}/'.format(n,d,mu)
        A,b = sklearn.datasets.load_svmlight_file('{}/rcv1/rcv1_train.binary'.format(args.data_dir))
        p, q = A.shape
        indn = np.arange(p)
        indd = np.arange(q)
        if args.shuffle:
            np.random.shuffle(indn)
            np.random.shuffle(indd)
        Atrain = A[indn[:n],:][:,indd[:d]]
        Atest = A[indn[n:min(2*n,p)],:][:,indd[:d]]
        A = Atrain
        b = (b+1)/2
        b = b.reshape([-1,1])
        btrain = b[indn[:n]]
        btest = b[indn[n:min(2*n,p)]]
        b = btrain
    elif data_name == 'gisette':#6000 5000
        test_name = './results/gisette-n{}-d{}-mu{:.1e}/'.format(n,d,mu)
        A,b = sklearn.datasets.load_svmlight_file('{}/gisette/gisette_scale'.format(args.data_dir))
        # A = A/np.max(A)
        p, q = A.shape
        indn = np.arange(p)
        indd = np.arange(q)
        if args.shuffle:
            np.random.shuffle(indn)
            np.random.shuffle(indd)
        Atrain = A[indn[:n],:][:,indd[:d]]
        Atest = A[indn[n:min(2*n,p)],:][:,indd[:d]]
        A = Atrain
        b = (b+1)/2
        b = b.reshape([-1,1])
        btrain = b[indn[:n]]
        btest = b[indn[n:min(2*n,p)]]
        b = btrain
    elif data_name == 'MNIST':
        test_name = './results/MNIST-n{}-d{}-mu{:.1e}/'.format(n,d,mu)
        A,b = sklearn.datasets.load_svmlight_file('{}/MNIST/mnist'.format(args.data_dir))
        p, q = A.shape
        A = A/255
        p, q = A.shape
        indn = np.arange(p)
        indd = np.arange(q)
        if args.shuffle:
            np.random.shuffle(indn)
            np.random.shuffle(indd)
        Atrain = A[indn[:n],:][:,indd[:d]]
        Atest = A[indn[n:min(2*n,p)],:][:,indd[:d]]
        A = Atrain
        b = (b%2==0)+0.0
        b = b.reshape([-1,1])
        btrain = b[indn[:n]]
        btest = b[indn[n:min(2*n,p)]]
        b = btrain

    elif data_name == 'realsim':#72309 20958
        test_name = './results/realsim-n{}-d{}-mu{:.1e}/'.format(n,d,mu)
        A,b = sklearn.datasets.load_svmlight_file('{}/realsim/real-sim'.format(args.data_dir))
        p, q = A.shape
        indn = np.arange(p)
        indd = np.arange(q)
        if args.shuffle:
            np.random.shuffle(indn)
            np.random.shuffle(indd)
        Atrain = A[indn[:n],:][:,indd[:d]]
        Atest = A[indn[n:min(2*n,p)],:][:,indd[:d]]
        A = Atrain
        b = (b+1)/2
        b = b.reshape([-1,1])
        btrain = b[indn[:n]]
        btest = b[indn[n:min(2*n,p)]]
        b = btrain
    elif data_name == 'epsilon':#400,000 2,000
        test_name = './results/epsilon-n{}-d{}-mu{:.1e}/'.format(n,d,mu)
        A,b = sklearn.datasets.load_svmlight_file('{}/epsilon/epsilon_normalized'.format(args.data_dir))
        p, q = A.shape
        indn = np.arange(p)
        indd = np.arange(q)
        if args.shuffle:
            np.random.shuffle(indn)
            np.random.shuffle(indd)
        Atrain = A[indn[:n],:][:,indd[:d]]
        Atest = A[indn[-1-n:-1],:][:,indd[:d]]
        A = Atrain
        b = (b+1)/2
        b = b.reshape([-1,1])
        btrain = b[indn[:n]]
        btest = b[indn[-1-n:-1]]
        b = btrain
    elif data_name == 'a8a-kernel':#10,000 10,000
        test_name = './results/a8a-kernel-n{}-d{}-mu{:.1e}/'.format(n,d,mu)
        Atrain, btrain = pickle.load(open('{}/a8a/a8a_kernel_train.p'.format(args.data_dir),'rb'))
        Atest, btest = pickle.load(open('{}/a8a/a8a_kernel_test.p'.format(args.data_dir),'rb'))
        A = Atrain[:n,:][:,:d]
        b = btrain[:n]
        Atest = Atest[:n,:][:,:d]
        btest = btest[:n]
        b = (b+1)/2
        b = b.reshape([-1,1])
        btest = (btest+1)/2
        btest = btest.reshape([-1,1])

    elif data_name == 'w7a-kernel':#12,000 12,000
        test_name = './results/w7a-kernel-n{}-d{}-mu{:.1e}/'.format(n,d,mu)
        Atrain, btrain = pickle.load(open('{}/w7a/w7a_kernel_train.p'.format(args.data_dir),'rb'))
        Atest, btest = pickle.load(open('{}/w7a/w7a_kernel_test.p'.format(args.data_dir),'rb'))
        A = Atrain[:n,:][:,:d]
        b = btrain[:n]
        Atest = Atest[:n,:][:,:d]
        btest = btest[:n]
        b = (b+1)/2
        b = b.reshape([-1,1])
        btest = (btest+1)/2
        btest = btest.reshape([-1,1])

    elif data_name == 'phishing-kernel':#5,000 5,000
        test_name = './results/phishing-kernel-n{}-d{}-mu{:.1e}/'.format(n,d,mu)
        Atrain, btrain = pickle.load(open('{}/phishing/phishing_kernel_train.p'.format(args.data_dir),'rb'))
        Atest, btest = pickle.load(open('{}/phishing/phishing_kernel_test.p'.format(args.data_dir),'rb'))
        A = Atrain[:n,:][:,:d]
        b = btrain[:n]
        Atest = Atest[:n,:][:,:d]
        btest = btest[:n]
        b = (b+1)/2
        b = b.reshape([-1,1])
        btest = (btest+1)/2
        btest = btest.reshape([-1,1])

    print(A.shape)
    if not os.path.exists(test_name):
        os.makedirs(test_name)

    if optim == 'Newton-Sketch':
        method_name = "Newton-Sketch-m{}".format(m)

    if optim == 'Newton-Sketch-ada':
        if args.lbdpow>1:
            method_name = "Newton-Sketch-ada-m{}-lbdtol{}-lbdpow{}-lbdtol2-{}".format(m, args.lbdtol, args.lbdpow, args.lbdtol2)
        else:
            method_name = "Newton-Sketch-ada-m{}-lbdtol{}".format(m, args.lbdtol)
    if optim == 'Newton':
        method_name = "Newton"

    if optim == 'GradientDescent':
        if args.use_line_search:
            method_name = "Gradient-Descent-lr{:.1e}-LS".format(args.lr)
        else:
            method_name = "Gradient-Descent-lr{:.1e}".format(args.lr)

    if optim == 'NAG':
        if args.use_line_search:
            method_name = "NAG-lr{:.1e}-LS".format(args.lr)
        else:
            method_name = "NAG-lr{:.1e}".format(args.lr)

    if optim == 'Newton-Sketch' or optim == 'Newton-Sketch-ada':
        sketch_name = args.sketch_type
        if args.sketch_type=='SJLT':
            sketch_name = sketch_name+'-s{}'.format(args.sparsity)
        method_name = '{}-{}'.format(method_name, sketch_name)
    if len(args.id)>0:
        method_name = method_name+'-{}'.format(args.id)

    print('{}{}'.format(test_name, method_name))

    if optim == 'Newton-Sketch':
        solver = SketchedNewtonLogisticRegression(sketch_size = m, track_progress=True,
                                          verbose=True, max_iter=args.max_iter,
                                          sketch_type=args.sketch_type, sparsity=args.sparsity,
                                          mu=mu, ada_m = False, opt=args.opt)
    elif optim == 'Newton-Sketch-ada':
        solver = SketchedNewtonLogisticRegression(sketch_size = m, track_progress=True,
                                          verbose=True, max_iter=args.max_iter,
                                          sketch_type=args.sketch_type, sparsity=args.sparsity,
                                          mu=mu, ada_m = True, 
                                          lbd_tol=args.lbdtol, lbd_power=args.lbdpow, opt=args.opt)
    elif optim == 'Newton':
        solver = SketchedNewtonLogisticRegression(sketch_size = m, track_progress=True,
                                          verbose=True, max_iter=args.max_iter,
                                          sketch_type=False, mu=mu, opt=args.opt)
    elif optim == 'GradientDescent':
        loss_ref_name = '{}/{}/{}-n{}-d{}-mu{:.1e}-loss-ref.p'.format(args.data_dir, data_name, data_name, n,d,mu)
        loss_ref = pickle.load(open(loss_ref_name,'rb'))[0]
        solver = GradientDescentLogisticRegression(lr=args.lr, use_line_search=args.use_line_search, mu=mu, 
                                                    cstop=2,tol=1e-6, verbose=True, track_progress=True, 
                                                    max_iter=args.max_iter, f_ref=loss_ref)

    elif optim == 'NAG':
        loss_ref_name = '{}/{}/{}-n{}-d{}-mu{:.1e}-loss-ref.p'.format(args.data_dir, data_name, data_name, n,d,mu)
        loss_ref = pickle.load(open(loss_ref_name,'rb'))[0]
        solver = AcceleratedGradientDescentLogisticRegression(lr=args.lr, use_line_search=args.use_line_search, mu=mu, 
                                                    cstop=2,tol=1e-6, verbose=True, track_progress=True, 
                                                    max_iter=args.max_iter, f_ref=loss_ref)

    solver.fit(A, b, Atest, btest)

    save_name = '{}{}.p'.format(test_name, method_name)

    results = [solver.coef_, solver.nprog_, solver.result_]
    pickle.dump(results,open(save_name,'wb'))


if __name__ == '__main__':
    main()