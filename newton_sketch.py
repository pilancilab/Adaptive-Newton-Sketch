from optim import *
from predict import *
from helpers import *

import numpy as np

class SketchedNewtonLogisticRegression:
    @auto_assign
    def __init__(self,
            sketch_type='Gaussian',
            sketch_size=None,
            max_iter=10,
            mu = 1e-2,
            opt = 'cg',
            verbose=False,
            track_progress=False,
            f_ref=None,
            tolerance = 1e-6,
            ada_m = False,
            lbd_tol = 1,
            lbd_tol2 = 1,
            sparsity = 1,
            lbd_power = 1
            ):
        pass

    def fit(self, A, y, Atest, ytest):
        if self.sketch_size is None:
            self.sketch_size = int(np.sqrt(A.shape[0]))
        self.coef_, self.nprog_, self.result_ = run_sketched_newton(
                A, y, Atest, ytest,
                self.sketch_type, self.sketch_size, self.max_iter, self.mu,
                self.opt, self.verbose, self.track_progress, self.f_ref, self.tolerance,
                self.ada_m, self.lbd_tol, self.lbd_tol2, self.sparsity, self.lbd_power)
        return

    def predict(self, A):
        if not hasattr(self, 'coef_'):
            raise Exception('Call fit before prediction')
        return predict_classes(A, self.coef_)

    def predict_proba(self, A):
        if not hasattr(self, 'coef_'):
            raise Exception('Call fit before prediction')
        return predict_proba(A, self.coef_)

    def score(self, A, y):
        if not hasattr(self, 'coef_'):
            raise Exception('Call fit before prediction')
        preds = predict_classes(A, self.coef_)
        return np.mean(y == preds)

# For performance comparison purposes
class GradientDescentLogisticRegression:
    @auto_assign
    def __init__(self,
            max_iter=10,
            mu = 1e-1,
            lr = 5e-5,
            tol = 1e-3,
            verbose=False,
            track_progress=False,
            interval=10,
            f_ref = None,
            use_line_search = False,
            cstop = 1,
            ):
        pass

    def fit(self, A, y, Atest, ytest):
        self.coef_, self.nprog_, self.result_ = logis_gradient_descent(
                A, y, Atest, ytest, 
                self.max_iter, self.mu, self.lr, self.tol,
                self.verbose, self.track_progress, self.interval,
                self.f_ref, self.use_line_search, self.cstop)
        return self

    def predict(self, A):
        if not hasattr(self, 'coef_'):
            raise Exception('Call fit before prediction')
        return predict_classes(A, self.coef_)

    def predict_proba(self, A):
        if not hasattr(self, 'coef_'):
            raise Exception('Call fit before prediction')
        return predict_proba(A, self.coef_)

    def score(self, A, y):
        if not hasattr(self, 'coef_'):
            raise Exception('Call fit before prediction')
        preds = predict_classes(A, self.coef_)
        return np.mean(y == preds)

class AcceleratedGradientDescentLogisticRegression:
    @auto_assign
    def __init__(self,
            max_iter=10,
            mu = 1e-1,
            lr = 5e-5,
            tol = 1e-3,
            verbose=False,
            track_progress=False,
            interval=10,
            f_ref = None,
            use_line_search = False,
            cstop = 1,
            ):
        pass

    def fit(self, A, y, Atest, ytest):
        self.coef_, self.nprog_, self.result_ = logis_accelerated_gradient_descent(
                A, y, Atest, ytest, 
                self.max_iter, self.mu, self.lr, self.tol,
                self.verbose, self.track_progress, self.interval,
                self.f_ref, self.use_line_search, self.cstop)
        return self

    def predict(self, A):
        if not hasattr(self, 'coef_'):
            raise Exception('Call fit before prediction')
        return predict_classes(A, self.coef_)

    def predict_proba(self, A):
        if not hasattr(self, 'coef_'):
            raise Exception('Call fit before prediction')
        return predict_proba(A, self.coef_)

    def score(self, A, y):
        if not hasattr(self, 'coef_'):
            raise Exception('Call fit before prediction')
        preds = predict_classes(A, self.coef_)
        return np.mean(y == preds)
