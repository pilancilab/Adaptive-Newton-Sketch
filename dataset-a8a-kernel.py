import sklearn.datasets
import numpy as np
from tqdm import tqdm
import scipy.sparse
import pickle
import time

print('start')
data_dir = './datasets'

def build_kernel_matrix(A, B, kernel_type='Gaussian', kernel_opt = {}):
    if kernel_opt.get('bandwidth',-1)==-1:
        kernel_opt['bandwidth'] = -1
    n, d = A.shape
    A_sum = np.sum(A**2,axis=1)
    B_sum = np.sum(B**2,axis=1)
    if kernel_type == 'Gaussian':
        dist_mat = -2*np.matmul(B, A.T)+B_sum.reshape([-1,1])+A_sum.reshape([1,-1])
        bandwidth = kernel_opt['bandwidth']
        if bandwidth == -1:
            bandwidth = np.median(dist_mat)/2/np.log(d+1)
        K = np.exp(-dist_mat*0.5/bandwidth)
    return K, bandwidth

np.random.seed(1)
A,b = sklearn.datasets.load_svmlight_file('{}/a8a/a8a'.format(data_dir))
A = A.A
indn = np.arange(22696)
np.random.shuffle(indn)
Atrain = A[indn[:10000],:]
Atest = A[indn[10000:],:]
btest = b[indn[10000:]]
btrain = b[indn[:10000]]

kernel_opt = {'bandwidth':10}

Ktrain, bandwidth = build_kernel_matrix(Atrain,Atrain, kernel_opt = kernel_opt)
print(bandwidth)
kernel_opt = {'bandwidth':bandwidth}
Ktest, _ =  build_kernel_matrix(Atrain,Atest, kernel_opt = kernel_opt)

with open('{}/a8a/a8a_kernel_train.p'.format(data_dir),'wb') as f:
    K_part = Ktrain
    b_part = btrain
    pickle.dump([K_part,b_part],f)

with open('{}/a8a/a8a_kernel_test.p'.format(data_dir),'wb') as f:
    K_part = Ktest
    b_part = btest
    pickle.dump([K_part,b_part],f)

