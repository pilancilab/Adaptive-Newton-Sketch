# gen_loss_ref.py

from plot_figure_utils import get_info
import pickle, os
import numpy as np

def get_loss_ref(alg_list):
    loss_ref = -1
    for alg in alg_list:
        path = alg['path']
        iters, loss, tlbd, train_acc, test_acc, times, m, tau = get_info(path)
        if loss_ref<0:
            loss_ref = np.min(loss)
        else:
            loss_ref = min(loss_ref, np.min(loss))
    loss_ref = loss_ref-1e-7
    return loss_ref

save_dir = 'loss_ref/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# gisette
mu = 1e-3
data_name = 'gisette-n3000-d5000'

alg_list = []
NSA = {'path':'./results/{}-mu{:.1e}/Newton-Sketch-ada-m10-lbdtol{}-SJLT-s1.p'.format(data_name,mu,2.),
   'name':'NS-ada-SJLT'}
alg_list.append(NSA)
NSA = {'path':'./results/{}-mu{:.1e}/Newton-Sketch-ada-m10-lbdtol{}-RRS.p'.format(data_name,mu,2.),
   'name':'NS-ada-RRS'}
alg_list.append(NSA)

loss_ref = get_loss_ref(alg_list)
save_name = 'loss_ref/{}-mu{:.1e}-loss-ref.p'.format(data_name,mu)
pickle.dump([loss_ref],open(save_name,'wb'))

# rcv1
mu = 1e-3
data_name = 'rcv1-n10000-d47236'

NSA = {'path':'./results/{}-mu{:.1e}/Newton-Sketch-ada-m100-lbdtol2.0-SJLT-s1.p'.format(data_name, mu),
       'name':'NS-ada-SJLT'}
NSA2 = {'path':'./results/{}-mu{:.1e}/Newton-Sketch-ada-m100-lbdtol1.0-RRS.p'.format(data_name, mu),
       'name':'NS-ada-RRS'}
alg_list = [NSA, NSA2]

loss_ref = get_loss_ref(alg_list)
save_name = 'loss_ref/{}-mu{:.1e}-loss-ref.p'.format(data_name,mu)
pickle.dump([loss_ref],open(save_name,'wb'))


# mnist
mu = 1e-1
data_name = 'MNIST-n30000-d780'

alg_list = []
NSA = {'path':'./results/{}-mu{:.1e}/Newton-Sketch-ada-m100-lbdtol0.5-lbdpow2.0-lbdtol2-6.0-SJLT-s1.p'.format(data_name, mu),
       'name':'NS-ada-SJLT'}
alg_list.append(NSA)
NSA = {'path':'./results/{}-mu{:.1e}/Newton-Sketch-ada-m100-lbdtol0.5-lbdpow2.0-lbdtol2-6.0-RRS.p'.format(data_name, mu),
       'name':'NS-ada-SJLT'}
alg_list.append(NSA)

loss_ref = get_loss_ref(alg_list)
save_name = 'loss_ref/{}-mu{:.1e}-loss-ref.p'.format(data_name,mu)
pickle.dump([loss_ref],open(save_name,'wb'))

# realsim
mu = 1e-3
data_name = 'realsim-n50000-d20958'

alg_list = []
NSA = {'path':'./results/{}-mu{:.1e}/Newton-Sketch-ada-m100-lbdtol{}-SJLT-s1.p'.format(data_name,mu,2.),
   'name':'NS-ada-SJLT'}
alg_list.append(NSA)
NSA = {'path':'./results/{}-mu{:.1e}/Newton-Sketch-ada-m100-lbdtol{}-RRS.p'.format(data_name,mu,2.),
   'name':'NS-ada-RRS'}
alg_list.append(NSA)

loss_ref = get_loss_ref(alg_list)
save_name = 'loss_ref/{}-mu{:.1e}-loss-ref.p'.format(data_name,mu)
pickle.dump([loss_ref],open(save_name,'wb'))

# epsilon
mu = 1e-1
lbdtol_list=  [1.,2.]
data_name = 'epsilon-n50000-d2000'

alg_list = []
NSA = {'path':'./results/{}-mu{:.1e}/Newton-Sketch-ada-m100-lbdtol{}-SJLT-s1.p'.format(data_name,mu,1.),
   'name':'NS-ada-SJLT'}
alg_list.append(NSA)
NSA = {'path':'./results/{}-mu{:.1e}/Newton-Sketch-ada-m100-lbdtol{}-RRS.p'.format(data_name,mu,1.),
   'name':'NS-ada-RRS'}
alg_list.append(NSA)

loss_ref = get_loss_ref(alg_list)
save_name = 'loss_ref/{}-mu{:.1e}-loss-ref.p'.format(data_name,mu)
pickle.dump([loss_ref],open(save_name,'wb'))

# a8a-kernel
mu = 1e1
data_name = 'a8a-kernel-n10000-d10000'

alg_list = []
NSA = {'path':'./results/{}-mu{:.1e}/Newton-Sketch-ada-m10-lbdtol{}-SJLT-s1.p'.format(data_name,mu,.5),
   'name':'NS-ada-SJLT'}
alg_list.append(NSA)
NSA = {'path':'./results/{}-mu{:.1e}/Newton-Sketch-ada-m10-lbdtol{}-RRS.p'.format(data_name,mu,.5),
   'name':'NS-ada-RRS'}
alg_list.append(NSA)

loss_ref = get_loss_ref(alg_list)
save_name = 'loss_ref/{}-mu{:.1e}-loss-ref.p'.format(data_name,mu)
pickle.dump([loss_ref],open(save_name,'wb'))

# phishing-kernel
mu = 1e1
data_name = 'phishing-kernel-n5000-d5000'

alg_list = []
for lbdtol in lbdtol_list:
    NSA = {'path':'./results/{}-mu{:.1e}/Newton-Sketch-ada-m100-lbdtol{}-SJLT-s1.p'.format(data_name,mu,lbdtol),
       'name':'NS-ada-100-lbdtol{}'.format(lbdtol)}
    alg_list.append(NSA)

loss_ref = get_loss_ref(alg_list)
save_name = 'loss_ref/{}-mu{:.1e}-loss-ref.p'.format(data_name,mu)
pickle.dump([loss_ref],open(save_name,'wb'))