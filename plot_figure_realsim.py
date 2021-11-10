import matplotlib.pyplot as plt 
plt.rcParams.update({'font.size': 20})
from plot_figure_utils import *
import os

figsize=(16,6)

mu = 1e-3
data_name = 'realsim-n50000-d20958'

alg_list = []

NSA = {'path':'./results/{}-mu{:.1e}/Newton-Sketch-ada-m100-lbdtol{}-SJLT-s1.p'.format(data_name,mu,2.),
   'name':'NS-ada-SJLT'}
alg_list.append(NSA)
NSA = {'path':'./results/{}-mu{:.1e}/Newton-Sketch-ada-m100-lbdtol{}-RRS.p'.format(data_name,mu,2.),
   'name':'NS-ada-RRS'}
alg_list.append(NSA)

m=800
NS = {'path':'./results/{}-mu{:.1e}/Newton-Sketch-m{}-SJLT-s1.p'.format(data_name, mu,m),
   'name':'NS-SJLT'}
alg_list.append(NS)
m=3200
NS = {'path':'./results/{}-mu{:.1e}/Newton-Sketch-m{}-RRS.p'.format(data_name, mu,m),
   'name':'NS-RRS'}
alg_list.append(NS)

GD = {'path':'./results/{}-mu{:.1e}/Gradient-Descent-lr1.0e+00-LS.p'.format(data_name, mu),
       'name':'GD-LS'}
alg_list.append(GD)

NAG = {'path':'./results/{}-mu{:.1e}/NAG-lr1.0e+00-LS.p'.format(data_name, mu),
       'name':'NAG-LS'}
alg_list.append(NAG)

contents = ['relative-error', 'tilde-lambda', 'train-acc', 'test-acc', 'sketch-dim']
savedir = './results/{}-mu{:.1e}/figures/'.format(data_name, mu)

if not os.path.exists(savedir):
    os.makedirs(savedir)
iter_max = 800
time_max = 800
iter_max_acc = 30
time_max_acc = 50
for content in contents:
    plot_alg(alg_list, savedir, content, figsize=figsize, iter_max = iter_max, time_max = time_max,
        iter_max_acc=iter_max_acc, time_max_acc=time_max_acc)
