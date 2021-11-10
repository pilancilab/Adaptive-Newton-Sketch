from plot_figure_utils import *
import os

figsize=(16,4)

mu = 1e1
m_list = [200,400]
lbdtol_list=  [1.,2.]
data_name = 'a8a-kernel-n10000-d10000'

alg_list = []
NSA = {'path':'./results/{}-mu{:.1e}/Newton-Sketch-ada-m10-lbdtol{}-SJLT-s1.p'.format(data_name,mu,.5),
   'name':'NS-ada-SJLT'}
alg_list.append(NSA)
NSA = {'path':'./results/{}-mu{:.1e}/Newton-Sketch-ada-m10-lbdtol{}-RRS.p'.format(data_name,mu,.5),
   'name':'NS-ada-RRS'}
alg_list.append(NSA)

m = 100
NS = {'path':'./results/{}-mu{:.1e}/Newton-Sketch-m{}-SJLT-s1.p'.format(data_name, mu,m),
   'name':'NS-SJLT'}
alg_list.append(NS)
m = 800
NS = {'path':'./results/{}-mu{:.1e}/Newton-Sketch-m{}-RRS.p'.format(data_name, mu,m),
   'name':'NS-RRS'}
alg_list.append(NS)

NE = {'path':'./results/{}-mu{:.1e}/Newton.p'.format(data_name, mu),
       'name':'Newton'}
alg_list.append(NE)

GD = {'path':'./results/{}-mu{:.1e}/Gradient-Descent-lr1.0e+00-LS.p'.format(data_name, mu),
       'name':'GD-LS'}
alg_list.append(GD)

NAG = {'path':'./results/{}-mu{:.1e}/NAG-lr1.0e+00-LS.p'.format(data_name, mu),
       'name':'NAG-LS'}
alg_list.append(NAG)

contents = ['relative-error', 'tilde-lambda', 'train-acc', 'test-acc', 'sketch-dim', 'tau']
savedir = './results/{}-mu{:.1e}/figures/'.format(data_name, mu)

if not os.path.exists(savedir):
    os.makedirs(savedir)
iter_max = 100
time_max = 200
iter_max_acc = 50
time_max_acc = 100
for content in contents:
    plot_alg(alg_list, savedir, content, figsize=figsize, iter_max = iter_max, time_max = time_max,
        iter_max_acc=iter_max_acc, time_max_acc=time_max_acc)
