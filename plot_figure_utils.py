#plot_figure_utils.py
import pickle
import numpy as np
import matplotlib.pyplot as plt

def get_info(path):
    _, log, _ = pickle.load(open(path,'rb'))
    loss = log[:,0]
    tlbd = log[:,2]
    train_acc = log[:,3]
    test_acc = log[:,4]
    times = log[:,5]+log[:,6]
    m = log[:,7]
    tau = log[:,8]
    iters = np.arange(len(loss))
    times = np.cumsum(times)
    return iters, loss, tlbd, train_acc, test_acc, times, m, tau

def plot_alg(alg_list, savedir, content='relative-error',figsize=(12,4), 
             iter_max = -1, time_max=-1, iter_max_acc = -1, time_max_acc=-1):
    fig,ax = plt.subplots(1,2,figsize=figsize)
    if content=='relative-error':
        loss_ref = -1
        for alg in alg_list:
            path = alg['path']
            iters, loss, tlbd, train_acc, test_acc, times, m, tau = get_info(path)
            if loss_ref<0:
                loss_ref = np.min(loss)
            else:
                loss_ref = min(loss_ref, np.min(loss))
        loss_ref = loss_ref-1e-6
        print(loss_ref)

    for alg in alg_list:
        path = alg['path']
        name = alg['name']
        iters, loss, tlbd, train_acc, test_acc, times, m, tau = get_info(path)
        tlbd = np.abs(tlbd)
        if content=='relative-error':
            relerr = np.abs(loss-loss_ref)/(1+np.abs(loss_ref)) 
            ax[0].plot(iters,relerr,label=name)
            ax[1].plot(times,relerr,label=name)
            ax[0].set_yscale('log')
            ax[1].set_yscale('log')
        elif content=='tilde-lambda':
            if 'GD' in name or 'NAG' in name:
                continue
            ax[0].plot(iters,tlbd,label=name)
            ax[1].plot(times,tlbd,label=name)
            ax[0].set_yscale('log')
            ax[1].set_yscale('log')
        elif content=='train-acc':
            ax[0].plot(iters,1-train_acc+1e-16,label=name)
            ax[1].plot(times,1-train_acc+1e-16,label=name)
            ax[0].set_yscale('log')
            ax[1].set_yscale('log')
        elif content=='test-acc':
            ax[0].plot(iters,1-test_acc+1e-16,label=name)
            ax[1].plot(times,1-test_acc+1e-16,label=name)
            ax[0].set_yscale('log')
            ax[1].set_yscale('log')
        elif content=='sketch-dim':
            if 'GD' in name or 'NAG' in name:
                continue
            ax[0].plot(iters, m, label=name)
            ax[1].plot(times, m, label=name)
            ax[0].set_yscale('log')
            ax[1].set_yscale('log')
        elif content=='tau':
            ax[0].plot(iters, tau, label=name)
            ax[1].plot(times, tau, label=name)
            ax[0].set_yscale('log')
            ax[1].set_yscale('log')

    ax[0].legend()
    ax[0].set_xlabel('iteration')
    for j in range(2):
        if content=='train-acc':
            ax[j].set_title('train-error')
        elif content=='test-acc':
            ax[j].set_title('test-error')
        else:
            ax[j].set_title(content)
    ax[1].legend()
    ax[1].set_xlabel('time')
    if content == 'relative-error':
        if iter_max>0:
            ax[0].set_xlim(0,iter_max)
        if time_max>0:
            ax[1].set_xlim(0,time_max)
    if content in ['train-acc','test-acc']:
        if iter_max_acc>0:
            ax[0].set_xlim(0,iter_max_acc)
        if time_max_acc>0:
            ax[1].set_xlim(0,time_max_acc)

    fig.savefig('{}/{}.png'.format(savedir, content))