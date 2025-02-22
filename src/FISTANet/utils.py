import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import mlflow


def plot_y_comp(Yn, Yhat, Y, Psi, context, epoch, batch_id, sample_ids):
    fig, axs = plt.subplots(3, 1, num=1, clear=True)
    fig.set_figheight(900/plt.rcParams['figure.dpi'])  
    fig.set_figwidth(600/plt.rcParams['figure.dpi'])
    loss = nn.MSELoss()
    for ai, i in enumerate(sample_ids):
        yn = Yn[i, :, :].cpu().squeeze().detach()
        yhat = Yhat[i, :, :].cpu().squeeze().detach()
        y = Y[i, :, :].cpu().squeeze().detach()
        for yy, lab in [(yn, r'$\widetilde{\boldsymbol{y}}$'),
                        (yhat, r'$\hat{\boldsymbol{y}}$'),
                        (y,  r'$\boldsymbol{y}$')]:
            mse = loss(yy, y)
            axs[ai].plot(yy, label=rf'{lab} (MSE: {mse})', linewidth=0.5)
        axs[ai].legend(prop={'size': 6})
    mlflow.log_figure(fig, f'plots/Y-comp/{context}_ep-{epoch}_batch-{batch_id}.png')
    plt.close()


def plot_x_comp(X0, Xhat, context, epoch, batch_id, sample_ids):
    fig, axs = plt.subplots(3, 2, num=1, clear=True)
    fig.set_figheight(900/plt.rcParams['figure.dpi'])
    fig.set_figwidth(1300/plt.rcParams['figure.dpi'])
    for ai, i in enumerate(sample_ids):
        x0 = X0[i, :, :].cpu().squeeze().detach()
        xhat = Xhat[i, :, :].cpu().squeeze().detach()
        for j in range(2):
            for x, lab in [(x0, r'$\boldsymbol{x}^{(0)}$'),
                           (xhat, r'$\hat{\boldsymbol{x}}$')]:
                axs[ai, j].plot(np.linspace(0, 1, x.shape[0]), x,
                                label=rf'{lab} ' + \
                                      rf'(l0-norm: {torch.sum(x<1e-3)} | ' + \
                                      rf'l1-norm: {torch.mean(torch.abs(x))})',
                                linewidth=0.5)
            axs[ai, j].legend(prop={'size': 6})
        axs[ai, 1].set_ylim(torch.min(xhat), torch.max(xhat))
    mlflow.log_figure(fig, f'plots/X-comp/{context}_ep-{epoch}_batch-{batch_id}.png')
    plt.close()
