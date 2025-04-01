import torch
from torch import nn, Tensor
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

import seaborn as sns
from torchdyn.core import NeuralODE
from torchcfm.utils import torch_wrapper

def sample_conditional_pt(x0: Tensor, x1: Tensor, t: Tensor, sigma: float):
    '''
    Code taken from this tutorial:
    https://github.com/atong01/conditional-flow-matching/blob/main/examples/2D_tutorials/Flow_matching_tutorial.ipynb

    Draw a sample from the probability path N(t * x1 + (1 - t) * x0, sigma), see (Eq.14) [1].

    Parameters
    ----------
    x0 : Tensor, shape (bs, *dim)
        represents the source minibatch
    x1 : Tensor, shape (bs, *dim)
        represents the target minibatch
    t : FloatTensor, shape (bs)

    Returns
    -------
    xt : Tensor, shape (bs, *dim)

    References
    ----------
    [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
    '''
    t = t.reshape(-1, *([1] * (x0.dim() - 1)))
    mu_t = t * x1 + (1 - t) * x0
    epsilon = torch.randn_like(x0)
    return mu_t + sigma * epsilon

def sample_x0(batch_size: int):
    '''
    Simple source distribution.
    '''
    x0 = torch.randn(batch_size)
    return x0.unsqueeze(-1)

def sample_x1(batch_size: int):
    '''
    Some complex distribution.
    '''
    mode = torch.randint(0, 2, (batch_size,))  # Choose mode 0 or 1
    x1 = 0.5 * torch.randn(batch_size) + torch.where(mode == 0, -3.0, 3.0)
    return x1.unsqueeze(-1)

if __name__ == '__main__':
    ## define network
    h_size = 32
    batch_size = 128
    flow = nn.Sequential(
        nn.Linear(2, h_size),
        nn.ReLU(),
        nn.Linear(h_size, h_size),
        nn.ReLU(),
        nn.Linear(h_size, 1))

    opt = torch.optim.Adam(flow.parameters(), lr = 1e-3)

    ## train
    iterations = 300
    loss_hist = []
    for i in range(iterations):
        opt.zero_grad()

        ## sample source and target distributions
        x0 = sample_x0(batch_size)
        x1 = sample_x1(batch_size)
        t = torch.rand(batch_size)
        xt = sample_conditional_pt(x0, x1, t, sigma = 0.01)

        ## predict velocity field at time t
        vt = flow(torch.cat([xt, t[:, None]], dim = 1))
        loss = torch.mean(torch.square(vt - (x1 - x0)))
        loss.backward()
        opt.step()

        ## log
        loss_hist.append(loss.item())
        print(f'Iteration: {i} Loss: {loss.item()}')

    ## propagate dynamical system
    node = NeuralODE(torch_wrapper(flow), solver = 'dopri5', sensitivity = 'adjoint', atol = 1e-4, rtol = 1e-4)
    timesteps = 100
    t_span = torch.linspace(0, 1, timesteps)
    with torch.no_grad():
        traj = node.trajectory(
            sample_x0(20000),
            t_span = t_span).squeeze()
        
    ## create a 2d heatmap of the flow
    ## traj is (time x samples)
    bins = 100  # Number of bins for the heatmap
    heatmap, _, _ = np.histogram2d(
        traj.flatten(),
        np.repeat(t_span, traj.shape[1]),
        bins = [bins, timesteps],
        range = [[traj.min(), traj.max()], [0.0, 1.0]]  # Range for x and t
    )

    ## normalizing heatmap
    heatmap = heatmap / heatmap.sum()
    
    ## grabbing source and target distributions from the trajectory
    x0 = traj[0, :]
    x1 = traj[-1, :]

    '''
    The code below is for generating the plot.
    '''
    fig = plt.figure()
    gs = GridSpec(1, 3, width_ratios=[1, 4, 1], wspace = 0.05)
    
    # Left KDE plot (for x0)
    ax_left = plt.subplot(gs[0])
    sns.kdeplot(y = x0, ax = ax_left, fill = True, color = 'blue')
    ax_left.invert_xaxis()
    ax_left.set_xticks([])
    ax_left.set_yticks([])
    ax_left.set_xlabel('')
    ax_left.set_ylabel('')
    ax_left.spines[['top', 'right', 'bottom', 'left']].set_visible(False)  # Clean up

    # Center Heatmap
    ax_center = plt.subplot(gs[1])
    ax_center.set_xticks([])
    ax_center.set_yticks([])
    ax_center.set_xticklabels([])
    ax_center.set_yticklabels([])
    ax_center.imshow(heatmap, aspect='auto', cmap = 'inferno')

    # Right KDE plot (for x1)
    ax_right = plt.subplot(gs[2])
    sns.kdeplot(y = x1, ax=ax_right, fill=True, color='red')
    ax_right.set_xticks([])
    ax_right.set_yticks([])
    ax_right.set_xlabel('')
    ax_right.set_ylabel('')
    ax_right.spines[['top', 'right', 'bottom', 'left']].set_visible(False)  # Clean up
    fig.savefig('flow.png', dpi = 300)
    plt.show()