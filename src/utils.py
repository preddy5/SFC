
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from scipy import signal

from prettytable import PrettyTable

def get_mgrid_np(sidelen):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    x = np.linspace(0, sidelen-1, sidelen, dtype=int)
    y = np.linspace(0, sidelen-1, sidelen, dtype=int)
    xv, yv = np.meshgrid(x, y)
    mgrid = np.stack([xv, yv], axis=-1)
    return mgrid


def get_1Dgrid_np(sidelen):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    x = np.linspace(0, sidelen-1, sidelen, dtype=int)
    return x

def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    pix_distance = 1/sidelen
    tensors = tuple(dim * [torch.linspace(-1+pix_distance, 1-pix_distance, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    # mgrid = mgrid.reshape(-1, dim)
    return mgrid

def update_progress(bar, info):
    message = ''
    for k in info.keys():
        message += '{}: {:.4f} '.format(k, info[k])
    bar.set_description(message)
    bar.refresh() # to show immediately the update

def save_tensors(filename, tensors_list, nrows=8, normalize=True):
    with torch.no_grad():
        all_rows = []
        for i in tensors_list:
            i = i.detach()
            # mu = torch.mean(i, dim=(2, 3), keepdim=True)
            # sd = torch.std(i, dim=(2, 3), keepdim=True)
            # i = (i - mu) / sd
            all_rows.append(torchvision.utils.make_grid(i, nrow=nrows, normalize=normalize, pad_value=1))
        visualization = torch.cat(all_rows, dim=2).permute(1,2,0).detach().cpu().numpy()
        if not normalize:
            visualization[visualization < 0] = 0
            visualization[visualization > 1] = 1
    plt.imsave(filename, visualization)

def save_indv_parts(folder, tensor, epoch):
    tensor_np = tensor.permute(0, 2, 3, 1).detach().cpu().numpy()
    num_parts =  tensor_np.shape[0]
    filename_template = folder + '/{}_{}.png'
    if tensor_np.shape[-1]==2:
        tensor_np = tensor_np[:,:,:,0]*tensor_np[:,:,:,1]
    for i in range(num_parts):
        plt.imsave(filename_template.format(epoch, i), tensor_np[i], vmin=tensor_np.min(), vmax=tensor_np.max())

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def make_tensor(t, dtype=torch.float32):
    return torch.tensor(t, dtype=dtype)

def create_gaussian_weights(img_size, n_channels, std):
    # borrowed from https://github.com/monniert/dti-sprites/
    g1d_h = signal.gaussian(img_size[0], std)
    g1d_w = signal.gaussian(img_size[1], std)
    g2d = np.outer(g1d_h, g1d_w)
    return torch.from_numpy(g2d).unsqueeze(0).expand(n_channels, -1, -1).float()

def inv_hist(loss):
    hist, bin_edges = np.histogram(loss, bins=np.arange(0,1.1,0.01), density=False)
    hist = hist.sum()/(hist+1)
    # hist = 1 - hist/hist.sum()

    inds = np.digitize(loss, bin_edges[1:-1])
    weights = hist[inds]
    return weights

def addperm(x,l):
    return [ l[0:i] + [x] + l[i:]  for i in range(len(l)+1) ]

def permute(l):
    if len(l) == 0:
        return [[]]
    return [x for y in permute(l[1:]) for x in addperm(l[0],y) ]