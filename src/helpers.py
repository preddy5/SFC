
import torch


def hard_composite(layers, shape=[8,3,64,64], return_alpha=False):
    n = len(layers)
    if n==0:
        if return_alpha:
            zeros = torch.zeros(shape).cuda()
            return zeros, zeros
        return torch.zeros(shape).cuda()
    n_channels = layers[0].shape[1]-1
    alpha = (1 - layers[0][:, n_channels:, :, :])
    rgb = layers[0][:, :n_channels] * layers[0][:, n_channels:, :, :]
    for i in range(1, n):
        rgb = rgb + layers[i][:, :n_channels] * layers[i][:, n_channels:, :, :] * alpha
        alpha = (1-layers[i][:, n_channels:, :, :]) * alpha
    if return_alpha:
        return rgb, 1 - alpha.repeat(1, rgb.shape[1], 1, 1)
    return rgb

# def hard_composite(layers, shape=[8,3,64,64], return_alpha=False):
#     n = len(layers)
#     if n==0:
#         return torch.zeros(shape).cuda()
#     n_channels = layers[0].shape[1]-1
#     # alpha = (1 - layers[0][:, n_channels:, :, :])
#     rgb = layers[0][:, :n_channels] * layers[0][:, n_channels:, :, :]
#     for i in range(1, n):
#         rgb = rgb + layers[i][:, :n_channels] * layers[i][:, n_channels:, :, :] #* alpha
#         # alpha = (1-layers[i][:, n_channels:, :, :]) * alpha
#     if return_alpha:
#         return rgb, alpha
# #     rgb = rgb + alpha
#     return rgb


def quantize_mask(mask, threshold=0.5):
    mask[mask>=threshold] = 1
    mask[mask<threshold] = 0
    return mask

def hard_composite_mask_wo_bg(layers, shape=[8,3,64,64], learn_bg=False):
    n = len(layers)
    if learn_bg:
        n = n-1
    if n==0:
        return torch.zeros(shape).to(layers[0].device)
    n_channels = layers[0].shape[1]-1
    mask = quantize_mask(layers[0][:, n_channels:, :, :])
    alpha = (1 - mask)
    rgb = 1 * mask
    for i in range(1, n):
        mask = quantize_mask(layers[i][:, n_channels:, :, :])
        rgb = rgb + (i+1) * mask * alpha
        alpha = (1-mask) * alpha
#     rgb = rgb + alpha
    return rgb

def color_layers(layers, color=[0,0,1], blend_percent=0.5):
    changed_layers = []
    n_channels = layers[0].shape[1]-1
    color = torch.tensor(color).to(layers[0].device, dtype=torch.float)[None, :, None, None]
    blend_percent = 1.0 # 1 = rgb 0= only color
    for l in layers:
        mask = l[:, n_channels:]
        # mask = quantize_mask(mask)
        layer_c = l[:, :n_channels] * mask*blend_percent + l[:, n_channels:]*color*(1-blend_percent)
        changed_layers.append(layer_c)
    return changed_layers