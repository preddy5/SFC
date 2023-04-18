
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from BBN.helpers import hard_composite, hard_composite_mask_wo_bg
import torchgeometry as tgm


class BBN_Layer(nn.Module):

    def __init__(self, num_parts, num_channels, filter_size, num_updates=3, train=True, rotation_angles=[0, ], scale=False,
                 max_clone_parts=32, num_colors=0, padding=0, binarize=False, min_clone_part_count=5000,
                 num_background=0, bg_size=0, search_redo=False, num_deformations=0, **kwargs):
        super().__init__()
        self.layers_cache_indv = []
        self.num_parts = num_parts
        part_norm_corr = torch.zeros([num_parts])
        part_usage_count = torch.zeros([num_parts])

        print([num_parts, num_channels, filter_size, filter_size])
        print('Number of updates: ', num_updates, ' Padding: ', padding, ' Search Redo: ', search_redo)
        self.clone_graph = {}
        self.clone_graph['init_num_nodes']= num_parts
        self.clone_graph['edges'] = []
        self.clone_graph['clone_epochs'] = []
        self.clone_graph['epoch_part_count'] = []

        part_train = self.init_parts([num_parts, num_channels, filter_size, filter_size])
        self.part_train = nn.Parameter(part_train)
        # self.part_train.data.fill_(0.0)
        self.learn_colors = False
        self.learn_bg = False
        self.learn_deformations = False

        if num_colors>0:
            # assert num_channels==1 or num_channels==2
            part_colors = torch.Tensor(num_colors, 3)
            self.part_colors = nn.Parameter(part_colors)
            # self.part_colors.data.fill_(0.0)

            # self.part_colors = [[1,0.25,0.25], [1,0.25,1], [1,1,0.25], [1,1,1],
            #                     [0.25,0.25,0.25], [0.25,0.25,1], [0.25,1,0.25], [0.25,1,1]]
            # self.part_colors = torch.Tensor(self.part_colors)
            self.learn_colors = True

        if num_deformations>0:
            deformations = torch.cat([torch.eye(2, 2), torch.zeros(2, 1)], dim=1)[None, ...]
            deformations = deformations.repeat(num_deformations, 1, 1)
            self.deformations = nn.Parameter(deformations)
            self.learn_deformations = True

        if num_background>0:
            # bg = torch.Tensor(num_background, 3, bg_size, bg_size)
            bg = self.init_parts([num_background, 3, bg_size, bg_size])
            # bg = torch.rand_like(bg)*0.05
            self.bg = nn.Parameter(bg)
            # self.bg.data.fill_(0.0)
            self.learn_bg = True

        self.register_buffer('part_norm_corr', part_norm_corr)
        self.register_buffer('part_usage_count', part_usage_count)

        # torch.nn.init.xavier_normal_(self.part_train)
        # = torch.zeros([num_parts, num_channels, filter_size, filter_size])
        # .bias.data.fill_(0.01)
        self.num_updates = num_updates
        self.part_size = filter_size
        self.train = train
        self.loss = 0
        self.rotate = len(rotation_angles)>1
        self.rotation_angles = rotation_angles
        self.scale = scale
        self.max_clone_parts = max_clone_parts
        self.padding = padding
        self.binarize = binarize
        self.min_clone_part_count = min_clone_part_count
        self.search_redo = search_redo
        self.omega = 30

    def init_parts(self, shape):
        num_parts, num_channels, _, filter_size = shape
        def gaussian_fn(M, std):
            n = torch.arange(0, M, dtype=torch.float) - (M - 1.0) / 2.0
            sig2 = 2 * std * std
            w = torch.exp(-n ** 2 / sig2)
            return w

        def gkern(kernlen=256, std=128):
            """Returns a 2D Gaussian kernel array."""
            gkern1d = gaussian_fn(kernlen, std=std)
            gkern2d = gkern1d[:, None] * gkern1d[None, :]
            return gkern2d
        part_train = gkern(filter_size, std=int(filter_size*2.5))
        part_train = part_train.repeat(num_parts, num_channels, 1, 1)
        part_train = part_train-1#*np.pi/60
        part_train[:, :num_channels-1] = part_train[:, :num_channels-1] * 0
        # part_train = part_train*0
        return part_train

    def clone_part(self, ):
        with torch.no_grad():
            cloned_part = False
            edges = []
            mean_part_corr = self.part_norm_corr / self.part_usage_count
            min_clone_part_count = 0.25*self.part_usage_count.mean()
            mean_part_corr, indices = torch.sort(mean_part_corr)
            num_parts = self.part_train.shape[0]
            max_clones = False
            clone_counter = self.max_clone_parts - num_parts

            if num_parts>=self.max_clone_parts:
                print('Maximum clones reached: ', self.max_clone_parts)
                max_clones = True
            parts_cache = []
            part_usage_count_cache = []
            zero_tensor = torch.tensor([0], dtype=torch.float32).to(self.part_norm_corr.device)
            for i in range(num_parts):
                part_idx = indices[i]
                current_part_corr = mean_part_corr[i]
                current_part_count = self.part_usage_count[part_idx]
                if current_part_count >min_clone_part_count:
                    if not max_clones:
                        if current_part_corr<0.99 and clone_counter>0:
                            clone_counter -= 1
                            parts_cache.append(self.part_train[part_idx])
                            part_usage_count_cache.append(zero_tensor)
                            print('CLONED PART: ', part_idx)
                            edges.append([part_idx.item(), len(parts_cache)-1])
                            cloned_part = True
                    parts_cache.append(self.part_train[part_idx])
                    edges.append([part_idx.item(), len(parts_cache)-1])
                    part_usage_count_cache.append(current_part_count)
                else:
                    print('REMOVED PART: ', part_idx)

            self.num_parts = len(parts_cache)
            self.part_train = nn.Parameter(torch.stack(parts_cache, dim=0))
            self.clone_graph['epoch_part_count'].append(self.part_train.shape[0])
            self.clone_graph['edges'].append(edges)

            self.part_norm_corr = torch.tensor([0] * self.num_parts, dtype=torch.float32).to(self.part_norm_corr.device)
            if cloned_part:
                self.part_usage_count = torch.tensor([0]*self.num_parts, dtype=torch.float32).to(self.part_norm_corr.device)
            else:
                self.part_usage_count = torch.tensor(part_usage_count_cache).to(self.part_norm_corr.device)
        return True

    def ncc2d_bg(self, image, parts, foreground_alpha, alpha_A, background, padding):
        with torch.no_grad():
            n_channels = parts.shape[1] - 1
            global_alpha = 1-foreground_alpha

            part_rgb = parts[:, :n_channels]
            part_alpha = parts[:, n_channels:]
            part_alpha_repeat = part_alpha.repeat(1, n_channels, 1, 1)

            part_filter = part_rgb * (part_alpha)

            imgXglobal_alpha = image * global_alpha
            corr = F.conv2d(imgXglobal_alpha, part_filter, padding=padding)
            corr_bg_temp = imgXglobal_alpha * background
            corr_bg_temp_sum = corr_bg_temp.sum(dim=1, keepdims=True).sum(dim=2, keepdims=True).sum(dim=3,
                                                                                                    keepdims=True)
            corr_bg = corr_bg_temp_sum - F.conv2d(corr_bg_temp, part_alpha_repeat, padding=padding)

            if type(alpha_A) == type(None):
                return corr
            I_alpha_a = image * alpha_A
            I_alpha_a = I_alpha_a.sum(dim=1, keepdims=True).sum(dim=2, keepdims=True).sum(dim=3, keepdims=True)
            I_norm = (image ** 2).sum(dim=1, keepdims=True).sum(dim=2, keepdims=True).sum(dim=3, keepdims=True)

            alpha_a_sigma = (alpha_A ** 2).sum(dim=1, keepdims=True).sum(dim=2, keepdims=True).sum(dim=3, keepdims=True)
            denom = F.conv2d(global_alpha ** 2, part_filter ** 2, padding=padding)  # + 1e-8
            ab_2 = F.conv2d(2 * alpha_A * global_alpha, part_filter, padding=padding)

            global_alphaXbackground = global_alpha * background
            bg_sq_ = (global_alphaXbackground) ** 2
            bg_sq_sum = bg_sq_.sum(dim=1, keepdims=True).sum(dim=2, keepdims=True).sum(dim=3, keepdims=True)
            bg_sq__ = F.conv2d(bg_sq_, part_alpha_repeat ** 2, padding=padding)
            bg_sq = bg_sq_sum + bg_sq__ - F.conv2d(2 * bg_sq_, part_alpha_repeat, padding=padding)

            ac_2_temp = 2 * alpha_A * global_alphaXbackground
            ac_2_temp_sum = ac_2_temp.sum(dim=1, keepdims=True).sum(dim=2, keepdims=True).sum(dim=3, keepdims=True)
            ac_2 = ac_2_temp_sum - F.conv2d(ac_2_temp, part_alpha_repeat, padding=padding)

            bc_2_ = 2 * global_alpha * global_alphaXbackground
            bc_2__ = F.conv2d(bc_2_, part_filter, padding=padding)
            bc_2 = bc_2__ - F.conv2d(bc_2_, part_filter * part_alpha, padding=padding)

            #     denom = torch.sqrt(denom1 * denom2) + 1e-8
            denom_div = 1/torch.sqrt(I_norm * (denom + alpha_a_sigma + bg_sq + ab_2 + ac_2 + bc_2))
            denom_div = torch.where(torch.isnan(denom_div), torch.zeros_like(denom_div), denom_div)
            norm_corr = (corr + I_alpha_a + corr_bg) *denom_div
        return norm_corr

    def cache_indv_layers(self, layers_cache_indv, inv_corr, idx=None):
        bs = inv_corr.shape[0]
        if len(layers_cache_indv) == 0:
            for i in range(bs):
                layers_cache_indv.append([])
        if type(idx)!=type(None):
            for i in range(bs):
                layers_cache_indv[i].insert(idx, inv_corr[i:i + 1])
        else:
            for i in range(bs):
                layers_cache_indv[i].append(inv_corr[i:i + 1])
        return layers_cache_indv

    def cache_indv_layers_io(self, layers_cache_indv, idx, operation, insert_layers=[], selection=None):
        '''
        idx - which layer to change
        selection - where to change
        '''
        indv_layer = []
        bs = len(layers_cache_indv)
        if type(selection) == type(None):
            selection = range(bs)
        num_changes = len(selection)
        if type(idx) != list and type(idx) != np.ndarray:
            idx = [idx, ] * num_changes

        if operation == 'pop':
            for i in range(num_changes):
                indv_layer.append(layers_cache_indv[selection[i]].pop(idx[i]))
        if operation == 'insert':
            for i in range(num_changes):
                layers_cache_indv[selection[i]].insert(idx[i], insert_layers[i])
        if operation == 'insert_tensor':
            for i in range(num_changes):
                layers_cache_indv[selection[i]].insert(idx[i], insert_layers[i:i + 1])
        #             import pdb; pdb.set_trace()
        return layers_cache_indv, indv_layer

    def create_layer_cache(self, layers_cache_indv):
        layer_cache = []
        bs = len(layers_cache_indv)
        num_layers = len(layers_cache_indv[0])
        for i in range(num_layers):
            layer = torch.cat([layers_cache_indv[j][i] for j in range(bs)], dim=0)
            layer_cache.append(layer)
        return layer_cache

    def quant_mask(self, corr, return_val=False):
        corr_unroll = corr.view(corr.size(0), -1)  # tensor of shape (batch_size, height * width)
        val, idx = corr_unroll.max(dim=1, keepdim=True)
        mask = torch.zeros_like(corr_unroll)
        #     mask[np.arange(mask.shape[0]), idx] = 1
        for i in range(mask.shape[0]):
            mask[i, idx[i]] = 1
        mask_reshape = mask.reshape(corr.shape)
        if return_val:
            return mask_reshape, val
        return mask_reshape

    def rotate_filters(self, filters, thetas=[0, 90, 180, 270], scale=1.0):
        all_filters = []
        bs, c, w, h = filters.shape
        for angle in thetas:
            center = torch.zeros(1, 2)
            sc = torch.ones(1)#*scale
            rot_matrix = tgm.get_rotation_matrix2d(center, angle* torch.ones(1), sc)
            rot_matrix = rot_matrix.to(filters.device).repeat(bs, 1, 1)

            grid = torch.nn.functional.affine_grid(rot_matrix, filters.shape)
            grid_scale = grid/scale

            rotate_filters = torch.nn.functional.grid_sample(filters, grid_scale, mode='bilinear', padding_mode='zeros')
            all_filters.append(rotate_filters)
        all_filters = torch.cat(all_filters, dim=0)
        return all_filters

    def deform_filters(self, filters, params):
        n = params.shape[0]
        bs = filters.shape[0]
        filters = filters.repeat(n, 1, 1, 1)
        params = params.repeat_interleave(bs, dim=0)
        grid = torch.nn.functional.affine_grid(params, filters.shape)
        filters = torch.nn.functional.grid_sample(filters, grid, mode='bilinear', padding_mode='zeros')
        return filters

    def color_filters(self, filters):
        n, c, w, h = filters.shape
        all_filters = []
        colors = torch.sin(self.part_colors*self.omega)*0.5+0.5
        # colors = torch.sin(self.part_colors*self.omega)*0.5+0.5
        for i in range(colors.shape[0]):
            if c==1:
                filters_alpha = filters[:, :, :, :]
            else:
                filters_alpha = filters[:, -1:, :, :]
            filters_colored = filters[:, :-1, :, :]*colors[i:i+1, :, None, None]
            filters_colored = torch.cat([filters_colored, filters_alpha], dim=1)
            all_filters.append(filters_colored)
        all_filters = torch.cat(all_filters, dim=0)
        return all_filters

    def search_norm_bg(self, img, bg_sig, foreground=None):
        with torch.no_grad():
            if type(foreground)!=type(None):
                rgb, alpha = foreground#[:,:-1], foreground[:,-1:]
                img = (img - rgb)
                div_alpha = 1/(1-alpha)
                div_alpha = torch.where(torch.isnan(div_alpha), torch.zeros_like(div_alpha), div_alpha)
                img = img*div_alpha
            bs = img.shape[0]
            num_bg = self.bg.shape[0]
            img_reshape = img.reshape([bs, -1])
            bg_sig_ = bg_sig[:, -1:]*bg_sig[:, :-1]
            bg_reshape = bg_sig_.reshape([num_bg, -1]).transpose(1,0)

            bg_denom = (bg_reshape**2).sum(dim=0, keepdims=True)
            img_denom = (img_reshape**2).sum(dim=1, keepdims=True)
            denom_div = 1/torch.sqrt((img_denom*bg_denom))
            denom_div = torch.where(torch.isnan(denom_div), torch.zeros_like(denom_div), denom_div)

            corr = (img_reshape@bg_reshape)*denom_div
            return corr

    def normal_norm_corr_local(self, I, I_hat, mask):
        I = I * mask
        I_hat = I_hat * mask
        I_norm = (I ** 2).sum(dim=1, keepdims=True).sum(dim=2, keepdims=True).sum(dim=3, keepdims=True)
        I_hat_norm = (I_hat ** 2).sum(dim=1, keepdims=True).sum(dim=2, keepdims=True).sum(dim=3, keepdims=True)
        corr = (I * I_hat).sum(dim=1, keepdims=True).sum(dim=2, keepdims=True).sum(dim=3, keepdims=True)
        norm_corr = corr / torch.sqrt(I_norm * I_hat_norm)
        # safe_tensor = torch.where(torch.isnan(norm_corr),
        #                           torch.zeros_like(norm_corr), norm_corr)
        return norm_corr

    def record_part_corr(self, img, layer_cache, part_id_cache):
        with torch.no_grad():
            global_alpha = torch.ones_like(img)
            n_channels = global_alpha.shape[1]
            num_layers = len(layer_cache)
            if self.learn_bg:
                num_layers -= 1
            for i in range(num_layers):
                current_layer = layer_cache[i]
                part_id = part_id_cache[i]
                batch_corr_masked = self.normal_norm_corr_local(img , current_layer[:, :n_channels],
                                                                current_layer[:, n_channels:]* global_alpha)

                inf_mask = torch.where(torch.isinf(batch_corr_masked)+torch.isnan(batch_corr_masked),
                            torch.zeros_like(batch_corr_masked), torch.ones_like(batch_corr_masked))
                batch_part_corr_masked = batch_corr_masked[:, :, 0, 0] * part_id
                batch_part_corr_masked = torch.where(torch.isinf(batch_part_corr_masked)+torch.isnan(batch_part_corr_masked),
                            torch.zeros_like(batch_part_corr_masked), batch_part_corr_masked)

                part_corr_masked = batch_part_corr_masked.sum(dim=0)
                self.part_norm_corr += part_corr_masked.detach().clone()
                part_corr_masked_np = part_corr_masked.detach().clone().cpu().numpy()
                if np.all(np.isnan(part_corr_masked_np)):
                    import pdb; pdb.set_trace()
                self.part_usage_count += (part_id* inf_mask[:, :, 0, 0]).sum(dim=0).detach().clone() # removing inf from count
                global_alpha = global_alpha * (1 - current_layer[:, n_channels:])

    def record_part_usage(self, inv_corr, mask_reshape, part_id_cache_indv, idx=None):
        with torch.no_grad():
            visibility = inv_corr[:, -1:].sum(dim=[2, 3])
            visibility = torch.where(visibility > 0, torch.ones_like(visibility),
                                     torch.zeros_like(visibility))  # 5 is a hyperparameter can be 0
            part_id = mask_reshape.sum(dim=[2, 3]) * visibility
            part_id_clone = part_id[:, :self.num_parts].clone()
            for rotation_idx in range(1, part_id.shape[1] // self.num_parts):
                rotation_idx = rotation_idx * self.num_parts
                part_id_clone += part_id[:, rotation_idx:rotation_idx + self.num_parts].clone()
            if type(idx)==type(None):
                part_id_cache_indv = self.cache_indv_layers(part_id_cache_indv, part_id_clone)
            else:
                part_id_cache_indv = self.cache_indv_layers(part_id_cache_indv, part_id_clone, idx=idx)
        return part_id_cache_indv

    def brute_force_search(self, img, part_sig, pad, num_layers, layers_cache,
                           layers_cache_indv, part_id_cache_indv, record_local_corr, is_redo=False):
        for j in range(num_layers):
            bg_selected = []
            if j==0:
                foreground = None
                if is_redo:
                    foreground = hard_composite(layers_cache, img.shape, True)
                if self.learn_bg:
                    num_bg, c, bg_size, bg_size = self.bg.shape
                    bg_sig = torch.sin(self.bg * self.omega) * 0.5 + 0.5
                    bg_sig = torch.cat([bg_sig, torch.ones(num_bg, 1, bg_size, bg_size).to(self.bg.device)], dim=1)
                    corr_bg = self.search_norm_bg(img, bg_sig, foreground=foreground)
                    mask_reshape = self.quant_mask(corr_bg)
                    bg_selected = mask_reshape@bg_sig.reshape([num_bg, -1])
                    bg_selected = [bg_selected.reshape([-1, c+1, bg_size, bg_size]), ]
            if is_redo:
                layers_cache_indv, cache_layer = self.cache_indv_layers_io(layers_cache_indv, j, 'pop')
                part_id_cache_indv, cache_layer = self.cache_indv_layers_io(part_id_cache_indv, j, 'pop')
                layers_cache = self.create_layer_cache(layers_cache_indv)

            num_layers_ = len(layers_cache)
            layers_cache = layers_cache + bg_selected
            for k in range(num_layers_ + 1):
                foreground, foreground_alpha = hard_composite(layers_cache[:k], img.shape, True)
                background = hard_composite(layers_cache[k:], shape=img.shape)

                corr_b2f = self.ncc2d_bg(img, part_sig, foreground_alpha,
                                         foreground,
                                         background=background,
                                         padding=pad)
                mask_reshape_b2f, current_val_b2f = self.quant_mask(corr_b2f, True)
                if k == 0:
                    mask_reshape = mask_reshape_b2f
                    current_val = current_val_b2f
                    layer_idx = np.array([0, ] * len(layers_cache_indv))
                else:
                    b2f_greater = (current_val_b2f > current_val)
                    b2f_greater = (b2f_greater[:, :, None, None]).type(torch.float)
                    b2f_greater_np = b2f_greater.cpu().numpy()

                    layer_idx_b2f = np.array([k, ] * len(layers_cache_indv))

                    mask_reshape = mask_reshape_b2f * b2f_greater + (1 - b2f_greater) * mask_reshape
                    current_val = current_val_b2f * b2f_greater[:, :, 0, 0] + (
                            1 - b2f_greater[:, :, 0, 0]) * current_val
                    layer_idx = layer_idx_b2f * b2f_greater_np[:, 0, 0, 0] + (
                            1 - b2f_greater_np[:, 0, 0, 0]) * layer_idx

            # if type(previous_val) == type(None):
            #     previous_val = current_val
            # else:
            #     threshold = 0.01 # robustness to noise.
            #     changes = (current_val - previous_val) > threshold
            #     changes = (changes[:, :, None, None]).type(torch.float)
            #     mask_reshape = mask_reshape * changes
            #     previous_val = current_val * changes[:, :, 0, 0] + (1 - changes[:, :, 0, 0]) * previous_val

            inv_corr = F.conv_transpose2d(mask_reshape, part_sig)
            if pad != 0:
                inv_corr = inv_corr[:, :, pad:-pad, pad:-pad]

            layer_idx = layer_idx.astype(int)
            layers_cache_indv, _ = self.cache_indv_layers_io(layers_cache_indv, layer_idx, 'insert_tensor', inv_corr)
            layers_cache = self.create_layer_cache(layers_cache_indv)

            if record_local_corr:  # logging each part local corr
                if not is_redo:
                    self.record_part_usage(inv_corr, mask_reshape, part_id_cache_indv)
                else:
                    self.record_part_usage(inv_corr, mask_reshape, part_id_cache_indv, j)
        return layers_cache_indv , layers_cache, bg_selected


    def forward(self, img, num_layers, record_local_corr, eval=False):

        n_channels = self.part_train.shape[1] - 1
        bs, c, h, w = img.shape
        layers_cache = []
        layers_cache_indv = []
        for _ in range(bs):
            layers_cache_indv.append([])
        part_id_cache_indv = []
        self.loss = 0
        pad = self.padding

        part_sig = torch.sin(self.part_train*self.omega)*0.5+0.5

        if self.learn_deformations:
            # deformations = torch.sin(self.deformations*self.omega)*2
            part_sig = self.deform_filters(part_sig, self.deformations)

        if self.rotate:
            rotation_padding = int(self.part_size * 0.207 *max(self.rotation_angles)/90) + 1
            part_sig = F.pad(part_sig, [rotation_padding,]*4, 'constant', 0)
            # rotation_angels = rotation_angels*360/6.28318531
            part_sig = self.rotate_filters(part_sig,  self.rotation_angles)

        if self.scale:
            scaled_part_sig = []
            for sc in np.arange(0.8,1.05,0.05):
                scaled_part_sig.append(self.rotate_filters(part_sig, [0,], scale=sc))
            part_sig = torch.cat(scaled_part_sig, dim=0)

        if self.learn_colors:
            part_sig = self.color_filters(part_sig)

        for t in range(self.num_updates):
            if t == 0:
                layers_cache_indv , layers_cache, bg_selected = self.brute_force_search(img, part_sig, pad, num_layers,
                                                                           layers_cache, layers_cache_indv,
                                                                           part_id_cache_indv, record_local_corr)
            else:
                layers_cache_indv , layers_cache, bg_selected = self.brute_force_search(img, part_sig, pad, num_layers,
                                                                           layers_cache, layers_cache_indv,
                                                                           part_id_cache_indv, record_local_corr,
                                                                           is_redo=True)
            layers_cache = self.create_layer_cache(layers_cache_indv) + bg_selected
            final_composite = hard_composite(layers_cache)  # , background[:,:,None,None])
            self.loss = F.mse_loss(img, final_composite[:, :3], reduction='none').mean(dim=[1, 2, 3])


        if record_local_corr:
            layers_cache = self.create_layer_cache(layers_cache_indv) + bg_selected # redudant but i'll let it be
            part_id_cache = self.create_layer_cache(part_id_cache_indv)
            self.record_part_corr(img, layers_cache, part_id_cache)
        if eval:
            return final_composite, layers_cache
        return final_composite
