
import os
from shutil import copytree, ignore_patterns, rmtree
import click
from PIL import Image

import sys
sys.path.append('..'); sys.path.append('.')

from custom_argparser import args, config
import torch
from tqdm import tqdm
from bbn_layer import BBN_Layer
from torch.utils.tensorboard import SummaryWriter
from utils import count_parameters, update_progress, save_tensors, save_indv_parts


torch.manual_seed(42)
device = 'cuda'
#---------------------------------------------------------------------------------------------------------
#---------------------------------VARIABLES---------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------
create_new= True
save_dir = os.path.join(config['logging_params']['save_dir'], config['logging_params']['name']) + '/{}/{}'
data_path = config['model_params']['data_path']
epochs = config['model_params']['epochs']
img_size = config['model_params']['img_size']
num_layers = config['model_params']['num_layers']
batch_size = config['model_params']['batch_size']
dataset = config['model_params']['dataset']
add_checker_bg = config['logging_params']['add_checker_bg']
#---------------------------------------------------------------------------------------------------------
#---------------------------------DATA---------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision import transforms

transform = transforms.Compose([transforms.ToTensor(),
                                ])
dataset = datasets.ImageFolder(data_path, transform=transform)
dataloader_train =  DataLoader(dataset,
                                     batch_size= batch_size,
                                     shuffle = True,
                                     drop_last=True)
if add_checker_bg:
    filter_size = config['model_params']['filter_size']
    checker_bg = Image.open('./media/checker_bg.png').convert('RGB').resize([filter_size, filter_size], Image.ANTIALIAS)
    checker_bg = transform(checker_bg).unsqueeze(0)

#---------------------------------------------------------------------------------------------------------
#---------------------------------MODEL---------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------
class Single_BBN(nn.Module):

    def __init__(self, num_parts, num_channels, filter_size, num_updates=3, train=True, **kwargs):
        super().__init__()
        self.layer1 = BBN_Layer(num_parts, num_channels, filter_size, num_updates, train, **kwargs)

    def get_loss(self):
        loss = self.layer1.loss
        return loss

    def forward(self, img, num_layers, record_local_corr=True, eval=False):
        output = self.layer1(img, num_layers, record_local_corr, eval)
        return output

model_instance = Single_BBN(**config['model_params'])
model_params = list(model_instance.parameters())

lr = 1
optim = torch.optim.Adadelta(lr=lr, params=model_params)
count_parameters(model_instance)
model_instance.to(device)
#---------------------------------------------------------------------------------------------------------
#---------------------------------LOADING---------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------
init_epoch = 0
if os.path.exists(save_dir.format(args.version, '')):
    if click.confirm('Folder exists do you want to override?', default=True):
        rmtree(save_dir.format(args.version, ''))
    else:
        checkpoint = torch.load(save_dir.format(args.version, 'checkpoints/')+'checkpoint_generic.pt')
        model_instance.load_state_dict(checkpoint['model_params'])
        # optim.load_state_dict(checkpoint['optimizer_state_dict'])
        init_epoch = checkpoint['init_epoch']
        create_new = False
if create_new:
    os.makedirs(save_dir.format(args.version, ''))
    os.makedirs(save_dir.format(args.version, 'checkpoints'))
    os.makedirs(save_dir.format(args.version, 'images'))
    os.makedirs(save_dir.format(args.version, 'parts_clones'))   # used for creating the graphs
    copytree('./', os.path.join(save_dir.format(args.version, ''), 'code'),
         ignore=ignore_patterns('*.pyc', 'tmp*', 'logs*', 'data*', 'experiment_scripts*', 'notebook*'))

writer = SummaryWriter(log_dir=save_dir.format(args.version, ''))
#---------------------------------------------------------------------------------------------------------
#---------------------------------Training---------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------
phrase = 'generic'
if init_epoch==0:
    part_sig = torch.sin(model_instance.layer1.part_train * 15) * 0.5 + 0.5
    save_indv_parts(save_dir.format(args.version, 'parts_clones'), part_sig, 0)
for i_epoch in tqdm(range(init_epoch, epochs)):
    loss_print = 0
    max_iter_epoch  = len(dataloader_train)
    progress = tqdm(enumerate(dataloader_train), total=max_iter_epoch)
    for i_batch, sample_batched in progress:
        [img, label] = [i.to(device) for i in sample_batched]

        final_comp = model_instance(img, num_layers)
        loss = model_instance.get_loss()

        loss_print += loss.mean().detach().cpu().numpy()
        log = {'loss': loss_print/(i_batch+1), }
        update_progress(progress, log)
        writer.add_scalar('Loss/train', loss_print/(i_batch+1))

        if i_batch % (max_iter_epoch//10) == 0:
            filename = save_dir.format(args.version, 'images/input_recons_{}_{}.png').format(i_epoch, i_batch)
            save_tensors(filename, [img, final_comp, ], img.shape[0])
            # filename = save_dir.format(args.version, 'images/reconstruction_{}.png').format(i_batch)
            # save_tensors(filename, [final_comp, ], final_comp.shape[0])

            #### START SAVE PARTS
            filename = save_dir.format(args.version, 'images/parts_{}_{}_white.png.png').format(i_epoch, i_batch)
            part_sig = torch.sin(model_instance.layer1.part_train * 15) * 0.5 + 0.5
            part_sig = part_sig.detach().cpu()
            n_channels = part_sig.shape[1] - 1
            part_sig_ = part_sig[:, :n_channels] * part_sig[:, n_channels:] + checker_bg*(1-part_sig[:, n_channels:])
            save_tensors(filename, [part_sig_, ], )
            filename = save_dir.format(args.version, 'images/parts_{}_{}_black.png.png').format(i_epoch, i_batch)
            part_sig_ = part_sig[:, :n_channels] * part_sig[:, n_channels:] + (1-checker_bg)*(1-part_sig[:, n_channels:])
            save_tensors(filename, [part_sig_, ], )
            #### END SAVE PARTS
            if i_epoch % 10 == 0:
                phrase = str(i_epoch)
            else:
                phrase = 'generic'
        optim.zero_grad()
        loss.mean().backward()
        optim.step()

    torch.save({
        'model_params': model_instance.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        'init_epoch': i_epoch+1,
        'clone_graph': model_instance.layer1.clone_graph,
            }, save_dir.format(args.version, 'checkpoints/')+'checkpoint_{}.pt'.format(phrase))
    print(model_instance.layer1.part_norm_corr / model_instance.layer1.part_usage_count, model_instance.layer1.part_norm_corr, model_instance.layer1.part_usage_count)
    if i_epoch % 1 == 0:
        if model_instance.layer1.clone_part(i_epoch+1):
            model_params = list(model_instance.parameters())
            optim = torch.optim.Adadelta(lr=lr, params=model_params)
            part_sig = torch.sin(model_instance.layer1.part_train * 15) * 0.5 + 0.5
            part_sig = part_sig.detach().cpu()
            part_sig_ = part_sig[:, :n_channels] * part_sig[:, n_channels:] + (1-checker_bg)*(1-part_sig[:, n_channels:])
            save_indv_parts(save_dir.format(args.version, 'parts_clones'), part_sig_, i_epoch+1)


writer.close()
