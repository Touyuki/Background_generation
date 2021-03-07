class Opion():

    def __init__(self):
        self.dataroot = r'I:\irregular holes\paris_eval_gt'  # image dataroot
        self.maskroot = r'I:\irregular holes\testing_mask_dataset'  # mask dataroot
        self.batchSize = 1  # Need to be set to 1
        self.fineSize = 256  # image size
        self.input_nc = 3  # input channel size for first stage
        self.input_nc_g = 6  # input channel size for second stage
        self.output_nc = 3  # output channel size
        self.ngf = 64  # inner channel
        self.ndf = 64  # inner channel
        self.which_model_netD = 'basic'  # patch discriminator

        self.which_model_netF = 'feature'  # feature patch discriminator
        self.which_model_netG = 'unet_csa'  # seconde stage network
        self.which_model_netP = 'unet_256'  # first stage network
        self.triple_weight = 1
        self.name = 'CSA_inpainting'
        self.n_layers_D = '3'  # network depth
        self.gpu_ids = [0]
        self.model = 'csa_net'
        self.checkpoints_dir = r'.\checkpoints'  #
        self.norm = 'instance'
        self.fixed_mask = 1
        self.use_dropout = False
        self.init_type = 'normal'
        self.mask_type = 'center'
        self.lambda_A = 100
        self.threshold = 5 / 16.0
        self.stride = 1
        self.shift_sz = 1  # size of feature patch
        self.mask_thred = 1
        self.bottleneck = 512
        self.gp_lambda = 10.0
        self.ncritic = 5
        self.constrain = 'MSE'
        self.strength = 1
        self.init_gain = 0.02
        self.cosis = 1
        self.gan_type = 'lsgan'
        self.gan_weight = 0.2
        self.overlap = 4
        self.skip = 0
        self.display_freq = 1000
        self.print_freq = 50
        self.save_latest_freq = 5000
        self.save_epoch_freq = 2
        self.continue_train = False
        self.epoch_count = 1
        self.phase = 'train'
        self.which_epoch = ''
        self.niter = 20
        self.niter_decay = 100
        self.beta1 = 0.5
        self.lr = 0.0002
        self.lr_policy = 'lambda'
        self.lr_decay_iters = 50
        self.isTrain = True

import time
from util.data_load import Data_load
from models.models import create_model
import torch
import os
import torchvision
from torch.utils import data
import torchvision.transforms as transforms
opt = Opion()
transform_mask = transforms.Compose(
    [transforms.Resize((opt.fineSize,opt.fineSize)),
     transforms.ToTensor(),
    ])
transform = transforms.Compose(
    [
     transforms.Resize((opt.fineSize,opt.fineSize)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])

dataset_test = Data_load(opt.dataroot, opt.maskroot, transform, transform_mask)
iterator_test = (data.DataLoader(dataset_test, batch_size=opt.batchSize,shuffle=True))
print(len(dataset_test))
model = create_model(opt)
total_steps = 0
load_epoch=30
model.load(load_epoch)

save_dir = './measure/true'
if os.path.exists(save_dir) is False:
    os.makedirs(save_dir)

epoch = 1
i = 0
for image, mask in (iterator_test):
    iter_start_time = time.time()
    image = image.cuda()
    mask = mask.cuda()
    mask = mask[0][0]
    mask = torch.unsqueeze(mask, 0)
    mask = torch.unsqueeze(mask, 1)
    mask = mask.byte()

    model.set_input(image, mask)
    model.set_gt_latent()
    model.test()
    real_A, real_B, fake_B = model.get_current_visuals()
    pic = (torch.cat([real_A, real_B, fake_B], dim=0) + 1) / 2.0
    torchvision.utils.save_image(pic, '%s/Epoch_(%d)_(%dof%d).jpg' % (
        save_dir, epoch, total_steps + 1, len(dataset_test)), nrow=1)



