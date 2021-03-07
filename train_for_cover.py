class Opion():

    def __init__(self):
        self.dataroot = "input/0129f/new/"  # image dataroot
        self.maskroot = "input/0129f/mask/"  # mask dataroot
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
        self.checkpoints_dir ="weight/"  #
        self.norm = 'instance'
        self.fixed_mask = 1
        self.use_dropout = False
        self.init_type = 'normal'
        self.mask_type = 'random'
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
        self.display_freq = 1
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
print(opt.dataroot)
print(opt.maskroot)
save_dir="output/10/"
transform_mask = transforms.Compose(  ##将mask标准化
    [transforms.Resize((opt.fineSize,opt.fineSize)),
     transforms.ToTensor(),
    ])
transform = transforms.Compose(  ##将图片标准化
    [#transforms.RandomHorizontalFlip(),
     transforms.Resize((opt.fineSize,opt.fineSize)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])
dataset_train = Data_load(opt.dataroot, opt.maskroot, transform, transform_mask)
iterator_train = (data.DataLoader(dataset_train, batch_size=opt.batchSize,shuffle=True))
print(len(dataset_train))
model = create_model(opt)
total_steps = 0
iter_start_time = time.time()
count=0
for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):

    epoch_start_time = time.time()
    epoch_iter = 0
    for image,mask in (iterator_train):
        print(1)
        print("img.type",image.type)
        image = image.cuda()
        print("image",image.size())
        mask = mask.cuda()
        print("mask",mask.size())
        mask = mask[0][0]
        print("mask2",mask.size())
        mask = torch.unsqueeze(mask, 0)
        print("mask3",mask.size())
        mask = torch.unsqueeze(mask, 1)
        print("mask4",mask.size())
        mask = mask.byte()
        # break

        total_steps += opt.batchSize
        epoch_iter += opt.batchSize
        model.set_input(image, mask)  # it not only sets the input data with mask, but also sets the latent mask.

        # input=model.set_input(image, mask)
        # pic_t = (torch.cat([input], dim=0) + 1) / 2.0
        # torchvision.utils.save_image(pic_t,str(count)+"input_test.jpg")
        # count=count+1
        # if count>20 :
        #     exit(0)


        model.set_gt_latent()
        model.optimize_parameters()

        if total_steps % opt.display_freq == 0:
            real_A, real_B, fake_B = model.get_current_visuals()
            # real_A=input, real_B=ground truth fake_b=output
            pic = (torch.cat([real_A, real_B, fake_B], dim=0) + 1) / 2.0
            print("pic.size", pic.size())
            torchvision.utils.save_image(pic, '%s/Epoch_(%d)_(%dof%d).jpg' % (
                save_dir, epoch, total_steps + 1, len(dataset_train)), nrow=2)
        if total_steps % 2000 == 0:
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            print(errors)

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    model.update_learning_rate()