#This code is released under the CC BY-SA 4.0 license.
## For unest+SGP
import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import numpy as np
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from .SGP_LSC import CLIP, mae_masking
from torch import nn
import os
import random
import monai.networks.nets as nets

class PTAhybridModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.
        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
        Returns:
            the modified parser.
        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        parser.set_defaults(netG='sf')
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_seg', type=float, default=0.5, help='weight for segmentation loss')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
        parser.add_argument('--n_downsampling', type=int, default=3, help='number of down')
        parser.add_argument('--depth', type=int, default=4, help='number of down')
        parser.add_argument('--edge', default=False,  action='store_true', help='apply attention on edge tokens only')
        parser.add_argument('--heads', type=int, default=12, help='number of down')
        parser.add_argument('--dropout', type=float, default=0.00, help='number of down')
        parser.add_argument('--fth', type=float, default=0.5, help='Foreground threshold')
        parser.add_argument('--vit_img_size', type=int, nargs='+', default=[224, 224], help='ViT image size')
        parser.add_argument('--window_size', type=int, default=2, help='number of down')
        parser.add_argument('--out_kernel', type=int, default=7, help='kernel size for output convolution layer')
        parser.add_argument('--patch_size', type=int, default=16, help='kernel size for output convolution layer')
        parser.add_argument('--feat_size', type=int, default=16, help='kernel size for output convolution layer')
        parser.add_argument('--vit_emb_size', type=int, default=768, help='kernel size for output convolution layer')
        parser.add_argument('--upsample', type=str, default='deconv', help='UNet upsampling mode')
        parser.add_argument('--vit-norm', type=str, default='layer', help='ViT norm')
        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B', 'seg_A', 'seg_B', 'att']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A', 'seg_A', 'mask_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B', 'seg_B', 'mask_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        if 'mask' in self.opt.netG  or 'att' in self.opt.netG:
            visual_names_A.append('att_A')
            visual_names_B.append('att_B')
        self.gt_shape_assist = True
        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']


        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, 
                                        depth=opt.depth, heads=opt.heads, img_size=opt.vit_img_size,  
                                        window_size=opt.window_size, edge=opt.edge, fth=opt.fth,
                                        out_kernel=self.opt.out_kernel,patch_size=opt.patch_size,
                                        feat_size=self.opt.feat_size, upsample=opt.upsample, vit_emb_size=opt.vit_emb_size,
                                        dropout_rate=opt.dropout, vit_norm=opt.vit_norm)
        self.netG_B = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, 
                                        depth=opt.depth, heads=opt.heads, img_size=opt.vit_img_size,  
                                        window_size=opt.window_size, edge=opt.edge, fth=opt.fth,
                                        out_kernel=self.opt.out_kernel, patch_size=opt.patch_size, 
                                        feat_size=self.opt.feat_size, upsample=opt.upsample, vit_emb_size=opt.vit_emb_size,
                                        dropout_rate=opt.dropout, vit_norm=opt.vit_norm)
        self.SGP =  CLIP(embed_dim = 512, image_resolution=120, vision_layers=6, vision_width=256, vision_patch_size=16).to(self.device)
        self.SGP.load_state_dict(torch.load('/media/NAS07/USER_PATH/zzx/weight/clip_1.pth'),True)
        self.LSC = nets.UNet(
        spatial_dims=2, 
        in_channels=1,
        out_channels=1, 
        channels=(16, 32, 64, 128, 256),  
        strides=(2, 2, 2, 2),  
        num_res_units=2 ).to(self.device)
        self.LSC.load_state_dict(torch.load('/media/NAS07/USER_PATH/zzx/weight/unet_weights_rotate.pth'),True)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            real, fake= 0.9, 0.1
            self.criterionGAN = networks.GANLoss(opt.gan_mode, target_real_label=real, target_fake_label=fake).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)


    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        #print("Input keys:", input.keys())
        self.real_A = input['A' if AtoB else 'B'].to(self.device).float()
        #print(self.real_A.shape)
        self.real_B = input['B' if AtoB else 'A'].to(self.device).float()
        self.real_A_pre =F.interpolate(self.real_A, size=(120, 120), mode='bilinear', align_corners=False, antialias=True)
        self.real_B_pre =F.interpolate(self.real_B, size=(120, 120), mode='bilinear', align_corners=False, antialias=True)
        self.mask_A = input['A_mask' if AtoB else 'B_mask'].float().to(self.device)
        self.mask_B = input['B_mask' if AtoB else 'A_mask'].float().to(self.device)
        self.down_mask_A = F.max_pool2d(self.mask_A , self.opt.patch_size)
        self.down_mask_B = F.max_pool2d(self.mask_B, self.opt.patch_size)
        with torch.no_grad():
            self.high_features = self.SGP.encode_image_high_field(self.real_B_pre).float()
            self.low_features = self.SGP.encode_image_high_field(self.real_A_pre).float()
        self.high_features /= self.high_features.norm(dim=-1, keepdim=True)
        self.low_features /= self.low_features.norm(dim=-1, keepdim=True)
        self.similarity = self.low_features.cpu().numpy() @ self.high_features.cpu().numpy().T
        self.cost_matrix = -self.similarity  
        self.row_ind, self.col_ind = linear_sum_assignment(self.cost_matrix)  # row_ind 是 a 的索引, col_ind 是对应 b 的索引
        self.real_B = self.real_B[self.col_ind]
        self.mask_B = self.mask_B[self.col_ind]
        self.down_mask_B = self.down_mask_B[self.col_ind]
        # print(self.real_B_rot.shape)
        self.real_B_rot = F.interpolate(self.real_B, size=(120, 120), mode='bilinear')
        self.real_B_rot = mae_masking(self.real_B_rot, num_patches=24, mask_ratio=0.2)
        self.real_B_rot = F.interpolate(self.real_B_rot, size=(128, 128), mode='bilinear')
        self.real_B_rot = self.LSC(self.real_B_rot)
        self.real_B_rot = F.interpolate(self.real_B_rot, size=(224, 224), mode='bilinear')

        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

        mask_A = mask_B = None
        if self.gt_shape_assist:
            mask_A, mask_B = self.down_mask_A, self.down_mask_B

        if self.opt.netG == 'unest_mask' or self.opt.netG == 'unest_att':
            self.fake_B, self.seg_A, self.att_A = self.netG_A(self.real_A, mask_A)  # G_A(A)
            self.fake_B_rot = F.interpolate(self.fake_B, size=(120, 120), mode='bilinear')
            self.fake_B_rot = mae_masking(self.fake_B_rot, num_patches=24, mask_ratio=0.2)
            self.fake_B_rot = F.interpolate(self.fake_B_rot, size=(128, 128), mode='bilinear')
            self.fake_B_rot = self.LSC(self.fake_B_rot)
            self.fake_B_rot = F.interpolate(self.fake_B_rot, size=(224, 224), mode='bilinear')
            self.rec_A, self.rec_segA, _ = self.netG_B(self.fake_B, mask_A)   # G_B(G_A(A))
            self.fake_A, self.seg_B, self.att_B = self.netG_B(self.real_B, mask_B)  # G_B(B)
            self.rec_B, self.rec_segB, _ = self.netG_A(self.fake_A, mask_B)   # G_A(G_B(B))
        else:
            self.fake_B, self.seg_A = self.netG_A(self.real_A, mask_A)  # G_A(A)
            self.fake_B_rot = F.interpolate(self.fake_B, size=(120, 120), mode='bilinear')
            self.fake_B_rot = mae_masking(self.fake_B_rot, num_patches=24, mask_ratio=0.2)
            self.fake_B_rot = F.interpolate(self.fake_B_rot, size=(128, 128), mode='bilinear')
            self.fake_B_rot = self.LSC(self.fake_B_rot)
            self.fake_B_rot = F.interpolate(self.fake_B_rot, size=(224, 224), mode='bilinear')
            self.rec_A, self.rec_segA = self.netG_B(self.fake_B, mask_A)   # G_B(G_A(A))
            self.fake_A, self.seg_B = self.netG_B(self.real_B, mask_B)  # G_B(B)
            self.rec_B, self.rec_segB = self.netG_A(self.fake_A, mask_B)   # G_A(G_B(B))
            self.att_A = self.att_B = None


    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator
        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator
        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A (with concatenation)"""
    # Concatenate fake_B and fake_B_rot along the channel dimension
        fake_B = self.fake_B_pool.query(self.fake_B)
        # Concatenate real_B and real_B_rot along the channel dimension)
        
        # Compute loss with concatenated data
        loss_D_A_1 = self.backward_D_basic(self.netD_A, self.real_B, fake_B)
        loss_D_A_2 = self.backward_D_basic(self.netD_A, self.real_B_rot, self.fake_B_rot)
        self.loss_D_A = loss_D_A_1 + loss_D_A_2
        self.loss_D_A.backward()

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)
        self.loss_D_B.backward()

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_seg = self.opt.lambda_seg
        mask_A = mask_B = None
        if self.gt_shape_assist:
            mask_A, mask_B = self.down_mask_A, self.down_mask_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B, mask_B)[0]
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A, mask_A)[0]
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0
        
        self.loss_att = self.loss_seg = self.loss_seg_A = self.loss_seg_B = 0
        if lambda_seg > 0:
            # target_A = F.interpolate(self.mask_A, size=(self.seg_A.shape[-2], self.seg_A.shape[-1]))
            # target_B = F.interpolate(self.mask_B, size=(self.seg_B.shape[-2], self.seg_B.shape[-1]))
            self.loss_seg_A = F.binary_cross_entropy(self.seg_A, self.down_mask_A)
            self.loss_seg_B = F.binary_cross_entropy(self.seg_B, self.down_mask_B)
            self.loss_seg = lambda_seg*(self.loss_seg_A + self.loss_seg_B)

            if self.att_A is not None:
                if self.att_A.size() != self.mask_A.size():
                    self.att_A = F.interpolate(self.att_A, size=(self.mask_A.size(-2), self.mask_A.size(-1)))
                    self.att_B = F.interpolate(self.att_B, size=(self.mask_B.size(-2), self.mask_B.size(-1)))
                self.loss_att_A = F.binary_cross_entropy(self.att_A, self.mask_A)
                self.loss_att_B = F.binary_cross_entropy(self.att_B, self.mask_B)
                self.loss_att = lambda_seg *(self.loss_att_A + self.loss_att_B)

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B \
            + self.loss_idt_A + self.loss_idt_B + self.loss_seg + self.loss_att
        self.loss_G.backward()


    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()  # calculate gradients for G_A and G_B
        self.optimizer_G.step()  # update G_A and G_B's weights