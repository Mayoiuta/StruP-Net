import numpy as np
import torch
import os
import itertools
from . import networks
import torch.nn.functional as F
import torch.nn as nn

from util.image_pool import ImagePool

from collections import OrderedDict

class Sobel_Module(nn.Module):
    def __init__(self, thre_min, thre_max, weight):
        super(Sobel_Module, self).__init__()

        self.thre_min = thre_min
        self.thre_max = thre_max
        self.weight = weight

        k0 = torch.FloatTensor(
            np.array(
            [[-1.,0.,1.],
            [-2.,0.,2.],
            [-1.,0.,1.]]
            ).reshape((1, 1, 3, 3)))
        
        self.kernel0 = nn.Parameter(data=k0, requires_grad=True)

        k1 = torch.FloatTensor(np.array(
            [[2.,1.,0.],
            [1.,0.,-1.],
            [0.,-1.,-2.]]
        ).reshape((1, 1, 3, 3)))

        self.kernel1 = nn.Parameter(data=k1, requires_grad=True)

        k2 = torch.FloatTensor(np.array(
            [[1.,2.,1.],
            [0.,0.,0.],
            [-1.,-2.,-1.]]
        ).reshape((1, 1, 3, 3)))

        self.kernel2 = nn.Parameter(data=k2, requires_grad=True)

        k3 = torch.FloatTensor(np.array(
            [[0,-1,-2],
            [1,0,-1],
            [2,1,0]]
        ).reshape((1, 1, 3, 3)))

        self.kernel3 = nn.Parameter(data=k3, requires_grad=True)
    
    
    def forward(self, x):
        e0 = torch.abs(F.conv2d(x,self.kernel0,padding=(1,1)))
        e1 = torch.abs(F.conv2d(x,self.kernel1,padding=(1,1)))
        e2 = torch.abs(F.conv2d(x,self.kernel2,padding=(1,1)))
        e3 = torch.abs(F.conv2d(x,self.kernel3,padding=(1,1)))
        grad1 = e0+e1+e2+e3

        grad1 = (grad1 > self.thre_min) * (grad1 < self.thre_max) * grad1

        return grad1 * self.weight

class LossNetwork(nn.Module):
    def __init__(self, weight, k):
        super(LossNetwork, self).__init__()

        inplace=True

        pad = int((k-1)/2)
        vgg_model = nn.Sequential(OrderedDict([
        ('0',nn.Conv2d(3, 64, kernel_size=(k, k), stride=(1, 1), padding=(pad, pad))),
        ('1',nn.ReLU(inplace)),
        ('2',nn.Conv2d(64, 64, kernel_size=(k, k), stride=(1, 1), padding=(pad, pad))),
        ('3',nn.ReLU(inplace)),
        ('4',nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)),
        ('5',nn.Conv2d(64, 128, kernel_size=(k, k), stride=(1, 1), padding=(pad, pad))),
        ('6',nn.ReLU(inplace)),
        ('7',nn.Conv2d(128, 128, kernel_size=(k, k), stride=(1, 1), padding=(pad, pad))),
        ('8',nn.ReLU(inplace)),
        ('9',nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)),
        ('10',nn.Conv2d(128, 256, kernel_size=(k, k), stride=(1, 1), padding=(pad, pad))),
        ('11',nn.ReLU(inplace)),
        ('12',nn.Conv2d(256, 256, kernel_size=(k, k), stride=(1, 1), padding=(pad, pad))),
        ('13',nn.ReLU(inplace)),
        ('14',nn.Conv2d(256, 256, kernel_size=(k, k), stride=(1, 1), padding=(pad, pad))),
        ('15',nn.ReLU(inplace)),
        ('16',nn.Conv2d(256, 256, kernel_size=(k, k), stride=(1, 1), padding=(pad, pad))),
        ('17',nn.ReLU(inplace)),
        ('18',nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)),
        ('19',nn.Conv2d(256, 512, kernel_size=(k, k), stride=(1, 1), padding=(pad, pad))),
        ('20',nn.ReLU(inplace)),
        ('21',nn.Conv2d(512, 512, kernel_size=(k, k), stride=(1, 1), padding=(pad, pad))),
        ('22',nn.ReLU(inplace)),
        ('23',nn.Conv2d(512, 512, kernel_size=(k, k), stride=(1, 1), padding=(pad, pad))),
        ('24',nn.ReLU(inplace)),
        ('25',nn.Conv2d(512, 512, kernel_size=(k, k), stride=(1, 1), padding=(pad, pad))),
        ('26',nn.ReLU(inplace)),
        ('27',nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)),
        ('28',nn.Conv2d(512, 512, kernel_size=(k, k), stride=(1, 1), padding=(pad, pad))),
        ('29',nn.ReLU(inplace)),
        ('30',nn.Conv2d(512, 512, kernel_size=(k, k), stride=(1, 1), padding=(pad, pad))),
        ('31',nn.ReLU(inplace)),
        ('32',nn.Conv2d(512, 512, kernel_size=(k, k), stride=(1, 1), padding=(pad, pad))),
        ('33',nn.ReLU(inplace)),
        ('34',nn.Conv2d(512, 512, kernel_size=(k, k), stride=(1, 1), padding=(pad, pad))),
        ('35',nn.ReLU(inplace)),
        ('36',nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)),])).cuda()
        vgg_model.eval()
 
        for param in vgg_model.parameters():
            param.requires_grad = False


        self.vgg_layers = vgg_model

        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '17': "relu3_4",
            '26': "relu4_4",
            '35': "relu5_4"
        }
       
        self.weight = weight        

        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)

    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return list(output.values())
 
    def forward(self, output, gt):
        loss = []
        output_features = self.output_features(output)
        gt_features = self.output_features(gt)
        for iter,(dehaze_feature, gt_feature,loss_weight) in enumerate(zip(output_features, gt_features,self.weight)):
            loss.append(F.mse_loss(dehaze_feature, gt_feature)*loss_weight)
        return sum(loss)

class DiceLoss(nn.Module):

    def __init__(self, sigmoid = True):
        super().__init__()
        self.smooth = 1
        self.sigmoid = sigmoid

    def forward(self, predict, target):
        b,c,h,w = predict.size()
        if c!= 1:
            full_loss = 0.0
            step = 0
            for i in range(c):
                p = predict[:,c:c+1,:,:]
                t = target[:,c:c+1,:,:]

                batch_size = p.size(0)

                if self.sigmoid:
                    p=torch.sigmoid(p)

                pre = p.view(batch_size, -1)
                tar = t.view(batch_size, -1)

                intersection = (pre * tar).sum(-1).sum()
                union = ((pre + tar).sum(-1)).sum()

                dice_coeff = 2 * (intersection + self.smooth) / (union + self.smooth)
                dice_loss = 1 - dice_coeff

                full_loss += torch.clamp(dice_loss, 0, 1)
                step += 1

            return full_loss * (1.0/step)

        else:
            predict = predict.squeeze(1)
            target = target.squeeze(1)
            assert predict.size() == target.size(), "The size of predict and target must be equal."
            batch_size = predict.size(0)

            if self.sigmoid:
                predict=torch.sigmoid(predict)

            pre = predict.view(batch_size, -1)
            tar = target.view(batch_size, -1)

            intersection = (pre * tar).sum(-1).sum()
            union = ((pre + tar).sum(-1)).sum()

            dice_coeff = 2 * (intersection + self.smooth) / (union + self.smooth)
            dice_loss = 1 - dice_coeff

            return torch.clamp(dice_loss, 0, 1)

class StruPNet(nn.Module):
    
    def name(self):
        return 'StruPNet'

    def __init__(self, opt):
        super(StruPNet, self).__init__()

        self.netG_A = networks.define_G(opt['input_nc'], opt['output_nc'],
                                        opt['ngf'], opt['which_model_netG'], opt['norm'], not opt['no_dropout'],
                                        final=opt['g_final'], img_size = opt['img_size'])
        self.netG_B = networks.define_G(opt['output_nc'], opt['input_nc'],
                                        opt['ngf'], opt['which_model_netG'], opt['norm'], not opt['no_dropout'],
                                        final=opt['g_final'], img_size = opt['img_size'])

        self.netG_seg_target = networks.define_G(opt['input_nc_seg'], opt['output_nc_seg'],
                                        opt['ngf'], opt['which_model_netSeg'], opt['norm'], not opt['no_dropout'],
                                        final = opt['seg_final'])
        
        self.netG_seg_source = networks.define_G(opt['input_nc_seg'], opt['output_nc_seg'],
                                        opt['ngf'], opt['which_model_netSeg'], opt['norm'], not opt['no_dropout'],
                                        final = opt['seg_final'])


        use_sigmoid = opt['no_lsgan']
        self.netD_A = networks.define_D(opt['output_nc'], opt['ndf'],
                                        opt['which_model_netD'],
                                        opt['n_layers_D'], opt['norm'], use_sigmoid)
        self.netD_B = networks.define_D(opt['input_nc'], opt['ndf'],
                                        opt['which_model_netD'],
                                        opt['n_layers_D'], opt['norm'], use_sigmoid)
        
        self.sobel_source = Sobel_Module(5,  15,  1/15)
        self.sobel_target = Sobel_Module(0.7,  2.5,  1/2.5)

        self.criterionGAN = networks.GANLoss(use_lsgan=True)
        self.criterionCycle = torch.nn.L1Loss()
        self.criterionSP = torch.nn.L1Loss()
        self.SegLoss = DiceLoss(sigmoid=False)

        self.lambda_A = opt['lambda_A']
        self.lambda_B = opt['lambda_B']
        self.lambda_C = opt['lambda_C']

        self.layer_weight = opt['layer_weight']

        self.optimizer_G = torch.optim.Adam(
                                        itertools.chain(self.netG_A.parameters(),
                                                        self.netG_seg_target.parameters(),
                                                        self.netG_seg_source.parameters(),
                                                        self.netG_B.parameters()),
                                        lr=opt['lr'], betas=(opt['beta1'], 0.999))
        self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt['lr'], betas=(opt['beta1'], 0.999))
        self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=opt['lr'], betas=(opt['beta1'], 0.999))

        self.warm_up = opt['warm_up']

        self.fake_A_pool = ImagePool(opt['pool_size'])
        self.fake_B_pool = ImagePool(opt['pool_size'])

        self.apply(self._init_weights)

        self.use_perloss = True
        self.perpectual_loss_netword = LossNetwork(opt['perpectual_weight'], opt['vgg_kernel_size'])



    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self,source,target):

            if self.training:
                fake_S_to_T, S_Feature = self.netG_A(source)

                cycle_fake_S_to_T_to_S, StoT_Feature = self.netG_B(fake_S_to_T)

                fake_T_to_S, T_Feature = self.netG_B(target)
                cycle_fake_T_to_S_to_T, TtoS_Feature = self.netG_A(fake_T_to_S)

                seg_s = self.netG_seg_source(source)
                seg_s2t = self.netG_seg_target(fake_S_to_T)
                seg_s2t2s = self.netG_seg_source(cycle_fake_S_to_T_to_S)

                seg_t = self.netG_seg_target(target)
                seg_t2s = self.netG_seg_source(fake_T_to_S)
                seg_t2s2t = self.netG_seg_target(cycle_fake_T_to_S_to_T)
                

                dict = {
                    'StoT': fake_S_to_T,
                    'StoTtoS': cycle_fake_S_to_T_to_S,
                    'TtoS': fake_T_to_S,
                    'TtoStoT':cycle_fake_T_to_S_to_T,

                    'seg_s':seg_s,
                    'seg_s2t':seg_s2t,
                    'seg_s2t2s':seg_s2t2s,
                    'seg_t':seg_t,
                    'seg_t2s':seg_t2s,
                    'seg_t2s2t':seg_t2s2t,

                    'S_Feature':S_Feature,
                    'StoT_Feature':StoT_Feature,
                    'T_Feature':T_Feature,
                    'TtoS_Feature':TtoS_Feature
                }

                return dict
            else:

                seg = self.netG_seg_target(target)
                dict = {
                    'seg':seg,
                }

                return dict

    def train_step(self,source_data, source_mask, target_data, epoch, idx):

        result_dict = self.forward(source_data, target_data)

        discrim_1_1 = self.netD_A(result_dict['StoT'])
        discrim_2_2 = self.netD_B(result_dict['TtoS'])  # 全是虚假图像，真实标签为0


        """
        optimize generators and segmentation networks
        """
        self.optimizer_G.zero_grad()
        # GAN loss
        # D_A(G_A(A))
        loss_G_A = self.criterionGAN(discrim_1_1, True)
        # D_B(G_B(B))
        loss_G_B = self.criterionGAN(discrim_2_2, True)
        # cycle consistency loss
        loss_cycle_A = self.criterionCycle(result_dict['StoTtoS'], source_data) * self.lambda_A
        loss_cycle_B = self.criterionCycle(result_dict['TtoStoT'], target_data) * self.lambda_B
        
        # Segmentation loss
        l_supervised = self.SegLoss(result_dict['seg_s'].float(), source_mask.float())
        l_1 = self.SegLoss(result_dict['seg_s2t2s'].float(), source_mask.float())
        l_2 = self.SegLoss(result_dict['seg_t2s2t'].float(), result_dict['seg_t'].float())
        l_3 = self.SegLoss(result_dict['seg_s2t'].float(), source_mask.float())
        l_4 = self.SegLoss(result_dict['seg_t2s'].float(), result_dict['seg_t'].float())

        if epoch == 0 :
            print(idx," : ",
                        round(loss_G_A.item(),3),
                        round(loss_G_B.item(),3),
                        round(loss_cycle_A.item(),3),
                        round(loss_cycle_B.item(),3),

                        round(l_supervised.item(),3),
                        #round(l_1.item(),3),
                        #round(l_2.item(),3),
                        round(l_3.item(),3),
                        #round(l_4.item(),3)
                        )

        if epoch < self.warm_up:
            lambdaseg = [1., 0., 0., 1., 0.]

        else:
            lambdaseg = [1., 0.1, 0.1, 1., 0.1]


        loss_seg = l_supervised * lambdaseg[0] + \
                l_1 * lambdaseg[1] + \
                l_2 * lambdaseg[2] + \
                l_3 * lambdaseg[3] + \
                l_4 * lambdaseg[4]


        step_train_loss = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_seg

        if epoch >= self.warm_up:
            """
            F-SP
            """
            loss_feature_align_S = 0.0
            for f_q, f_k, layer_weight in zip(result_dict['S_Feature'], result_dict['StoT_Feature'], self.layer_weight):
                loss = self.criterionSP(f_q,f_k) * layer_weight
                loss_feature_align_S += loss
            
            loss_feature_align_T = 0.0
            for f_q, f_k, layer_weight in zip(result_dict['T_Feature'], result_dict['TtoS_Feature'], self.layer_weight):
                loss = self.criterionSP(f_q,f_k) * layer_weight
                loss_feature_align_T += loss


            loss_feature_align_S = loss_feature_align_S * self.lambda_C
            loss_feature_align_T = loss_feature_align_T * self.lambda_C
            step_train_loss = step_train_loss + loss_feature_align_S + loss_feature_align_T


            """
            I-SP
            """
            sobelS = self.sobel_source(source_data)
            sobelStoT = self.sobel_target(result_dict['StoT'])
            sobelT = self.sobel_target(target_data)
            sobelTtoS = self.sobel_source(result_dict['TtoS'])

            loss_perpectual1 = self.perpectual_loss_netword(sobelStoT.expand(-1,3,-1,-1),sobelS.expand(-1,3,-1,-1))
            loss_perpectual2 = self.perpectual_loss_netword(sobelTtoS.expand(-1,3,-1,-1),sobelT.expand(-1,3,-1,-1))
            step_train_loss = step_train_loss + loss_perpectual1 + loss_perpectual2

        step_train_loss.backward()
        self.optimizer_G.step()


        """
        optimize DA
        """
        self.optimizer_D_A.zero_grad()

        fake_1_1 = self.fake_A_pool.query(result_dict['StoT'])
        discrim_1 = self.netD_A(target_data)
        discrim_1_1 = self.netD_A(fake_1_1.detach())


        loss_DA_real = self.criterionGAN(discrim_1, True)
        loss_DA_fake = self.criterionGAN(discrim_1_1, False)
        loss_DA = (loss_DA_real + loss_DA_fake) * 0.5
        loss_DA.backward()
        self.optimizer_D_A.step()


        """
        optimize DB
        """
        self.optimizer_D_B.zero_grad()

        fake_2_2 = self.fake_B_pool.query(result_dict['TtoS'])
        discrim_2 = self.netD_B(source_data)
        discrim_2_2 = self.netD_B(fake_2_2.detach())

        loss_DB_real = self.criterionGAN(discrim_2, True)
        loss_DB_fake = self.criterionGAN(discrim_2_2, False)
        loss_DB = (loss_DB_real + loss_DB_fake) * 0.5
        loss_DB.backward()
        self.optimizer_D_B.step()

        return step_train_loss.item() + loss_DA.item() + loss_DB.item()

    def val_step(self, patch_data, label_data):
        result_dict = self.forward(None, patch_data)
        pred = result_dict['seg']
        return self.SegLoss(pred.float(),label_data.float()).item()

    def updata_lr(self, lr):
        for params in self.optimizer_G.param_groups:
            params['lr'] = lr
        for params in self.optimizer_D_A.param_groups:
            params['lr'] = lr
        for params in self.optimizer_D_B.param_groups:
            params['lr'] = lr
