import os
import torch
import SimpleITK as sitk

from trainer import Trainer
from z_utils import *
from data.Intracranial_Dataset import MRA2CTA_DataGenerator, MRA2CTA_DataGenerator_Val
from models.StruPNet import StruPNet

def get_data_split_from_file(MRA2CTA):
    """
    
    """
    return None

def mra2cta_datagenerator(iter_num):
    MRA_img_list, MRA_seg_list, CTA_img_list_train, CTA_img_list_val, CTA_seg_list_val = get_data_split_from_file(MRA2CTA = True)

    train_Dataloader = MRA2CTA_DataGenerator([MRA_img_list,MRA_seg_list, CTA_img_list_train],
                                iter_num = iter_num, norm1 = 'MRA', norm2 = 'CTA')       
    val_Dataloader = MRA2CTA_DataGenerator_Val([CTA_img_list_val,CTA_seg_list_val],norm = 'CTA')

    return train_Dataloader, val_Dataloader

def infer_target(model, file, mod, model_name, checkpoint_dir = './CheckpointAndLog/'):
    print(file)
    img = sitk.ReadImage(file)
    img = sitk.GetArrayFromImage(img)

    if mod == 'CTA':
        img = CTA_Norm(img)
    elif mod == 'MRA':
        img = MRA_Norm(img)

    seg = np.zeros_like(img)

    for i in range(img.shape[0]):
        one_slice = img[i,:,:].copy()
        one_slice = torch.FloatTensor(one_slice).unsqueeze(0).unsqueeze(0).cuda()
        _ = torch.zeros_like(one_slice)

        output = model(_,one_slice)['seg']
        output = output.cpu().squeeze(dim=0).detach().numpy()
        seg[i:i+1, :, :] += output

    seg = (seg>=0.5).astype(np.uint8)
    seg = sitk.GetImageFromArray(seg)
    
    if not os.path.exists(checkpoint_dir+ model_name +'/InferResult'):
        os.makedirs(checkpoint_dir+ model_name +'/InferResult')
    
    new_name = file.split('/')[-1]
    new_name = new_name.split('.')[0]
    seg_save_dir = checkpoint_dir+ model_name +'/InferResult/'+new_name+'.nii.gz'
    print(seg_save_dir)
    sitk.WriteImage(seg ,seg_save_dir)

    return seg_save_dir

if __name__ == '__main__':

    torch.cuda.empty_cache()
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"

    seed = 12121
    same_seeds(seed)

    num_workers = 1
    batch_size = 1
    model_name = "test"

    iter_num = 1024

    train_Dataloader, val_Dataloader = mra2cta_datagenerator(iter_num)

    opt = {}
    opt['input_nc'] = 1
    opt['output_nc'] = 1
    opt['ngf'] = 32
    opt['ndf'] = 32
    opt['which_model_netG'] = 'resunet_feature_align_GCNLinear5'
    opt['img_size'] = 448
    opt['which_model_netSeg'] = 'resunet'
    opt['norm'] = 'instance'
    opt['no_dropout'] = True
    opt['seg_final'] = 'sigmoid'                    # final layer of segmentation networks
    opt['g_final'] = 'tanh'                         # final layer of generators
    opt['pool_size'] = 50

    opt['input_nc_seg'] = 1
    opt['output_nc_seg'] = 1
    opt['which_model_netD'] = 'basic'
    opt['n_layers_D'] = 3

    opt['no_lsgan'] = False

    opt['lambda_A'] = 5                             # weight of cycle consistency loss
    opt['lambda_B'] = 5                             # weight of cycle consistency loss
    opt['lambda_C'] = 0.1                           # weight of F-SP loss
    opt['layer_weight'] = [0.0625 ,0.125,0.25,0.5,1]           #[1/16, 1/8, 1/4, 1/2, 1]

    opt['lr'] = 1e-4
    opt['beta1'] = 0.5

    opt['warm_up'] = 1                              # warm up epoch
    
    opt['pl_model'] = 'vgg19'
    opt['perpectual_weight'] = [0.0625 ,0.125,0.25,0.5,1]       #[1/16, 1/8, 1/4, 1/2, 1]
    opt['vgg_kernel_size'] = 3

    print(opt)


    # train             #################################################################################################################                
    model = StruPNet(opt)

    model_param_size = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (model_param_size / 1e6))

    trainer = Trainer(batch_size,
                      batch_size,
                      num_workers,
                      train_Dataloader,
                      val_Dataloader,
                      model,
                      model_name,
                      max_epoch=50,
                      initial_lr=opt['lr'])

    trainer.run_train()



    # test             #################################################################################################################
    model = StruPNet(opt)
    print("finish")
    model = model.cuda()
    checkpoint = torch.load('./CheckpointAndLog/' + model_name + '/CheckPoint/' + model_name + '_50_' + str(opt['lr']) +'.pkl')
    net_dict = checkpoint['net_dict']
    model.load_state_dict(net_dict)
    model.eval()

    files = [
        '../VesselData/CTA_resampled_cut/002/img_no_skull.nii.gz',
        '../VesselData/CTA_resampled_cut/009/img_no_skull.nii.gz',
        '../VesselData/CTA_resampled_cut/078/img_no_skull.nii.gz',
        '../VesselData/CTA_resampled_cut/086/img_no_skull.nii.gz',
        '../VesselData/CTA_resampled_cut/110/img_no_skull.nii.gz',
        '../VesselData/CTA_resampled_cut/112/img_no_skull.nii.gz',
    ]

    label_file_list = [
        '../VesselData/CTA_resampled_cut/002/label.nii.gz',
        '../VesselData/CTA_resampled_cut/009/label.nii.gz',
        '../VesselData/CTA_resampled_cut/078/label.nii.gz',
        '../VesselData/CTA_resampled_cut/086/label.nii.gz',
        '../VesselData/CTA_resampled_cut/110/label.nii.gz',
        '../VesselData/CTA_resampled_cut/112/label.nii.gz',
    ]

    """
    1. infer and save results
    2. compute metrics
    """
    seg_save_file_list = []
    for file in files:
        seg_save_file_list.append(infer_target(model, file, mod='CTA', model_name=model_name))
    
    from ComputeMetric import ComputeMetric
    ComputeMetric(seg_save_file_list, label_file_list)




