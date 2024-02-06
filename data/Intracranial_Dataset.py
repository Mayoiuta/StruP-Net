import os
import torch
import SimpleITK as sitk
from torch.utils.data import Dataset
import numpy as np
from z_utils import MRA_Norm,CTA_Norm,SYN_Norm


class MRA2CTA_DataGenerator(Dataset):
    def __init__(self, data, iter_num, norm1 = 'MRA', norm2 = 'CTA'):     
        self.norm1 = norm1
        self.norm2 = norm2

        assert self.norm1 in ['MRA','CTA','SYN'], 'DataGenerator Norm must in [“MRA”，“CTA”，“SYN”]'
        assert self.norm2 in ['MRA', 'CTA', 'SYN'], 'DataGenerator Norm must in [“MRA”，“CTA”，“SYN”]'

        self.source_img_list = data[0]
        self.source_seg_list = data[1]
        self.target_img_list = data[2]

        self.iter_num = iter_num

        print("source slice num:  ",len(self.source_img_list))
        print("traget slice num:  ",len(self.target_img_list))



    def preprocess(self, source, mask, target):
        if self.norm1 == 'MRA':
            source = MRA_Norm(source)
        elif self.norm1 == 'CTA':
            source = CTA_Norm(source)
        else:
            source = SYN_Norm(source)


        if self.norm2 == 'MRA':
            target = MRA_Norm(target)
        elif self.norm2 == 'CTA':
            target = CTA_Norm(target)
        else:
            target = SYN_Norm(target)

        mask = mask.astype(np.uint8)
        return source, mask, target

    def data_aug(self,source,mask,target):
        """
        """
        return source,mask,target



    def __getitem__(self, index):
        """
        len(source domain data) != len(target domain data)
        """

        source1 = np.random.randint(0,len(self.source_img_list))
        SourceImgSlice = sitk.GetArrayFromImage(sitk.ReadImage(self.source_img_list[source1]))
        SourceSegSlice = sitk.GetArrayFromImage(sitk.ReadImage(self.source_seg_list[source1]))

        target1 = np.random.randint(0,len(self.target_img_list))
        TargetImgSlice = sitk.GetArrayFromImage(sitk.ReadImage(self.target_img_list[target1]))

        input_s , input_m, input_t = self.preprocess(np.squeeze(SourceImgSlice),
                                                     np.squeeze(SourceSegSlice),
                                                     np.squeeze(TargetImgSlice))

        input_s = torch.FloatTensor(input_s).unsqueeze(0)
        input_t = torch.FloatTensor(input_t).unsqueeze(0)
        input_m = torch.FloatTensor(input_m).unsqueeze(0)

        return input_s,input_m,input_t

    def __len__(self):
        return self.iter_num
        

class MRA2CTA_DataGenerator_Val(Dataset):
    def __init__(self, data, norm = 'CTA'):     
        self.norm = norm

        assert self.norm in ['MRA','CTA','SYN'], 'Norm must in [“MRA”，“CTA”，“SYN”]'

        self.val_img_list = data[0]
        self.val_mask_list = data[1]

        print("val slice num : ", len(self.val_img_list))

    def preprocess(self,val,valmask):         

        if self.norm == 'MRA':
            val = MRA_Norm(val)
        elif self.norm == 'CTA':
            val = CTA_Norm(val)
        else:
            val = SYN_Norm(val)

        valmask = valmask.astype(np.uint8)
        return val,valmask

    def __getitem__(self, index):

        data_index = index
        Img = sitk.ReadImage(self.val_img_list[data_index])
        ImgSlice = sitk.GetArrayFromImage(Img)
        Mask = sitk.ReadImage(self.val_mask_list[data_index])
        MaskSlice = sitk.GetArrayFromImage(Mask)

        input_t, label_t = self.preprocess(np.squeeze(ImgSlice),
                                           np.squeeze(MaskSlice))

        input_t = torch.FloatTensor(input_t).unsqueeze(0)
        label_t = torch.FloatTensor(label_t).unsqueeze(0)

        return input_t,label_t

    def __len__(self):
        return len(self.val_img_list)
