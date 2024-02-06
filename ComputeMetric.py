import SimpleITK as sitk
import numpy as np

def all_metric(infer_path,refer_path):
    print("infer:  ", infer_path)
    print("refer:  ", refer_path)

    infer_ = sitk.ReadImage(infer_path)
    refer_ = sitk.ReadImage(refer_path)


    infer_arr = sitk.GetArrayFromImage(infer_)
    refer_arr = sitk.GetArrayFromImage(refer_)

    TP = np.sum((infer_arr == 1) * (refer_arr == 1))
    #TN = np.sum((infer_arr == 0) * (refer_arr == 0))
    FP = np.sum((infer_arr == 1) * (refer_arr == 0))
    FN = np.sum((infer_arr == 0) * (refer_arr == 1))

    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
   
    intersection = (infer_arr * refer_arr).sum()
    union = infer_arr.sum() + refer_arr.sum()

    dice = 2 * intersection / union
    iou = intersection / (union - intersection)

    return dice, iou, precision, recall



def ComputeMetric(pred_list, label_list):

    dice_aa = []
    iou_aa = []
    precision_aa = []
    recall_aa = []

    


    for idx, infer_path in enumerate(pred_list):
        refer_path = label_list[idx]
        dice2, iou2, precision, recall  =  all_metric(infer_path,refer_path)

        dice_aa.append(dice2*100)
        iou_aa.append(iou2*100)
        precision_aa.append(precision*100)
        recall_aa.append(recall*100)

    print("dice     :","{:.4f}".format(np.mean(dice_aa))," -- ", "{:.4f}".format(np.std(dice_aa)))
    print("iou      :","{:.4f}".format(np.mean(iou_aa))," -- ", "{:.4f}".format(np.std(iou_aa)))
    print("precision:","{:.4f}".format(np.mean(precision_aa))," -- ", "{:.4f}".format(np.std(precision_aa)))
    print("recall   :","{:.4f}".format(np.mean(recall_aa))," -- ", "{:.4f}".format(np.std(recall_aa)))

    print("dice:",dice_aa)
    print("iou:",iou_aa)
    print("precision:",precision_aa)
    print("recall:",recall_aa)

if __name__=='__main__':



    ComputeMetric("testmodel")

  