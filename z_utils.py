import numpy as np
import os
import torch

def ToPng(arr):     #(-1,1)  ->  (0,255)
    arr = (arr + 1) / 2
    arr = arr * 255
    return arr.astype(np.uint8)

def ArrToCTA(arr):                      #(-1,1)  ->  (-1024,1024)
    return (arr * 1024).astype(np.int32)

def ArrToMRA(arr):                      #(-1,1)  ->  (0,300)
    return ((arr * 150) + 150).astype(np.int32)

def ArrToSYN(arr):                      #(-1,1)  ->  (0,300)
    return ((arr * 150) + 150).astype(np.int32)

def MRA_Norm(arr):                      #(0,300)  ->  (-1,1)
    arr = arr.astype(np.int32)
    arr = arr.clip(0, 300)
    arr = (arr-150) / 150
    return arr

def CTA_Norm(arr):                      #(-1024,1024)  ->  (-1,1)
    arr = arr.astype(np.int32)
    arr = arr.clip(-1024, 1024)
    arr = (arr) / 1024
    return arr

def SYN_Norm(arr):
    arr = arr.astype(np.int32)
    arr = arr.clip(0, 300)
    arr = (arr-150) / 150
    return arr

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def same_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)  
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed)  
        torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.enabled = False

