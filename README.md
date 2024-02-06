# StruP-Net
Part of the code comes from [CycleGAN Torch](https://github.com/junyanz/CycleGAN) and [SynSeg-Net](https://github.com/MASILab/SynSeg-Net).

# Train
- Train:

Fill in the 'train.py / get_data_split_from_file', then
```bash
python train.py
```

- If run this code on other data:

1. Please check the normalization method in 'models / Dataseg.py' and 'z_utils.py'.
2. Change the parameters of Sobel filters in 'models / StruP-Net.py'.
