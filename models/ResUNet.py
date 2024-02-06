import torch
import torch.nn as nn

def downsample():
    return nn.MaxPool2d(kernel_size=2, stride=2)
def deconv(in_channels, out_channels):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
 
class GCN_Linear(nn.Module):
    def __init__(self, channel, hw):
        super(GCN_Linear, self).__init__()  

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.para = torch.nn.Parameter(torch.ones((1,channel,hw,hw), dtype = torch.float32))
        self.adj = torch.nn.Parameter(torch.ones((channel,channel), dtype = torch.float32))

        self.linear = nn.Linear(channel,channel)
             
    def forward(self, x):
        b, c, H, W = x.size()
        fea_matrix = x.view(b,c,H*W)
        c_adj = self.avg_pool(x).view(b,c)

        m = torch.ones((b,c,H,W), dtype = torch.float32)

        for i in range(0,b):

            t1 = c_adj[i].unsqueeze(0)
            t2 = t1.t()
            c_adj_s = torch.abs(torch.abs(torch.sigmoid(t1-t2)-0.5)-0.5)*2
            c_adj_s = (c_adj_s.t() + c_adj_s)/2

            output0 = torch.mul(torch.mm(self.adj*c_adj_s,fea_matrix[i]).view(1,c,H,W),self.para)

            m[i] = output0

        output = m.cuda()

        output = output.permute(0, 2, 3, 1)       # (N, C, H, W) -> (N, H, W, C)
        output = self.linear(output)
        output = output.permute(0, 3, 1, 2)

        return output

class ResEncoder(nn.Module):
    def __init__(self, in_channels, out_channels,norm_layer):
        super(ResEncoder, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.norm1 = norm_layer(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.norm2 = norm_layer(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.convres = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            bias=True,
        )

    def forward(self, x):
        res = self.convres(x)
        y = self.relu1(self.norm1(self.conv1(x)))
        y = self.relu2(self.norm2(self.conv2(y)))
        y = y + res

        return y

class ResUNet(nn.Module):
    def __init__(self, in_channels,num_classes, base_channels = 32,norm_layer=nn.BatchNorm2d,final_layer = 'tanh'):
        super(ResUNet, self).__init__()

        self.encoder1 = ResEncoder(in_channels, 1*base_channels,norm_layer)
        self.encoder2 = ResEncoder(1*base_channels, 2*base_channels,norm_layer)
        self.encoder3 = ResEncoder(2*base_channels, 4*base_channels,norm_layer)
        self.encoder4 = ResEncoder(4 * base_channels, 8 * base_channels,norm_layer)

        self.bottom = ResEncoder(8*base_channels, 16*base_channels,norm_layer)

        self.up1 = deconv(16 * base_channels, 8 * base_channels)
        self.decoder1 = ResEncoder(16 * base_channels, 8 * base_channels,norm_layer)

        self.up2 = deconv(8 * base_channels, 4 * base_channels)
        self.decoder2 = ResEncoder(8 * base_channels, 4 * base_channels,norm_layer)

        self.up3 = deconv(4 * base_channels, 2 * base_channels)
        self.decoder3 = ResEncoder(4 * base_channels, 2 * base_channels,norm_layer)

        self.up4 = deconv(2 * base_channels, 1 * base_channels)
        self.decoder4 = ResEncoder(2 * base_channels, 1 * base_channels,norm_layer)

        self.down = downsample()
        self.final = nn.Conv2d(base_channels, num_classes, kernel_size=1, stride=1)

        if final_layer == 'sigmoid':
            self.final_act = nn.Sigmoid()
        elif final_layer == 'tanh':
            self.final_act = nn.Tanh()
        elif final_layer == 'softmax':
            self.final_act = nn.Softmax(dim = 1)

    def forward(self, x):
        enc1 = self.encoder1(x)                 #64,128
        down1 = self.down(enc1)                 #64,64

        enc2 = self.encoder2(down1)             #128,64
        down2 = self.down(enc2)                 #128,32

        enc3 = self.encoder3(down2)             #256,32
        down3 = self.down(enc3)                 #256,16

        enc4 = self.encoder4(down3)             #512,16
        down4 = self.down(enc4)                 #512,8

        output = self.bottom(down4)             #1024,8

        output = self.up1(output)               #512,16
        output = torch.cat([output,enc4],dim=1) #1024,16
        output = self.decoder1(output)          #512,16

        output = self.up2(output)               #256,32
        output = torch.cat([output,enc3],dim=1)       #512,32
        output = self.decoder2(output)          #256,32

        output = self.up3(output)               #128,64
        output = torch.cat([output,enc2],dim=1)       #256,64
        output = self.decoder3(output)          #128,64

        output = self.up4(output)               #64,128
        output = torch.cat([output,enc1],dim=1)       #128,128
        output = self.decoder4(output)          #64,128

        output = self.final_act(self.final(output))

        return output
 
class ResUNet_FeatureAlign_GCNLinear5(nn.Module):
    def __init__(self, in_channels, num_classes, base_channels = 32,norm_layer=nn.BatchNorm2d,final_layer = 'tanh', img_size = 448):
        super(ResUNet_FeatureAlign_GCNLinear5, self).__init__()

        self.encoder1 = ResEncoder(in_channels, 1*base_channels,norm_layer)
        self.encoder2 = ResEncoder(1*base_channels, 2*base_channels,norm_layer)
        self.encoder3 = ResEncoder(2*base_channels, 4*base_channels,norm_layer)
        self.encoder4 = ResEncoder(4 * base_channels, 8 * base_channels,norm_layer)

        self.bottom = ResEncoder(8*base_channels, 16*base_channels,norm_layer)

        self.up1 = deconv(16 * base_channels, 8 * base_channels)
        self.decoder1 = ResEncoder(16 * base_channels, 8 * base_channels,norm_layer)

        self.up2 = deconv(8 * base_channels, 4 * base_channels)
        self.decoder2 = ResEncoder(8 * base_channels, 4 * base_channels,norm_layer)

        self.up3 = deconv(4 * base_channels, 2 * base_channels)
        self.decoder3 = ResEncoder(4 * base_channels, 2 * base_channels,norm_layer)

        self.up4 = deconv(2 * base_channels, 1 * base_channels)
        self.decoder4 = ResEncoder(2 * base_channels, 1 * base_channels,norm_layer)

        self.down = downsample()
        self.final = nn.Conv2d(base_channels, num_classes, kernel_size=1, stride=1)

        self.feature_align = nn.ModuleList([
            GCN_Linear(base_channels, img_size),
            GCN_Linear(2 * base_channels, int(img_size/2)),
            GCN_Linear(4 * base_channels, int(img_size/4)),
            GCN_Linear(8 * base_channels, int(img_size/8)),
            GCN_Linear(16 * base_channels, int(img_size/16))
        ])

        if final_layer == 'sigmoid':
            self.final_act = nn.Sigmoid()
        elif final_layer == 'tanh':
            self.final_act = nn.Tanh()

    def forward(self, x):

        feature_list = []

        enc1 = self.encoder1(x)               
        feature_list.append(self.feature_align[0](enc1))
        down1 = self.down(enc1)                

        enc2 = self.encoder2(down1)            
        feature_list.append(self.feature_align[1](enc2))
        down2 = self.down(enc2)               

        enc3 = self.encoder3(down2)            
        feature_list.append(self.feature_align[2](enc3))
        down3 = self.down(enc3)                

        enc4 = self.encoder4(down3)          
        feature_list.append(self.feature_align[3](enc4))
        down4 = self.down(enc4)              

        output = self.bottom(down4)           

        deep_feature = output
        feature_list.append(self.feature_align[4](deep_feature))

        output = self.up1(output)            
        output = torch.cat([output,enc4],dim=1) 
        output = self.decoder1(output)          

        output = self.up2(output)              
        output = torch.cat([output,enc3],dim=1)       
        output = self.decoder2(output)          

        output = self.up3(output)               
        output = torch.cat([output,enc2],dim=1)       
        output = self.decoder3(output)         

        output = self.up4(output)              
        output = torch.cat([output,enc1],dim=1)     
        output = self.decoder4(output)         

        output = self.final_act(self.final(output))

        return output, feature_list
