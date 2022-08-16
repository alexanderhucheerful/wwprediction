#import timm
import torch.nn as nn
#from timm.models.layers import Conv2dSame 

def make_mask(pretrained):
    masknet = nn.Module()
    def build_layer(blk):
        layers = []
        for _, m in blk.named_modules():
            if isinstance(m, Conv2dSame):
                kernel_size = [int(ks) //2 for ks in m.kernel_size]
                layer = nn.MaxPool2d(m.kernel_size, stride=m.stride, padding=kernel_size)            
                layers.append(layer) 
            elif isinstance(m, nn.Conv2d):
                layer = nn.MaxPool2d(m.kernel_size, stride=m.stride, padding=m.padding)   
                layers.append(layer) 
        return nn.Sequential(*layers)     

    masknet.layer0 = build_layer(pretrained.layer0)
    masknet.layer1 = build_layer(pretrained.layer1)
    masknet.layer2 = build_layer(pretrained.layer2)
    masknet.layer3 = build_layer(pretrained.layer3)
    return masknet


def make_vgg(in_dim, model_name="vgg19"):
    model = timm.create_model(model_name, pretrained=True, in_chans=in_dim)
    pretrained = nn.Module()
    pretrained.layer0 = nn.Sequential(*model.features[0:2])
    pretrained.layer1 = nn.Sequential(*model.features[2:7])
    # pretrained.layer2 = nn.Sequential(*model.features[7:12])
    # pretrained.layer3 = nn.Sequential(*model.features[12:21])
    return pretrained


def make_efficientnet(in_dim, model_name="tf_efficientnet_lite0"):
    model = timm.create_model(model_name, pretrained=False, in_chans=in_dim)
    pretrained = nn.Module()
    pretrained.layer0 = nn.Sequential(model.conv_stem, model.bn1, model.act1)
    pretrained.layer1 = nn.Sequential(*model.blocks[0:2])
    pretrained.layer2 = nn.Sequential(*model.blocks[2:3])
    pretrained.layer3 = nn.Sequential(*model.blocks[3:5])
    return pretrained

class RandomProj(nn.Module):
    def __init__(self, in_dim, thresh=0.3):        
        super().__init__()
        self.thresh = thresh
        self.eft = make_efficientnet(in_dim).eval()
        self.msk = make_mask(self.eft)

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        out0 = self.eft.layer0(x)
        out1 = self.eft.layer1(out0)
        out2 = self.eft.layer2(out1)
        out3 = self.eft.layer3(out2)
        outs = [out0, out1, out2, out3]
        return outs
    
    
    def forward_mask(self, x):
        x = x.max(1, keepdims=True)[0]
        x = (x > self.thresh).float()
        out0 = self.msk.layer0(x)
        out1 = self.msk.layer1(out0)
        out2 = self.msk.layer2(out1)
        out3 = self.msk.layer3(out2)
        outs = [out0, out1, out2, out3]
        return outs






