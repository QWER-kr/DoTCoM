import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange  
from timm.models.registry import register_model

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )

def conv_nxn_bn(inp, oup, kernal_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernal_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )

def conv_nxn_DW(inp, oup, kernal_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernal_size, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )
        
class QIBBlock(nn.Module):
    channel_divide = 4
    def __init__(self, inp, oup, stride=1, expansion=4, active_QG=True, active_MV=True):
        super().__init__()

        self.active_QG = active_QG
        self.active_MV = active_MV
        self.stride = stride
        assert stride in [1, 2]

        dw_inp, dw_oup = (inp, oup) if not active_QG else (oup, oup)
        hidden_dim =  int(dw_inp * expansion) if not active_QG else int(oup * expansion)
        self.use_res_connect = self.stride == 1 and dw_inp == dw_oup

        if active_QG:
            self.conv_block = nn.Sequential(
                nn.Conv2d(inp, oup, kernel_size=3, stride=stride, padding=1, groups=QIBBlock.channel_divide, bias=False),
                nn.BatchNorm2d(oup),
                nn.SiLU(),
                nn.Conv2d(oup, oup, kernel_size=3, stride=1, padding=1, groups=QIBBlock.channel_divide, bias=False),
                nn.BatchNorm2d(oup)
            )

            self.downsample = nn.Sequential()
            if stride != 1 or inp != oup:
                self.downsample = nn.Sequential(
                                nn.Conv2d(inp, oup, kernel_size=1, stride=stride, bias=False),
                                nn.BatchNorm2d(oup)
                                )
            
        if active_MV:
            if expansion == 1:
                self.conv = nn.Sequential(
                    nn.Conv2d(dw_inp, dw_inp, 3, stride if not active_QG else 1, 1, groups=dw_inp, bias=False),
                    nn.BatchNorm2d(dw_inp),
                    nn.SiLU(),
                    nn.Conv2d(dw_inp, dw_oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(dw_oup),
                )
            else:
                self.conv = nn.Sequential(
                    nn.Conv2d(dw_inp, hidden_dim, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.SiLU(),
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride if not active_QG else 1, 1, groups=hidden_dim, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.SiLU(),
                    nn.Conv2d(hidden_dim, dw_oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(dw_oup),
                )
            
    def forward(self, x):
        if not self.active_QG:
            if self.use_res_connect:
                return x + self.conv(x)
            else:
                return self.conv(x)
            
        elif not self.active_MV:
            return F.silu(self.conv_block(x) + self.downsample(x))
        
        elif self.active_QG and self.active_MV:
            if self.use_res_connect:
                out = F.silu(self.conv_block(x) + self.downsample(x))
                return out + self.conv(out)
            else:
                out= F.silu(self.conv_block(x) + self.downsample(x))
                return self.conv(out)
            
        else:
            raise ValueError("There are no convolutions selected")
            
class Co_Bias(nn.Module):
    def __init__(self, out_chn):
        super(Co_Bias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out  

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        out = self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))
    
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class DoT(nn.Module):
    expansion = 1
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_DW(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)

        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout)

        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_1x1_bn(dim + channel, channel)

        self.move1 = Co_Bias(DoT.expansion * dim)
        self.move2 = Co_Bias(channel + DoT.expansion * dim)
    
    def forward(self, x):
        y = x.clone()

        x = self.conv1(x)
        x = self.conv2(x)
        cat = x
        x = self.move1(x)

        _, _, h, w = x.shape
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        x = self.transformer(x)
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)

        x = self.conv3(x)
        x = torch.cat((x, cat), 1)
        x = self.move2(x)
        x = self.conv4(x)
        x += y
        return x
    
class DoTCoM_Mechanism(nn.Module):
    def __init__(self, inp, mid, oup, stride, expansion, dim, depth, kernel_size, 
                 patch_size, active_QG=True, active_MV=True, dropout=0.):
        super().__init__()

        self.mechanism = nn.ModuleList([])
        self.mechanism.append(QIBBlock(inp, mid, stride, expansion, active_QG, active_MV))
        self.mechanism.append(QIBBlock(mid, mid, 1, expansion, active_QG, active_MV))
        self.mechanism.append(DoT(mid, depth, mid, kernel_size, patch_size, dim * depth, dropout))
        self.mechanism.append(QIBBlock(mid, oup, 1, expansion, False, True))

    def forward(self, x):
        for layer in self.mechanism:
            x = layer(x)
        return x

class DoTCoM(nn.Module):
    def __init__(self, channels, dims, depth, num_classes=1000, patch_size=(2, 2), kernel_size=3, **kwargs):
        super(DoTCoM, self).__init__()

        self.base = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=3, stride=2 ,padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU()
        )
        
        self.IB = QIBBlock(channels[0], channels[1], 1, 6, False, True)

        self.QIB = nn.ModuleList([])
        self.QIB.append(QIBBlock(channels[1], channels[2], 2, 6, True, True))
        self.QIB.append(QIBBlock(channels[2], channels[2], 1, 6, True, True))

        self.DoT = nn.ModuleList([])
        self.DoT.append(DoTCoM_Mechanism(channels[2], channels[3], channels[4], 2, 6, dims[0], depth[0], kernel_size, patch_size, True, True))
        self.DoT.append(DoTCoM_Mechanism(channels[4], channels[5], channels[6], 1, 6, dims[1], depth[1], kernel_size, patch_size, True, True))
        self.DoT.append(DoTCoM_Mechanism(channels[6], channels[7], channels[8], 2, 6, dims[2], depth[2], kernel_size, patch_size, True, True))
        
        self.QG = QIBBlock(channels[8], channels[-1], 2, 1, True, False)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels[-1], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d): 
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        out = self.base(x)

        out = self.IB(out)
        out = self.QIB[0](out)
        out = self.QIB[1](out)

        out = self.DoT[0](out)
        out = self.DoT[1](out)
        out = self.DoT[2](out)

        out = self.QG(out)

        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

@register_model
def dotcom_large(pretrained=False, num_classes=1000, **kwargs):
    model = DoTCoM(channels=[16, 32, 64, 96, 128, 160, 192, 224, 256, 1024], dims=[96, 160, 224], 
                   depth=[2, 3, 4], num_classes=num_classes, patch_size=(2, 2), kernel_size=3, **kwargs)
    return model

@register_model
def dotcom_base(pretrained=False, num_classes=1000, **kwargs):
    model = DoTCoM(channels=[16, 32, 48, 64, 80, 96, 128, 160, 192, 768], dims=[64, 96, 160], 
                   depth=[2, 3, 4], num_classes=num_classes, patch_size=(2, 2), kernel_size=3, **kwargs)
    return model

@register_model
def dotcom_small(pretrained=False, num_classes=1000, **kwargs):
    model = DoTCoM(channels=[16, 16, 32, 48, 64, 80, 96, 96, 128, 512], dims=[48, 80, 96], 
                   depth=[2, 3, 4], num_classes=num_classes, patch_size=(2, 2), kernel_size=3, **kwargs)
    return model

@register_model
def dotcom_tiny(pretrained=False, num_classes=1000, **kwargs):
    model = DoTCoM(channels=[16, 16, 24, 32, 40, 48, 64, 80, 96, 384], dims=[32, 48, 80], 
                   depth=[2, 3, 4], num_classes=num_classes, patch_size=(2, 2), kernel_size=3, **kwargs)
    return model