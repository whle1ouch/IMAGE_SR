import torch
import torch.nn as nn
import torch.nn.functional as F


class Swish(nn.Module):
    
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x * F.sigmoid(x)


class ResBlock(nn.Module):
    
    def __init__(self, num_filters):
        super().__init__()
    
        self.conv1 = nn.Conv2d(num_filters, num_filters, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.act1 = Swish()
        self.conv2 = nn.Conv2d(num_filters, num_filters, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(num_filters)
    

    def forward(self, x):
        y = self.act1(self.bn1(self.conv1(x)))
        return self.bn2(self.conv2(y)) + x
    

class UpsampleBlock(nn.Module):
    
    def __init__(self, num_filters):
        super().__init__()
        self.conv = nn.Conv2d(num_filters, num_filters * 4, 3, 1, 1)
        self.shuffler = nn.PixelShuffle(2)
        self.act = Swish()
    
    def forward(self, x):
        return self.act(self.shuffler(self.conv(x)))
    
class DiscriminatorBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride=1, batch_norm=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn = nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity()
        self.act = Swish()
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
        
class Generator(nn.Module):
    
    def __init__(self, num_res_block, upsample_factor, base_filters=64, out_channel=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, base_filters, 3, 1, 1)
        self.act1 = Swish()
        
        self.res_blocks = nn.ModuleList([ResBlock(base_filters) for _ in range(num_res_block)])
        
        self.conv2 = nn.Conv2d(base_filters, base_filters, 3, 1, 1)
        self.bn = nn.BatchNorm2d(base_filters)
        self.con = 23
        
        self.up_blocks = nn.ModuleList([UpsampleBlock(base_filters) for _ in range(upsample_factor // 2)])
        
        self.conv3 = nn.Conv2d(base_filters, out_channel, 9, 1, 4)
        
        
    def forward(self, x):
        
        x = self.act1(self.conv1(x))
        y = x
        for block in self.res_blocks:
            y = block(y)
        
        x = self.bn(self.conv2(y)) + x
        
        for up in self.up_blocks:
            x = up(x)
        
        return self.conv3(x)
    
class Discriminator(nn.Module):
    def __init__(self, base_filters=64, out_channel=1):
        super().__init__()
        
        self.backbone = nn.ModuleList([
            DiscriminatorBlock(3, base_filters, batch_norm=False),
            DiscriminatorBlock(base_filters, base_filters, stride=2),
            DiscriminatorBlock(base_filters, base_filters * 2),
            DiscriminatorBlock(base_filters * 2, base_filters * 2, stride=2),
            DiscriminatorBlock(base_filters * 2, base_filters * 4),
            DiscriminatorBlock(base_filters * 4, base_filters * 4, stride=2),
            DiscriminatorBlock(base_filters * 4, base_filters * 8),
            DiscriminatorBlock(base_filters * 8, base_filters * 8, stride=2),
        ])
        
        self.conv = nn.Conv2d(base_filters * 8, out_channel, 1, 1, 0)
        
    
    def forward(self, x: torch.Tensor):
        for block in self.backbone:
            x = block(x)
        
        x = self.conv(x)
        return torch.sigmoid(F.avg_pool2d(x, x.size()[2:])).view(x.size()[0], -1)




def init_weight(module: nn.Module, mean: float= 0., std: float = 0.02):
    for name in module._modules:
        m = module._modules[name]
        if isinstance(m, (nn.ModuleList, ResBlock, UpsampleBlock, DiscriminatorBlock)):
            init_weight(m, mean, std)
        if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
            m.weight.data.normal_(mean, std)
            m.bias.data.zero_()