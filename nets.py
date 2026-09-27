import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import copy
import numpy as np


# ==========================================
# Global Classifier
# ==========================================
class Classifier(nn.Linear):
    def __init__(self, input_dim, num_classes):
        super(Classifier, self).__init__(in_features=input_dim, out_features=num_classes)

    def forward(self, x):
        return super(Classifier, self).forward(x)

# ==========================================
# Global Conditional Generator
# ==========================================

class ConditionalImageGenerator(nn.Module):
    def __init__(self, num_classes, noise_dim, img_channels=3, img_size=32):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.init_size = img_size // 4
        
        self.l1 = nn.Sequential(
            nn.Linear(noise_dim + num_classes, 256 * self.init_size * self.init_size),
            nn.BatchNorm1d(256 * self.init_size * self.init_size),
            nn.ReLU()
        )
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, img_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 256, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class ConditionalGenerator(nn.Module):
    def __init__(self, num_global_classes, noise_dim, output_dim, embedding_dim=32):
        super().__init__()
        self.noise_dim = noise_dim
        
        self.label_emb = nn.Embedding(num_global_classes, embedding_dim)
        
        input_dim = noise_dim + embedding_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, output_dim)
        )

    def forward(self, z, labels):
        c = self.label_emb(labels) 
        x = torch.cat([z, c], dim=1) 
        out = self.net(x)
        return out
    

class NLGenerator(nn.Module):
    def __init__(self, ngf=64, img_size=32, nc=3, nl=100, label_emb=None, le_emb_size=256, le_size=512, sbz=200):
        super(NLGenerator, self).__init__()
        self.params = (ngf, img_size, nc, nl, label_emb, le_emb_size, le_size, sbz)
        self.le_emb_size = le_emb_size
        self.label_emb = label_emb
        self.init_size = img_size // 4
        self.le_size = le_size
        self.nl = nl
        self.nle = int(np.ceil(sbz/nl))
        self.sbz = sbz

        self.n1 = nn.BatchNorm1d(le_size)
        self.sig1 = nn.Sigmoid()
        self.le1 = nn.ModuleList([nn.Linear(le_size, le_emb_size) for i in range(self.nle)])
        self.l1 = nn.Sequential(nn.Linear(le_emb_size, ngf * 2 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(ngf * 2),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(ngf*2, ngf*2, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(scale_factor=2),

            nn.Conv2d(ngf*2, ngf, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            nn.Sigmoid(),
        )

        self.dr1 = nn.Dropout(p=0.25)
        self.le_sig = nn.Sigmoid()

    def re_init_le(self):
        for i in range(self.nle):
            nn.init.normal_(self.le1[i].weight, mean=0, std=1)
            nn.init.constant_(self.le1[i].bias, 0)

    def forward(self, targets=None):
        le = self.label_emb[targets]
        le = self.n1(le)
        v = None
        for i in range(self.nle):
            if (i+1)*self.nl > le.shape[0]:
                sle = le[i*self.nl:]
            else:
                sle = le[i*self.nl:(i+1)*self.nl]
            sv = self.le1[i](sle)
            if v is None:
                v = sv
            else:
                v = torch.cat((v, sv))

        out = self.l1(v)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

    def reinit(self):
        return NLGenerator(self.params[0], self.params[1], self.params[2], self.params[3], self.params[4],
                             self.params[5], self.params[6], self.params[7])


# ==========================================
# Base
# ==========================================
class BaseHeteroModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_dim = 0 
    
    def forward(self, x):
        native_feat = self.feature_extractor(x)
        native_feat = torch.flatten(native_feat, 1)
        global_feat = self.adapter(native_feat)
        logits = self.classifier(global_feat)
        
        return global_feat, logits


# ==========================================
# 2. Heterogeneous Architectures
# ==========================================

# --- ID 0: MLP ---
class MLP(BaseHeteroModel):
    def __init__(self, in_channels, num_classes, img_size, global_dim=256):
        super().__init__()
        flat_dim = in_channels * img_size * img_size
        
        self.output_dim = 256 
        
        self.feature_extractor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, self.output_dim), 
            nn.ReLU()
        )

        self.adapter = nn.Sequential(
            nn.Linear(self.output_dim, global_dim),
            nn.ReLU() 
        )
        
        self.classifier = nn.Linear(global_dim, num_classes)

# --- ID 1: CNN ---
class CNN(BaseHeteroModel):
    def __init__(self, in_channels, num_classes, global_dim=256):
        super().__init__()
        self.output_dim = 64 
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), 
            nn.BatchNorm2d(32),
            nn.ReLU(), 
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(), 
            nn.MaxPool2d(2),

            nn.Conv2d(64, self.output_dim, 3, padding=1), 
            nn.BatchNorm2d(self.output_dim),
            nn.ReLU(), 

            nn.AdaptiveAvgPool2d((1, 1)) 
        )

        self.adapter = nn.Sequential(
            nn.Linear(self.output_dim, global_dim),
            nn.ReLU() 
        )
        
        self.classifier = nn.Linear(global_dim, num_classes)

# --- ID 2, 3: ResNet ---
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(BaseHeteroModel):
    def __init__(self, block, num_blocks, in_channels, num_classes, global_dim=256):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        if num_blocks[3] > 0:
            self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
            self.output_dim = 512 * block.expansion
        else:
            self.layer4 = nn.Identity()
            self.output_dim = 256 * block.expansion
            
        self.feature_extractor = nn.Sequential(
            self.conv1, self.bn1, nn.ReLU(),
            self.layer1, self.layer2, self.layer3, self.layer4,
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.adapter = nn.Sequential(
            nn.Linear(self.output_dim, global_dim),
            nn.ReLU()
        )
        
        self.classifier = nn.Linear(global_dim, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

# --- ID 4: MobileNetV2 ---
class MobileNetV2(BaseHeteroModel):
    def __init__(self, in_channels, num_classes, global_dim=256):
        super().__init__()
        model = models.mobilenet_v2(weights=None)
        if in_channels != 3:
            model.features[0][0] = nn.Conv2d(in_channels, 32, 3, 2, 1, bias=False)
        
        self.output_dim = 1280 
        
        self.feature_extractor = nn.Sequential(
            model.features,
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.adapter = nn.Sequential(
            nn.Linear(self.output_dim, global_dim),
            nn.ReLU() 
        )
        
        self.classifier = nn.Linear(global_dim, num_classes)

# --- ID 5: MobileNetV3 ---
class MobileNetV3(BaseHeteroModel):
    def __init__(self, in_channels, num_classes, global_dim=256):
        super().__init__()
        self.output_dim = 576
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.Hardswish(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1, groups=16, bias=False),
            nn.BatchNorm2d(32), nn.Hardswish(inplace=True),
            nn.Conv2d(32, self.output_dim, 1, bias=False),
            nn.BatchNorm2d(self.output_dim), nn.Hardswish(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
       
        self.adapter = nn.Sequential(
            nn.Linear(self.output_dim, global_dim),
            nn.ReLU() 
        )
        
        self.classifier = nn.Linear(global_dim, num_classes)

# --- ID 6: LeNet ---
class LeNet(BaseHeteroModel):
    def __init__(self, in_channels, num_classes, global_dim=256):
        super().__init__()
        self.output_dim = 84
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 6, 5, padding=2), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d((5, 5)), 
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
            nn.Linear(120, self.output_dim), nn.ReLU()
        )
        
        self.adapter = nn.Sequential(
            nn.Linear(self.output_dim, global_dim),
            nn.ReLU() 
        )
        
        self.classifier = nn.Linear(global_dim, num_classes)

# --- ID 7: AlexNet ---
class AlexNet(BaseHeteroModel):
    def __init__(self, in_channels, num_classes, global_dim=256):
        super().__init__()
        self.output_dim = 256 * 2 * 2 
        
        self.feature_extractor = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            
            # Layer 2
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Layer 3, 4, 5
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.AdaptiveAvgPool2d((2, 2)), 
            
            nn.Flatten()
        )
        
        self.adapter = nn.Sequential(
            nn.Linear(self.output_dim, global_dim),
            nn.ReLU()
        )
        
        self.classifier = nn.Linear(global_dim, num_classes)

# --- ID 8: ShuffleNetV2 ---
class ShuffleNetV2(BaseHeteroModel):
    def __init__(self, in_channels, num_classes, global_dim=256):
        super().__init__()
        model = models.shufflenet_v2_x0_5(weights=None)
        if in_channels != 3:
            model.conv1[0] = nn.Conv2d(in_channels, 24, 3, 2, 1, bias=False)
        
        self.output_dim = 1024
        backbone = list(model.children())[:-1] 
        self.feature_extractor = nn.Sequential(
            *backbone, 
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten() 
        )
        
        self.adapter = nn.Sequential(
            nn.Linear(self.output_dim, global_dim),
            nn.ReLU() 
        )
        
        self.classifier = nn.Linear(global_dim, num_classes)

# --- ID 9: SqueezeNet ---
class SqueezeNet(BaseHeteroModel):
    def __init__(self, in_channels, num_classes, global_dim=256):
        super().__init__()
        self.output_dim = 512
        model = models.squeezenet1_1(weights=None)
        if in_channels != 3:
            model.features[0] = nn.Conv2d(in_channels, 64, 3, 2)
            
        self.feature_extractor = nn.Sequential(
            model.features,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        self.adapter = nn.Sequential(
            nn.Linear(self.output_dim, global_dim),
            nn.ReLU()
        )
        
        self.classifier = nn.Linear(global_dim, num_classes)


# ==========================================
# Twin Branch Nets (for FedTED)
# ==========================================

class TwinBranchNets(nn.Module):
    def __init__(self, base_model: BaseHeteroModel):
        super().__init__()
        self.feature_extractor = base_model.feature_extractor
        self.adapter = base_model.adapter
        self.classifier = base_model.classifier  # generic branch
        self.twin_classifier = copy.deepcopy(base_model.classifier)  # personalized branch
        self.output_dim = base_model.output_dim
        self.use_twin = False
    
    def forward(self, x):
        native_feat = self.feature_extractor(x)
        native_feat = torch.flatten(native_feat, 1)
        global_feat = self.adapter(native_feat)
        
        if self.use_twin:
            logits = self.classifier(global_feat) + self.twin_classifier(global_feat)
        else:
            logits = self.classifier(global_feat)
        
        return global_feat, logits
    
# ==========================================
# GeFL (GAN)
# ==========================================

class DCGANGenerator(nn.Module):
    """ Conditional DCGAN Generator (32x32) """
    def __init__(self, num_classes, noise_dim=128, img_size=32, channels=3):
        super(DCGANGenerator, self).__init__()
        self.num_classes = num_classes
        self.noise_dim = noise_dim
        self.img_size = img_size

        input_dim = noise_dim + num_classes

        self.net = nn.Sequential(
            nn.ConvTranspose2d(input_dim, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z, y):
        z = z.view(z.size(0), self.noise_dim, 1, 1)
        y_onehot = F.one_hot(y, self.num_classes).float().view(y.size(0), self.num_classes, 1, 1)
        
        x = torch.cat([z, y_onehot], dim=1)
        return self.net(x)


class DCGANDiscriminator(nn.Module):
    def __init__(self, num_classes, img_size=32, channels=3):
        super(DCGANDiscriminator, self).__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        
        self.label_emb = nn.Embedding(num_classes, img_size * img_size)

        self.net = nn.Sequential(
            nn.Conv2d(channels + 1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 1, 4, 1, 0, bias=False)
        )

    def forward(self, x, y):
        y_emb = self.label_emb(y).view(y.size(0), 1, self.img_size, self.img_size)
        x_cat = torch.cat([x, y_emb], dim=1)
        
        return self.net(x_cat)
    

# ==========================================
# GeFL (DDPM)
# ==========================================

class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, is_res: bool = False) -> None:
        super().__init__()
        self.same_channels = in_channels==out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2 
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2

class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        layers = [ResidualConvBlock(in_channels, out_channels), nn.MaxPool2d(2)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x

class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)

class ContextUnet(nn.Module):
    def __init__(self, in_channels, n_feat = 256, n_classes=10):
        super(ContextUnet, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_classes = n_classes

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)
        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)
        self.to_vec = nn.Sequential(nn.AvgPool2d(8), nn.GELU())

        self.timeembed1 = EmbedFC(1, 2*n_feat)
        self.timeembed2 = EmbedFC(1, 1*n_feat)
        self.contextembed1 = EmbedFC(n_classes, 2*n_feat)
        self.contextembed2 = EmbedFC(n_classes, 1*n_feat)

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 8, 8), 
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )

        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x, c, t, context_mask):
        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2)
        
        c = nn.functional.one_hot(c, num_classes=self.n_classes).type(torch.float)
        
        context_mask = context_mask[:, None]
        context_mask = context_mask.repeat(1,self.n_classes)
        context_mask = (-1*(1-context_mask))
        c = c * context_mask
        
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        up1 = self.up0(hiddenvec)
        up2 = self.up1(cemb1*up1+ temb1, down2)
        up3 = self.up2(cemb2*up2+ temb2, down1)        
        out = self.out(torch.cat((up3, x), 1))
        return out
    

def ddpm_schedules(beta1, beta2, T):
    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t, "oneover_sqrta": oneover_sqrta,
        "sqrt_beta_t": sqrt_beta_t, "alphabar_t": alphabar_t,
        "sqrtab": sqrtab, "sqrtmab": sqrtmab,
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,
    }


class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, drop_prob=0.1):
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)

        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

    def forward(self, x, c):
        _ts = torch.randint(1, self.n_T+1, (x.shape[0],)).to(self.device)
        noise = torch.randn_like(x)

        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * noise
        )
        context_mask = torch.bernoulli(torch.zeros_like(c, dtype=torch.float)+self.drop_prob).to(self.device)
        
        return self.loss_mse(noise, self.nn_model(x_t, c, _ts / self.n_T, context_mask))

    def sample(self, n_sample, size, device, guide_w = 0.0, label=None):
        x_i = torch.randn(n_sample, *size).to(device)
        if label is not None:
            c_i = torch.full((n_sample,), label, dtype=torch.long, device=device)
        else:
            c_i = torch.arange(0, 10).to(device) 
            c_i = c_i.repeat(max(1, int(n_sample/c_i.shape[0])) + 1)[:n_sample]


        context_mask = torch.zeros_like(c_i).to(device)

        c_i = c_i.repeat(2)
        context_mask = context_mask.repeat(2)
        context_mask[n_sample:] = 1.

        x_i_store = []
        for i in range(self.n_T, 0, -1):
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample,1,1,1)

            x_i_double = x_i.repeat(2,1,1,1)
            t_is = t_is.repeat(2,1,1,1)

            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0

            eps = self.nn_model(x_i_double, c_i, t_is, context_mask)
            eps1 = eps[:n_sample]
            eps2 = eps[n_sample:]
            eps = (1+guide_w)*eps1 - guide_w*eps2
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
            if i%20==0 or i==self.n_T or i<8:
                x_i_store.append(x_i.detach().cpu().numpy())
        
        return x_i, x_i_store

class DDIM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, drop_prob=0.1, n_steps=50):
        super(DDIM, self).__init__()
        self.nn_model = nn_model.to(device)

        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()
        self.n_steps = n_steps

    def forward(self, x, c):
        _ts = torch.randint(1, self.n_T + 1, (x.shape[0],)).to(self.device)

        noise = torch.randn_like(x)
        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * noise
        )
        context_mask = torch.bernoulli(
            torch.zeros_like(c, dtype=torch.float) + self.drop_prob
        ).to(self.device)

        return self.loss_mse(noise, self.nn_model(x_t, c, _ts / self.n_T, context_mask))

    def sample(self, n_sample, size, device, guide_w=0.0, label=None, n_steps=None):
        if n_steps is None:
            n_steps = self.n_steps

        x_i = torch.randn(n_sample, *size).to(device)

        if label is not None:
            c_i = torch.full((n_sample,), label, dtype=torch.long, device=device)
        else:
            c_i = torch.arange(0, 10).to(device)
            c_i = c_i.repeat(max(1, int(n_sample / c_i.shape[0])) + 1)[:n_sample]

        context_mask = torch.zeros_like(c_i).to(device)
        c_i = c_i.repeat(2)
        context_mask = context_mask.repeat(2)
        context_mask[n_sample:] = 1.

        timesteps = torch.linspace(self.n_T, 1, n_steps).long().to(device)

        x_i_store = []
        for i in range(len(timesteps)):
            t = timesteps[i].item()
            t_prev = timesteps[i + 1].item() if i + 1 < len(timesteps) else 0

            t_is = torch.tensor([t / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample, 1, 1, 1)

            x_i_double = x_i.repeat(2, 1, 1, 1)
            t_is_double = t_is.repeat(2, 1, 1, 1)

            eps_double = self.nn_model(x_i_double, c_i, t_is_double, context_mask)
            eps1 = eps_double[:n_sample]
            eps2 = eps_double[n_sample:]
            eps = (1 + guide_w) * eps1 - guide_w * eps2

            alpha_bar_t = self.alphabar_t[t]
            alpha_bar_prev = self.alphabar_t[t_prev] if t_prev > 0 else torch.tensor(1.0).to(device)

            x0_pred = (x_i - torch.sqrt(1 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)
            x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

            x_i = torch.sqrt(alpha_bar_prev) * x0_pred + torch.sqrt(1 - alpha_bar_prev) * eps

            if i % 10 == 0 or i == len(timesteps) - 1:
                x_i_store.append(x_i.detach().cpu().numpy())

        return x_i, x_i_store


# ==========================================
# Factory Function
# 用 client_id % 10 決定這個 client 用哪一種 backbone -> Model Heterogeneity
# ==========================================
def get_heterogeneous_model(node_id, in_channels, num_classes, img_size, global_dim=256):
    model_idx = node_id % 10
    
    if model_idx == 0: return MLP(in_channels, num_classes, img_size, global_dim)
    
    if model_idx == 1: return CNN(in_channels, num_classes, global_dim)
    
    if model_idx == 2: return ResNet(BasicBlock, [1, 1, 1, 0], in_channels, num_classes, global_dim) # ResNet8
    if model_idx == 3: return ResNet(BasicBlock, [2, 2, 2, 2], in_channels, num_classes, global_dim) # ResNet18
    
    if model_idx == 4: return MobileNetV2(in_channels, num_classes, global_dim)
    if model_idx == 5: return MobileNetV3(in_channels, num_classes, global_dim)
    if model_idx == 6: return LeNet(in_channels, num_classes, global_dim)
    if model_idx == 7: return AlexNet(in_channels, num_classes, global_dim)
    if model_idx == 8: return ShuffleNetV2(in_channels, num_classes, global_dim)
    if model_idx == 9: return SqueezeNet(in_channels, num_classes, global_dim)
    
    return MLP(in_channels, num_classes, img_size, global_dim)


