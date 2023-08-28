from pathlib import Path
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision.models.vgg import vgg16, vgg19
from tqdm import tqdm

from .model import Generator, Discriminator, init_weight


class SRGANTrainer:
    
    def __init__(self, config):
        self.use_gpu = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        self.netG = None
        self.netD = None
        self.lr = config.lr
        self.epochs = config.epochs
        self.save_path = Path(config.save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
    
        self.epoch_pretrain = 10
        self.criterionG = None
        self.criterionD = None
        self.optimizerG = None
        self.optimizerD = None
        self.feature_extractor = None
        self.schedulerG = None
        self.schedulerD = None
        self.seed = config.seed
        self.upscale_factor = config.upscale_factor
        self.num_residuals = 16
        self.train_loader = None
        self.test_loader = None
    
    def build_model(self):
        self.netG = Generator(self.num_residuals, self.upscale_factor,64, 1).to(self.device)
        self.netD = Discriminator(64, 1).to(self.device)
        init_weight(self.netG)
        init_weight(self.netD)
        self.feature_extractor = vgg16(pretrained=True)
        
        self.criterionG = nn.MSELoss()
        self.criterionD = nn.BCELoss()
        torch.manual_seed(self.seed)
        
        if self.use_gpu:
            torch.cuda.manual_seed(self.seed)
            self.feature_extractor.cuda()
            cudnn.benchmark = True
            # self.criterionG.cuda()
            # self.criterionD.cuda()
        
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=self.lr, betas=(0.9, 0.999))
        self.optimizerD = optim.SGD(self.netD.parameters(), lr=self.lr / 100, momentum=0.9, nesterov=True)
        
        self.schedulerG = optim.lr_scheduler.MultiStepLR(self.optimizerG, milestones=[50, 75, 100], gamma=0.5)
        self.schedulerD = optim.lr_scheduler.MultiStepLR(self.optimizerD, milestones=[50, 75, 100], gamma=0.5)

    def build_dataset(self):
        ...
        
    def save(self):
        g_model_out_path = self.save_path / "srgan_generator_model.pth"
        d_model_out_path = self.save_path /"srgan_discriminator_model.path"
        torch.save(self.netG, g_model_out_path)
        torch.save(self.netD, d_model_out_path)
        print(f"Checkpoint saved to {g_model_out_path}")
        print(f"Checkpoint saved to {d_model_out_path}")
    
    def pretrain(self, epoch):
        self.netG.train()
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), bar_format="{l_bar}{bar:20}{r_bar}")
        for _, (data, target) in pbar:
            data, target = data.to(self.device), target.to(self.device)
            self.netG.zero_grad()
            loss = self.criterionG(self.netG(data), target)
            loss.backward()
            self.optimizerG.step()
            pbar.set_description(
                f"{epoch}/{self.epochs}, , G_Loss: {loss :.4f}")
    
    def train(self, epoch):
        nb = len(self.train_loader)
        pbar = tqdm(enumerate(self.train_loader), total=nb, bar_format="{l_bar}{bar:20}{r_bar}")
        g_train_loss = 0
        d_train_loss = 0
        
        self.netG.train()
        self.netD.train()
        for i, (data, target) in pbar:
            real_label = torch.ones(data.size(0), data.size(1)).to(self.device)
            fake_label = torch.zeros(data.size(0), data.size(1)).to(self.device)
            data, target = data.to(self.device), target.to(self.device)
            
            # Train Discriminator
            self.optimizerD.zero_grad()
            d_real = self.netD(target)
            d_real_loss = self.criterionD(d_real, real_label)

            d_fake = self.netD(self.netG(data))
            d_fake_loss = self.criterionD(d_fake, fake_label)
            d_total = d_real_loss + d_fake_loss
            d_train_loss += d_total.item()
            d_total.backward()
            self.optimizerD.step()

            # Train generator
            self.optimizerG.zero_grad()
            g_real = self.netG(data)
            g_fake = self.netD(g_real)
            gan_loss = self.criterionD(g_fake, real_label)
            mse_loss = self.criterionG(g_real, target)

            g_total = mse_loss + 1e-3 * gan_loss
            g_train_loss += g_total.item()
            g_total.backward()
            self.optimizerG.step()
            mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
            pbar.set_description(f"{epoch}/{self.epochs}, {mem}, G_Loss: {g_train_loss / (i+1) :.4f}, D_Loss: {d_train_loss / (i+1) :.4f}")
        avg_g_loss = g_train_loss / nb
        print(f"    Average G_Loss: {avg_g_loss :.4f}")
        return avg_g_loss
            
            
        
    def test(self, epoch):
        self.netG.eval()
        avg_psnr = 0
        nb = len(self.test_loader)
        pbar = tqdm(enumerate(self.test_loader), total=nb, bar_format="{l_bar}{bar:20}{r_bar}")
        with torch.no_grad():
            for i, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                prediction = self.netG(data)
                mse = self.criterionG(prediction, target)
                psnr = 10 * log10(1 / mse.item())
                avg_psnr += psnr
                pbar.set_description(f"{epoch}/{self.epochs}, PSNR: {avg_psnr / (i+1) :.4f}")
            print(f"    Average PSNR: {avg_psnr / nb:.4f} dB")
            
    def run(self):
        self.build_model()
        self.build_dataset()
        
        for epoch in range(1, self.epoch_pretrain + 1):
            self.pretrain(epoch)
        
        global_loss = 1e10
        for epoch in range(1, self.epochs + 1):
            avg_loss = self.train(epoch)
            self.test(epoch)
            self.schedulerG.step(epoch)
            self.schedulerD.step(epoch)
            
            if avg_loss < global_loss:
                global_loss = avg_loss
                self.save()
            
            