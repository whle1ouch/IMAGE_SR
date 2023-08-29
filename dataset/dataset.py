from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms as T
from tqdm import tqdm

from .common import center_pad

class DIV2K(Dataset):
    
    def __init__(self, data_path, name="train", upscale_factor=4 ):
        super().__init__()
        self.data_path = Path(data_path)
        self.name = name
        self.upscale_factor = upscale_factor
        self.cache_path = self.data_path / "cache"
        if (self.cache_path /  f"div2k_{self.name}_data.npy").exists():
            self.data = np.load(self.cache_path / f"div2k_{self.name}_data.npy")
            self.label = np.load(self.cache_path / f"div2k_{self.name}_label.npy")
        else:
            self.load_images()
        self.data = self.data.astype(np.float32)
        self.label = self.label.astype(np.float32)
        self.transforms = self.build_transforms()
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img, label = self.transforms(self.data[idx], self.label[idx])
        return img, label
    
    
    def load_images(self):
        image_path = self.data_path / "images"
        data_path =  image_path / f"DIV2K_{self.name}_LR_bicubic" / f"X{self.upscale_factor}"
        label_path = image_path / f"DIV2K_{self.name}_HR"
        self.img_files = []
        self.label_files = []
        for file_path in data_path.glob("*.png"):
            file_name = file_path.stem.replace(f"x{self.upscale_factor}", "")
            label_file  = label_path / f"{file_name}.png"
            if label_file.exists():
                self.img_files.append(file_path)
                self.label_files.append(label_file)
        pbar = tqdm(enumerate(self.img_files), total=len(self.img_files), bar_format='{l_bar}{bar:20}{r_bar}')
        lr_imgs, hr_imgs = [], []
        h, w = 0, 0
        for i, img_file in pbar:
            lr_img = np.array(Image.open(img_file).convert("RGB"), dtype=np.uint8)
            hr_img = np.array(Image.open(self.label_files[i]).convert("RGB"), dtype=np.uint8)
            h = max(h, lr_img.shape[0])
            w = max(w, lr_img.shape[1])
            lr_imgs.append(lr_img)
            hr_imgs.append(hr_img)
        
        self.data = np.stack([center_pad(image, h, w) for image in lr_imgs], axis=0)
        self.label = np.stack([center_pad(image, h * 4, w * 4) for image in hr_imgs], axis=0)
        
        np.save(self.cache_path / f"div2k_{self.name}_data.npy", self.data)
        np.save(self.cache_path / f"div2k_{self.name}_label.npy", self.label)
        
            
    def build_transforms(self):
        transforms = T.Compose([
            T.RandomCrop(size=(self.crop_size, self.crop_size)),
            T.Lambda(lambda r: r / 255.),
            T.ToTensor()
        ])
        return transforms
        