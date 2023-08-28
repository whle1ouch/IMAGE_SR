from pathlib import Path
from torch.utils.data import Dataset
from torchvision.transforms import transforms as T



class DIVIK(Dataset):
    
    def __init__(self, data_path):
        super().__init__()
        self.data_path = Path(data_path)
        self.img_file = []
        self.label_file = []
        self.transform = self.build_transforms()
    
    
    def build_transforms(self):
        transforms = T.Compose([
            T.Lambda(lambda r: r / 255.),
            T.ToTensor()
        ])
        return transforms
        