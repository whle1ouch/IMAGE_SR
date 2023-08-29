import numpy as np
from torchvision.transforms import functional as F

def center_pad(img, h, w):
    """
    Center pad an image to a specified size
    img: numpy.NDarray
    """
    ih, iw = img.shape[:2]
    dh, dw = (h - ih) // 2, (w - iw) // 2
    out = np.zeros(shape=(h, w, *img.shape[2:]), dtype=img.dtype)
    out[dh:ih+dh, dw:iw+dw, :] = img
    return out


class RandomCrop:
    
    
    def __init__(self, p):
        self.p = p
        
    def __call__(self, img, label):
        if np.random.rand() > self.p:
            return img, label
        
        

