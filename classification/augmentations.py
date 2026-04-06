import random
import torch
import torchvision.transforms.functional as F
from collections import Counter

class NoduleAugmentation:


    def __init__(self, p=0.8):
        self.p = p  

    def __call__(self, img):
        if random.random() > self.p:
            return img

        # 🔹 rotações
        angle = random.choice([ 90, 180])
        img = F.rotate(img, angle)

        # 🔹 flips
        if random.random() < 0.5:
            img = F.hflip(img)

        if random.random() < 0.5:
            img = F.vflip(img)

        # 🔹 ruído leve (gaussiano)
        if random.random() < 0.5:
            tensor = F.to_tensor(img)
            noise = torch.randn_like(tensor) * 0.02 
            tensor = torch.clamp(tensor + noise, 0.0, 1.0)
            img = F.to_pil_image(tensor)

        return img