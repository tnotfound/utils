import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

def plot_transform(path):
    img = Image.open(path)
    img_transform = transform(img)
    img_transform = np.transpose(img_transform, (1, 2, 0))
    img = np.transpose(img, (1, 2, 0))
    plt.imshow(img_transform)
    

# 前処理内容を設定
transform = transforms.Compose(
    [
     transforms.RandomResizedCrop((512, 512), scale=(0.8,1)),
     transforms.RandomRotation(90),
     transforms.RandomHorizontalFlip(p=0.5),
     transforms.RandomVerticalFlip(p=0.5),
     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0),
     transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

# 画像のパスを設定
path = '/Users/t0721/Desktop/dog.jpg'

# 画像を読み込んで表示
img = Image.open(path)
plt.imshow(img)

# 前処理して画像表示
plot_transform(path)
