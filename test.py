import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

path = "./Datasets/image/train/n01491361/n01491361_827.JPEG"
gt = Image.open(path)
plt.figure()
plt.imshow(gt)
dataloaderS = transforms.Compose([transforms.Resize((224,224)),transforms.Grayscale(3),
                                transforms.ToTensor()])
gt = dataloaderS(gt)
#datas = transforms.ToTensor()
#gt = datas(gt)
gt = gt.permute(2,1,0)
print(gt.size())
plt.figure()
plt.imshow(gt,cmap=plt.cm.gray)
plt.pause(10)