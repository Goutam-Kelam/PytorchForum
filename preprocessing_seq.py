import glob
from PIL import Image,ImageOps
import torch
import torchvision.transforms as transforms

def trans(img):
    t = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    img = t(img).view(1,1,224,224)
    return img

images = torch.zeros(1, 1, 224, 224)

for img_file in glob.glob("*.png"):
    img = Image.open(img_file).convert("L")
    img = ImageOps.fit(img,(224,224))
    t_img = trans(img)
    images = torch.cat((images,t_img),0)


images = images[:-1]    