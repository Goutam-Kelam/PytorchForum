import concurrent.futures
import glob
from PIL import Image,ImageOps
import torch
import torchvision.transforms as transforms

def trans(img):
    img = Image.open(img).convert("L")
    img = ImageOps.fit(img,(224,224))
    t = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    img = t(img).view(1,1,224,224)
    return img

images = torch.zeros(1, 1, 224, 224)

with concurrent.futures.ProcessPoolExecutor() as executor:
    img_files = glob.glob("*.png")
    for img_file,t_img in zip(img_files, executor.map(trans, img_files)):
        #img = Image.open(img_file).convert("L")
        #img = ImageOps.fit(img,(224,224))
        #t_img = trans(img)
        images = torch.cat((images,t_img),0)

    images = images[:-1]    
