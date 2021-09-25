import skimage.io as io
import  h5py
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import random
import cv2

####
from torchvision import transforms

class CDEDataset(Dataset):
    def __init__(self,im_list,root_X,root_y, transform = None, train = True):
        self.root_X = root_X
        self.root_y = root_y
        self.im_list= im_list
        self.transform = transform
        self.train = train
    
    def __len__(self):
        return len(self.root_X)#check if right len
    
    def __getitem__(self,index):
        image,gt = read_data(self.im_list[index], self.root_X, self.root_y, train = self.train)

        tr = transforms.ToTensor()
        gt = tr(gt)
        if self.transform:
            image = self.transform(image)
        else:
            image = tr(image)
        
        return image.float(), gt.float()

def read_data(image_name,root_X, root_GT, train = True):
    image_path = root_X + "/" + image_name
    
    image = Image.open(image_path)

    image = image.convert('RGB')

    gt_path = root_GT + f"/{image_name[:-4]}.h5"
    
    gt = np.asarray(h5py.File(gt_path, 'r')['density'])

    if False:
        ratio = 0.5
        crop_size = (int(image.size[0]*ratio),int(image.size[1]*ratio))
        rdn_value = random.random()
        if rdn_value<0.25:
            dx = 0
            dy = 0
        elif rdn_value<0.5:
            dx = int(image.size[0]*ratio)
            dy = 0
        elif rdn_value<0.75:
            dx = 0
            dy = int(image.size[1]*ratio)
        else:
            dx = int(image.size[0]*ratio)
            dy = int(image.size[1]*ratio)

        image = image.crop((dx,dy,crop_size[0]+dx,crop_size[1]+dy))
        gt = gt[dy:(crop_size[1]+dy),dx:(crop_size[0]+dx)]
        if random.random()>0.8:
            gt = np.fliplr(gt)
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
    
    target = cv2.resize(gt,(gt.shape[1]//8,gt.shape[0]//8),interpolation = cv2.INTER_CUBIC)*64

    return image, target
    
