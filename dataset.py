import skimage.io as io
import  h5py
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import random
from random import randint
import cv2


####
from torchvision import transforms

class CDEDataset(Dataset):
    def __init__(self,im_list,root_X,root_y, transform = None, train = True, dilated2 =True):
        self.root_X = root_X
        self.root_y = root_y
        self.im_list= im_list
        self.transform = transform
        self.train = train
        self.dilated2 = dilated2
    
    def __len__(self):
        return len(self.im_list)#check if right len
    
    def __getitem__(self,index):
        image,gt, count = read_data(self.im_list[index], self.root_X, self.root_y, train = self.train, dilated2 = self.dilated2)
        tr = transforms.ToTensor()
        gt = tr(gt)
        if self.transform:
            image = self.transform(image)
        else:
            image = tr(image)

        if self.train:
            return image.float(), gt.float()
        else:
            return image.float(), gt.float(), count

def read_data(image_name,root_X, root_GT, train = True, dilated2 = True):
    image_path = root_X + "/" + image_name
    
    image = Image.open(image_path)

    image = image.convert('RGB')



    gt_path = root_GT + f"/{image_name[:-4]}.h5"
    
    gt = np.asarray(h5py.File(gt_path, 'r')['density'])
    count = np.asarray(h5py.File(gt_path, 'r')['count'])

    gt_shape_dilated2 = (89,66)
    gt_shape_non_dilated = (99,76)

    if train:
        ratio = 0.5
        width, height = image.size
        crop_size = (267,359)

        start = (randint(0,height - crop_size[0]+1),randint(0,width - crop_size[1]+1))
        
        image = image.crop((start[1],start[0],start[1] + crop_size[1],start[0] + crop_size[0]))
        gt = gt[start[0]:(start[0] + crop_size[0]),start[1]:(start[1] + crop_size[1])]
        if random.random()>0.1:
            gt = np.fliplr(gt)
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        if dilated2:
            target = cv2.resize(gt,gt_shape_dilated2,interpolation = cv2.INTER_CUBIC)*(4.0454545*4.0337079)
        else:
            target = cv2.resize(gt,gt_shape_non_dilated,interpolation = cv2.INTER_CUBIC)*(3.5131579*3.6262626)

    else:
        target = gt
    
    #target = cv2.resize(gt,(gt.shape[1]//8,gt.shape[0]//8),interpolation = cv2.INTER_CUBIC)*64
    #target = cv2.resize(gt,(gt.shape[1]//4,gt.shape[0]//4),interpolation = cv2.INTER_CUBIC)*16
    #find better method for above (all in preprocessing?)


    return image, target, count
    
