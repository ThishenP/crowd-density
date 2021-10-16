import  h5py
import skimage.io as io
import  scipy.io
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import glob


def density_map(img_path ,mat_gt_path):
    image = io.imread(img_path)
    
    mat_gt = scipy.io.loadmat(mat_gt_path)
    
    try:
        coords = mat_gt['image_info'][0][0][0][0][0]
        #coords[:,[0, 1]] = coords[:,[1, 0]]
    except KeyError as e:
        coords = mat_gt['annPoints']
    
    
    coords_int = coords.astype(int)

    if len(image.shape)==3:
        gt = np.zeros_like(image[:,:,0])
    else:
        gt = np.zeros_like(image)
    
    try:
        for y,x in coords_int:
            
            gt[x,y] = 1   
    except:
        print(image.shape)
        print(coords_int[:,0].max())
        print(coords_int[:,1].max())

    dense = gaussian_filter(gt.astype(float),sigma =15)
    a = gaussian_filter(np.random.randint(0,100,[20,20]),sigma =15)
    return dense

def preprocess(img_path):
    for file in glob.glob(f"{img_path}/*.jpg"):
        print(file)
        gt_path = file.replace('\\','/').replace('images/','ground-truth/GT_').replace(".jpg",".mat")
        print(gt_path)
        h5 = h5py.File(file.replace('.jpg','.h5').replace('images','density_gt'), 'w')
        h5['density'] = density_map(file, gt_path)
        h5.close()

train_path = "../CDE_Data/train/images"
preprocess(train_path)
#test_path = "ShanghaiTech/ShanghaiTech/part_A/test_data/images"
#preprocess(test_path)