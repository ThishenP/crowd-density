import numpy as np
import scipy.spatial
import pdb
import h5py
import scipy.io
import skimage.io as io
import numpy as np
import glob
import os

def density_map(img_path ,mat_gt_path):
    image = io.imread(img_path)
    
    mat_gt = scipy.io.loadmat(mat_gt_path)
    
    try:
        coords = mat_gt['image_info'][0][0][0][0][0]
    except KeyError as e:
        coords = mat_gt['annPoints']
    
    
    coords_int = coords.astype(int)

    headcount = coords.shape[0]

    if len(image.shape)==3:
        gt = np.zeros_like(image[:,:,0])
    else:
        gt = np.zeros_like(image)
    
    dense = generate_density_map(shape = gt.shape, points = coords)
    print(dense.shape)
    return dense, headcount

def preprocess_SH(img_path):

    for file in glob.glob(f"{img_path}/*.jpg"):
        print(file)
        gt_path = file.replace('\\','/').replace('images/','ground-truth/GT_').replace(".jpg",".mat")
        print(gt_path)
        h5 = h5py.File(file.replace('.jpg','.h5').replace('images','density_gt'), 'w')
        density, count =  density_map(file, gt_path)
        h5['density'] = density
        h5['count'] = count
        
        h5.close()

def preprocess_UCF(path):
    print(path)
    for file in glob.glob(f"{path}/*.jpg"):
        print(file)
        gt_path = file.replace('\\','/').replace('images/','ground-truth/').replace(".jpg","_ann.mat")
        print(gt_path)
        h5 = h5py.File(file.replace('.jpg','.h5').replace('images','density_gt'), 'w')
        density, count =  density_map(file, gt_path)
        h5['density'] = density
        h5['count'] = count
        
        h5.close()
       
def generate_density_map(shape=(5,5),points=None,f_sz=15,sigma=4):
    """
    generate density map given head coordinations
    """
    im_density = np.zeros(shape[0:2])
    h, w = shape[0:2]
    if len(points) == 0:
        return im_density
    for j in range(len(points)):
        H = matlab_style_gauss2D((f_sz,f_sz),sigma)
        x = np.minimum(w,np.maximum(1,np.abs(np.int32(np.floor(points[j,0])))))
        y = np.minimum(h,np.maximum(1,np.abs(np.int32(np.floor(points[j,1])))))
        if x>w or y>h:
            continue
        x1 = x - np.int32(np.floor(f_sz/2))
        y1 = y - np.int32(np.floor(f_sz/2))
        x2 = x + np.int32(np.floor(f_sz/2))
        y2 = y + np.int32(np.floor(f_sz/2))
        dfx1 = 0
        dfy1 = 0
        dfx2 = 0
        dfy2 = 0
        change_H = False
        if x1 < 1:
            dfx1 = np.abs(x1)+1
            x1 = 1
            change_H = True
            print("1")
        if y1 < 1:
            dfy1 = np.abs(y1)+1
            y1 = 1
            change_H = True
            print("2")
        if x2 > w:
            dfx2 = x2 - w
            x2 = w
            change_H = True
            print("3")
        if y2 > h:
            dfy2 = y2 - h
            y2 = h
            change_H = True
            print("1")
        x1h = 1+dfx1
        y1h = 1+dfy1
        x2h = f_sz - dfx2
        y2h = f_sz - dfy2
        if change_H:
            H =  matlab_style_gauss2D((y2h-y1h+1,x2h-x1h+1),sigma)
        im_density[y1-1:y2,x1-1:x2] = im_density[y1-1:y2,x1-1:x2] +  H;
    return im_density
     
def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


# train_path = "../CDE_Data/train/images"
# preprocess_SH(train_path)

# val_path = "../CDE_Data/val/images"
# preprocess_SH(val_path)

test_path_SH_A = "../CDE_Data/ShanghaiTech/ShanghaiTech/part_A/test_data/images"
preprocess_SH(test_path_SH_A)

test_path_SH_B = "../CDE_Data/ShanghaiTech/ShanghaiTech/part_B/test_data/images"
preprocess_SH(test_path_SH_B)

test_path_UCF = "../CDE_Data/UCF_CC_50/images"
preprocess_UCF(test_path_UCF)
