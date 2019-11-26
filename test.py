from cv2 import cv2
import numpy as np
from imutils import paths
import random
folder=r'dataset_angle\0'
imagePaths = sorted(list(paths.list_images(folder)))
random.seed(42)
random.shuffle(imagePaths)
num=200
for i,file in enumerate(imagePaths):
    y=0
    img = cv2.imread(file)
    h=img.shape[0]
    rand_int=np.random.uniform(0.6,0.95)
    w=int(img.shape[1]*rand_int)
    if i % 2==0:    
        x=img.shape[1]
        crop_img = img[y:y+h, x-w:x]
    else:
        x=0
        crop_img = img[y:y+h, x:x+w]
    if len(str(num+i)) == 1:
        num = '0000000'+str(num+i)
    elif len(str(num+i)) == 2:
        num = '000000'+str(num+i)
    elif len(str(num+i)) == 3:
        num = '00000'+str(num+i)
    elif len(str(num+i)) == 4:
        num = '0000'+str(num+i)
    # cv2.imshow("cropped", crop_img)
    # cv2.waitKey(0)