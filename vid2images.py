import os
import random
import numpy as np
from imutils import paths
from cv2 import cv2
# Remove files in folders before inserting new ones
remove_only = False  # only remove, do not add new

# !-- SETUP---!
angle = False
classify = True
distance = False

augment_cropp = False  # crop for angle
num_samples_per_class = 1
num_augment = 1
# !-- SETUP END---!


def distance_setup():
    folders = [r"dataset_distance\15\\", r"dataset_distance\30\\", r"dataset_distance\45\\",
               r"dataset_distance\60\\", r"dataset_distance\75\\", r"dataset_distance\90\\",
               r"dataset_distance\105\\", r"dataset_distance\120\\", r"dataset_distance\135\\", r"dataset_distance\150\\"]
    for folder in folders:
        filelist = [f for f in os.listdir(folder) if f.endswith(".png")]
        for f in filelist:
            os.remove(os.path.join(folder, f))

    classes = [[r'vids_distance\15.mp4'],
               [r'vids_distance\30.mp4'],
               [r'vids_distance\45.mp4'],
               [r'vids_distance\60.mp4'],
               [r'vids_distance\75.mp4'],
               [r'vids_distance\90.mp4'],
               [r'vids_distance\105.mp4'],
               [r'vids_distance\120.mp4'],
               [r'vids_distance\135.mp4'],
               [r'vids_distance\150.mp4']]
    return classes, folders


def angle_setup():
    folders = [r"dataset_angle\0\\",
               r"dataset_angle\m11.25\\", r"dataset_angle\m22.5\\", r"dataset_angle\m33.75\\",r"dataset_angle\m45\\",
               r"dataset_angle\p11.25\\", r"dataset_angle\p22.5\\", r"dataset_angle\p33.75\\",r"dataset_angle\p45\\", ]
    for folder in folders:
        filelist = [f for f in os.listdir(folder) if f.endswith(".png")]
        for f in filelist:
            os.remove(os.path.join(folder, f))

    classes = [[r'vids_angle\0.mp4'],
               [r'vids_angle\m11.25.mp4'],
               [r'vids_angle\m22.5.mp4'],
               [r'vids_angle\m33.75.mp4'],
               [r'vids_angle\m45.mp4'],
               [r'vids_angle\p11.25.mp4'],
               [r'vids_angle\p22.5.mp4'],
               [r'vids_angle\p33.75.mp4'],
               [r'vids_angle\p45.mp4']]
    return classes, folders


def classify_setup():
    folders = [r'dataset_classify\station\\', r'dataset_classify\nostation\\']
    for folder in folders:
        filelist = [f for f in os.listdir(folder) if f.endswith(".png")]
        for f in filelist:
            os.remove(os.path.join(folder, f))

    classes = [[r'vids_distance\15.mp4', r'vids_distance\30.mp4', r'vids_distance\45.mp4', r'vids_distance\60.mp4', r'vids_distance\75.mp4', r'vids_distance\90.mp4',
               r'vids_distance\105.mp4',r'vids_distance\120.mp4',r'vids_distance\135.mp4', r'vids_distance\150.mp4',r'vids_angle\0.mp4',r'vids_angle\m11.25.mp4',
               r'vids_angle\m22.5.mp4',r'vids_angle\m33.75.mp4',r'vids_angle\p11.25.mp4', r'vids_angle\p22.5.mp4',r'vids_angle\p33.75.mp4'],
               [r'vid_classify_station\nostation_random.mp4',r'vid_classify_station\nostation_wall.mp4',r'vid_classify_station\nostationLeft.mp4',r'vid_classify_station\nostationRight.mp4']]
    return classes, folders


if angle:
    classes, folders= angle_setup()
elif classify:
    classes, folders= classify_setup()
else:
    classes, folders= distance_setup()
num_images_in_each= []
if not remove_only:
    for i, class_vids in enumerate(classes):
        vid= class_vids[0]
        vidcap= cv2.VideoCapture(vid)
        ret, image= vidcap.read()
        image = cv2.flip(image, 0)
        count= 0
        directory= folders[i]
        vid_index= 0
        while ret:
            ret, image= vidcap.read()
            image = cv2.flip(image, 0)
            im_num= count
            if len(str(im_num)) == 1:
                num= '0000000'+str(im_num)
            elif len(str(im_num)) == 2:
                num= '000000'+str(im_num)
            elif len(str(im_num)) == 3:
                num= '00000'+str(im_num)
            try:
                image.shape  # just to cehck if it exist
                cv2.imwrite(directory+num+".png", image)
            except:
                print(vid, 'only contains num frames:', count)
                break
            count += 1
            if count == num_samples_per_class:
                break
            if count >= (vid_index+1)*(num_samples_per_class/(len(class_vids))):
                vid_index += 1
                vid= class_vids[vid_index]
                vidcap= cv2.VideoCapture(vid)
        num_images_in_each.append(count)

if angle and augment_cropp:
    for j, directory in enumerate(folders):
        imagePaths= sorted(list(paths.list_images(directory)))
        random.seed(42)
        random.shuffle(imagePaths)
        last_num= num_images_in_each[j]
        for i, file in enumerate(imagePaths):
            y= 0
            img= cv2.imread(file)
            h= img.shape[0]
            rand_float= np.random.uniform(0.7, 0.95)
            w= int(img.shape[1]*rand_float)
            if i % 2 == 0:
                x= img.shape[1]
                crop_img= img[y:y+h, x-w:x]
            else:
                x= 0
                crop_img= img[y:y+h, x:x+w]
            if len(str(last_num+i)) == 1:
                num= '0000000'+str(last_num+i)
            elif len(str(last_num+i)) == 2:
                num= '000000'+str(last_num+i)
            elif len(str(last_num+i)) == 3:
                num= '00000'+str(last_num+i)
            elif len(str(last_num+i)) == 4:
                num= '0000'+str(last_num+i)
            cv2.imwrite(directory+num+".png", crop_img)
            if i >= num_augment:
                break
            # cv2.imshow("cropped", crop_img)
            # cv2.waitKey(0)
